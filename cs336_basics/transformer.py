import torch
import torch.nn as nn
from einops import einsum, rearrange
import numpy as np

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = np.sqrt(2/(in_features + out_features)).item()
        self.param = nn.Parameter(torch.torch.nn.init.trunc_normal_(W, std=std, a=-std, b=std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.param, x, "out_features in_features, ... in_features -> ... out_features")
    
class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        A = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.param = nn.Parameter(torch.nn.init.trunc_normal_(A, a=-3, b=3))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.param[token_ids]

class RMSNorm(nn.Module): 

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.param = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        d_model = self.d_model
        denoms = torch.sqrt(einsum(x.square(), "... d_model -> ... ")/d_model + self.eps)
        result = einsum(x*self.param, 1./denoms , "... d_model, ... -> ... d_model")
        return result.to(in_dtype)
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None=None, device=None, dtype=None):
        super().__init__()
        print("d_ff", d_ff)
        if d_ff == None:
            print("d_ff", d_ff)           
            d_ff = int(np.ceil(8*d_model // 3 / 64) * 64)
        self.W1 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def _silu(self, param: torch.tensor): 
        return param*torch.sigmoid(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_1x = self.W1.forward(x)
        W_3x = self.W3.forward(x)
        result = self.W2.forward(self._silu(W_1x)*W_3x)
        return result
    
class Rope(nn.Module): 

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        i_vec = torch.arange(max_seq_len, device=device)[:, None]
        k_vec = torch.arange(d_k//2, device=device)[None, :]
        thetas = i_vec / theta ** (2*k_vec/d_k)
        # Typo in the assignment. There it says that k in {1, ..., d/2}.
        # Can either view as sin/cos or as complex
        R = torch.stack((thetas.cos(), thetas.sin()))
        # Complex version: 
        # R = torch.polar(torch.ones_like(thetas), thetas)
        R.to(device=device)
        self.register_buffer("R", R, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        # Basic version
        even = x[...,token_positions,::2]
        odd = x[...,token_positions,1::2]
        c = self.R[0,token_positions, ...]
        s = self.R[1,token_positions,...] 
        tmp = even * s + odd * c      
        x[...,token_positions,::2] = even * c - odd *s
        x[...,token_positions,1::2] = tmp
        ## Complex version:
        # z = rearrange(x[...,token_positions,:], "... (d two) -> ... d two", two=2)
        # z=torch.view_as_complex(z)
        # print(self.R.shape, z.shape)
        # z.mul_(self.R[token_positions,:])
        # x[...,token_positions,:] = rearrange(torch.view_as_real(z), "... d two -> ... (d two)")
        
        return x
    
def softmax(x:torch.Tensor, dim: int):
    m = torch.max(x,dim=dim, keepdim=True)
    x_exp = torch.exp(x - torch.broadcast_to(m.values, x.shape))
    sums = torch.sum(x_exp, dim = dim, keepdim=True)
    return x_exp / torch.broadcast_to(sums, x_exp.shape)

def scaled_dot_product_attention(Q:torch.Tensor, K: torch.Tensor, V, mask = None):
    d_k = Q.shape[-1]
    QKT = einsum(Q, K, "batch_size ... queries d_k, batch_size ... keys d_k -> batch_size ... queries keys")
    QKT.mul_(1/np.sqrt(d_k))
    softmax_dim = len(QKT.shape) - 1
    if mask != None:
        result = torch.where(mask,
        0,
        -float('inf'))
        A = softmax(QKT + result,dim = softmax_dim)
    else: 
        A = softmax(QKT,dim = softmax_dim)
    return einsum(A, V, "... crud seq_length, ... seq_length d_v -> ... crud d_v")

class multihead_self_attention(nn.Module): 
    def __init__(self, d_model:int, num_heads:int, max_seq_length:None, theta:None, device=None, dtype=None):
        super().__init__()

        self.W_QKV = Linear(3*d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model,device=device, dtype=dtype)
        self.d_model = d_model
        self.num_heads = num_heads
        # if max_seq_length != None and theta != None:
        self.R = Rope(theta=theta, max_seq_len=max_seq_length, d_k=d_model//num_heads, device=device)


    def forward(self, X:torch.tensor, token_positions = None):
        QKV = self.W_QKV.forward(X)
        QKV = rearrange(QKV, "batch_size seq_length (three num_heads d_head) -> three num_heads batch_size seq_length d_head", three = 3, num_heads = self.num_heads)
        seq_length = QKV.shape[-2] # need to change to length of token positions
        if token_positions == None:
            token_positions = torch.arange(seq_length, device=QKV.device)
        QKV[:2, :] = self.R.forward(QKV[:2, :], token_positions=token_positions)
        cmask = torch.ones((seq_length,seq_length), dtype=torch.bool).tril()
        # may need to squeeze here, not sure.
        A = scaled_dot_product_attention(QKV[0, :], QKV[1,:], QKV[2, :], mask=cmask)
        # print(A.shape, self.W_O.param.shape)
        A = rearrange(A, "num_heads batch_size seq_length d_head -> batch_size seq_length (num_heads d_head)")
        out = self.W_O.forward(A)
        return out
        
class transformer_block(nn.Module): 
    def __init__(self, d_model:int, num_heads:int, d_ff:int = None, max_seq_length:int = None, theta:int = None, device=None, dtype=None):
        super().__init__()

        self.MHA = multihead_self_attention(num_heads=num_heads, d_model=d_model, max_seq_length=max_seq_length, theta=theta, device=device, dtype=dtype)
        self.FFN = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, device=device)
        self.RMSNorm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.RMSNorm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, X:torch.tensor):
        Y = X + self.MHA(self.RMSNorm1.forward(X))
        Z = Y + self.FFN.forward(self.RMSNorm2(Y))
        return Z
    
class transformer_lm(nn.Module): 
    def __init__(self, d_model:int, num_heads:int, vocab_size:int, context_length: int, num_layers: int, d_ff:int = None, theta:int = None, device=None, dtype=None):
        super().__init__()
        self.Embedding = Embedding(num_embeddings=vocab_size,embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = []
        for _ in range(num_layers):
            TB = transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_length=context_length, theta=theta, device=device,dtype=dtype)
            self.layers.append(TB)
        self.final_RMSNorm = RMSNorm(d_model=d_model, device=device,dtype=dtype)
        self.output_layer = Linear(out_features=vocab_size, in_features=d_model, device=device,dtype=dtype)

    def forward(self, X: torch.tensor):
        X = self.Embedding.forward(X)
        for layer in self.layers:
            X = layer(X)
        X = self.final_RMSNorm.forward(X)
        X = self.output_layer(X)
        return X
        # return softmax(X, dim=0) model does not include softmax.
