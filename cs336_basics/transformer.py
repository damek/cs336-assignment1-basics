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
        # print("shape of token_ids", token_ids.shape, "shape of params", self.param[token_ids].shape)
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
        norm = x.square().mean(dim=-1, keepdim=True)
        result = (x*torch.rsqrt(norm + self.eps))*self.param
        # denoms = torch.sqrt(einsum(x.square(), "... d_model -> ... ")/d_model + self.eps)
        # result = einsum(x*self.param, 1./denoms , "... d_model, ... -> ... d_model")
        return result.to(in_dtype)

def SiLU(param: torch.tensor): 
    return param*torch.sigmoid(param)    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None=None, device=None, dtype=None):
        super().__init__()
        if d_ff == None:
            # print("d_ff", d_ff)           
            d_ff = int(np.ceil(8*d_model // 3 / 64) * 64)
        self.W1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.W2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.W3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_1x = self.W1.forward(x)
        W_3x = self.W3.forward(x)
        result = self.W2.forward(SiLU(W_1x)*W_3x)
        return result
 
    
class PositionwiseFeedForwardSiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None=None, device=None, dtype=None):
        super().__init__()
        if d_ff == None:
            # print("d_ff", d_ff)           
            d_ff = 4*d_model
        self.W1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.W2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_1x = self.W1.forward(x)
        result = self.W2.forward(SiLU(W_1x))
        return result

class Rope(nn.Module): 

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        # self.token_positions_default = torch.arrange(max_seq_len, device=device)
        if theta != 0:
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
        if self.theta != 0:
            if token_positions != None:
                even = x[...,token_positions,::2]
                odd = x[...,token_positions,1::2]
                c = self.R[0,token_positions, ...]
                s = self.R[1,token_positions,...] 
                tmp = even * s + odd * c      
                x[...,token_positions,::2] = even * c - odd *s
                x[...,token_positions,1::2] = tmp
            else:
                even = x[...,::2]
                odd = x[...,1::2]
                c = self.R[0, ...]
                s = self.R[1,...] 
                tmp = even * s + odd * c      
                x[...,::2] = even * c - odd *s
                x[...,1::2] = tmp

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
    QKT.div_(np.sqrt(d_k))
    softmax_dim = len(QKT.shape) - 1
    seq_length = Q.shape[-2]
    if mask != None:
        result = torch.where(mask[:seq_length,:seq_length],
        0,
        -float('inf'))
        A = softmax(QKT + result,dim = softmax_dim)
    else: 
        A = softmax(QKT,dim = softmax_dim)
    return einsum(A, V, "... crud seq_length, ... seq_length d_v -> ... crud d_v")

class multihead_self_attention(nn.Module): 
    def __init__(self, d_model:int, num_heads:int, max_seq_length:None, theta:None, device=None, dtype=None):
        super().__init__()

        self.W_QKV = Linear(d_model, 3*d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model,device=device, dtype=dtype)
        self.d_model = d_model
        self.num_heads = num_heads
        # if max_seq_length != None and theta != None:
        self.R = Rope(theta=theta, max_seq_len=max_seq_length, d_k=d_model//num_heads, device=device)
        self.cmask = torch.ones((max_seq_length,max_seq_length), dtype=torch.bool, device=device).tril()

    def forward(self, X:torch.tensor, token_positions = None):
        QKV = self.W_QKV.forward(X)
        QKV = rearrange(QKV, "batch_size seq_length (three num_heads d_head) -> three num_heads batch_size seq_length d_head", three = 3, num_heads = self.num_heads)
        seq_length = QKV.shape[-2] # need to change to length of token positions
        # if token_positions == None:
        #     token_positions = self.token_positions_default
        QKV[:2, :] = self.R.forward(QKV[:2, :], token_positions=token_positions)
        # may need to squeeze here, not sure.
        # cmask = torch.ones((seq_length,seq_length), dtype=torch.bool).tril()
        A = scaled_dot_product_attention(QKV[0, :], QKV[1,:], QKV[2, :], mask=self.cmask)
        # print(A.shape, self.W_O.param.shape)
        A = rearrange(A, "num_heads batch_size seq_length d_head -> batch_size seq_length (num_heads d_head)")
        out = self.W_O.forward(A)
        return out
        
class transformer_block(nn.Module): 
    def __init__(self, d_model:int, num_heads:int, d_ff:int = None, max_seq_length:int = None, theta:int = None, pre_RMS = True, post_RMS = False, activation = "", device=None, dtype=None):
        super().__init__()

        self.MHA = multihead_self_attention(num_heads=num_heads, d_model=d_model, max_seq_length=max_seq_length, theta=theta, device=device, dtype=dtype)
        if activation.lower() == "silu":
            print("Using SiLU activation")
            self.FFN = PositionwiseFeedForwardSiLU(d_model=d_model, d_ff=d_ff, device=device)
        else:
            self.FFN = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, device=device)
        self.pre_RMS = pre_RMS
        self.post_RMS = post_RMS
        assert not (pre_RMS and post_RMS), "pre_RMS and post_RMS cannot both be True"
        self.RMSNorm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.RMSNorm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)


    def forward(self, X:torch.tensor):
        if self.pre_RMS:
            Y = X + self.MHA(self.RMSNorm1.forward(X))
            Z = Y + self.FFN.forward(self.RMSNorm2(Y))
            return Z
        if self.post_RMS:
            Y = self.RMSNorm1.forward(X + self.MHA(X))
            Z = self.RMSNorm2(Y + self.FFN(Y))
            return Z
        else:
            Y = X + self.MHA(X)
            Z = Y + self.FFN(Y)
            return Z
        # Y = X + self.MHA(self.RMSNorm1.forward(X))
        # Z = Y + self.FFN.forward(self.RMSNorm2(Y))
        # return Z
    
class transformer_lm(nn.Module): 
    def __init__(self, d_model:int, num_heads:int, vocab_size:int, context_length: int, num_layers: int, d_ff:int = None, theta:int = None, pre_RMS = True, post_RMS = False, activation = "", device=None, dtype=None):
        super().__init__()
        self.Embedding = Embedding(num_embeddings=vocab_size,embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            TB = transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_length=context_length, theta=theta, device=device,dtype=dtype, pre_RMS = pre_RMS, post_RMS = post_RMS, activation = activation)
            self.layers.append(TB)
        self.final_RMSNorm = RMSNorm(d_model=d_model, device=device,dtype=dtype)
        self.output_layer = Linear(out_features=vocab_size, in_features=d_model, device=device,dtype=dtype)
        self.pre_RMS =  pre_RMS
        self.post_RMS = post_RMS

    def forward(self, X: torch.tensor):
        X = self.Embedding.forward(X)
        for layer in self.layers:
            X = layer(X)
        if self.pre_RMS or self.post_RMS:
            X = self.final_RMSNorm.forward(X)
        X = self.output_layer(X)
        return X
        # return softmax(X, dim=0) model does not include softmax.

# I am assuming there will only be one batch dimension here. Otherwise targets needs more care.
def cross_entropy(logits: torch.tensor, targets):
    logits = rearrange(logits, "b c ... -> (b c) ...")
    targets = rearrange(targets, "b c ... -> (b c) ...")
    m = torch.max(logits,dim=-1, keepdim=True)
    subm = logits - torch.broadcast_to(m.values, logits.shape)
    x_exp = torch.exp(subm)
    sums = torch.sum(x_exp, dim =-1, keepdim=True)
    result = torch.gather(subm, 1, targets.unsqueeze(1)).sum()/len(targets)
    diff = result - torch.mean(torch.log(sums))
    return -diff

import tokenizer_utils
def decode(prompt : str, model, tokenizer:tokenizer_utils.Tokenizer, max_tokens=1, temperature=1, p=None, device=None):
    token_list = tokenizer.encode(prompt)
    eot_id = tokenizer.eot_id
    token_count = 0
    next_token = token_list[-1]
    while next_token != eot_id and token_count < max_tokens:
        tensor_view = torch.as_tensor(token_list).reshape((1, len(token_list))).to(device)
        next_token_distribution = softmax(model(tensor_view).squeeze()/temperature, dim=0)
        probs_sorted, count, indices = top_p(next_token_distribution[-1,:], p)
        next_id_idx = torch.multinomial(input=probs_sorted[:count], num_samples=1)
        next_id = indices[next_id_idx]
        token_list.append(next_id.item())
        next_token = next_id
        token_count += 1
    return tokenizer.decode(token_list)


def top_p(distribution, p=None):
    if p == None:
        return distribution, len(distribution), torch.arange(len(distribution), device=distribution.device)
    probs_sorted, indices = torch.sort(distribution, descending=True)
    mass = 0
    count = 0
    while count < len(distribution) and mass <= p:
        mass += probs_sorted[count]
        count += 1

    return probs_sorted, count, indices





        
        
        