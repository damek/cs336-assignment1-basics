import torch
# from torch.optim.optimizer import optimizer
from collections.abc import Callable, Iterable
from typing import Optional
import math
import os
import typing

import torch.optim.optimizer

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr = 1e-3, betas = (.9, .999), eps = 1e-8, weight_decay = 1e-2):
        if lr < 0 or betas[0] < 0 or betas[1] < 0 or weight_decay < 0 or eps < 0:
            raise ValueError(f"Invalid, negatove hyperparam") 
        defaults = {'lr' : lr, 'betas' : betas, 'eps' : eps, 'lambda_wd' : weight_decay}
        super().__init__(params,defaults)
        
    def set_lr(self, lr):
        """Update learning rate for all parameter groups"""
        for group in self.param_groups:
            group['lr'] = lr

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr'] 
            beta_1, beta_2 = group['betas']
            eps = group['eps']
            lambda_wd = group['lambda_wd']
            for p in group["params"]:
                if p.grad is None: 
                    continue
                
                state = self.state[p]
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p)
                if 'v' not in state:
                    state['v'] = torch.zeros_like(p)
                if 't' not in state:
                    state['t'] = 1
                
                m = state['m']
                v = state['v']
                t = state['t']
                grad = p.grad

                # m = beta_1*m + (1-beta_1)*grad
                m.mul_(beta_1).add_(grad, alpha = 1-beta_1)
                # v = beta_2*v + (1-beta_2)*grad.square()
                v.mul_(beta_2).addcmul_(grad, grad, value = 1-beta_2)
                alpha_t = lr*math.sqrt(1-math.pow(beta_2,t))/(1-math.pow(beta_1, t))
                # p.data -= alpha_t * m.div(v.sqrt() + eps)
                denom = v.sqrt().add(eps)
                p.data.addcdiv_(m, denom, value = -alpha_t)
                # p.data -= lr*lambda_wd*p.data
                p.data.mul_(1-lr*lambda_wd)

                state['m'] = m
                state['v'] = v
                state['t'] = t + 1


        return loss

def learning_rate_schedule(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_final_iter: int):

    if it < warmup_iters: 
        return (it/warmup_iters)*max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_final_iter:
        return min_learning_rate + .5*(1 + math.cos((it - warmup_iters)*math.pi/(cosine_cycle_final_iter - warmup_iters)))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate
    

def gradient_clipping(params: list[torch.tensor], max_l2_norm, eps = 1e-6):
    for param in params:
        if param.grad is None:
            continue
        norm = param.grad.norm()
        param.grad.mul_(max_l2_norm/(norm + eps))
    
def save_checkpoint(model : torch.nn.Module, optimizer: torch.optim.Optimizer, iteration : int , out : str | os.PathLike | typing.BinaryIO | typing.IO[bytes], args = None, ema_loss= None, valid_loss=float('inf')):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    obj = {'model_state' : model_state, 'optimizer_state' : optimizer_state, 'iteration' : iteration, 'args' : args, 'ema_loss' : ema_loss, 'valid_loss' : valid_loss}
    torch.save(obj, out)


import transformer
def load_checkpoint(    
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model = None,
    optimizer = None):

    obj = torch.load(src, weights_only=False)
    iter = obj['iteration']
    args = obj.get('args', None)
    if model == None and args == None:
        raise ValueError("Either model or args must be provided")
    if optimizer == None and args == None:
        raise ValueError("Either optimizer or args must be provided")
    if model == None:
        model = transformer.transformer_lm(vocab_size=args.vocab_size,d_ff=args.d_ff, d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers, context_length=args.context_length, theta=args.rope_theta_parameter, device=args.device, pre_RMS=True, post_RMS=False)
    if optimizer == None:
        optimizer = AdamW(model.parameters(), betas = args.betas, eps = args.eps, weight_decay=args.weight_decay)
    model.load_state_dict(obj['model_state'])
    optimizer.load_state_dict(obj['optimizer_state'])
    ema_loss = obj.get('ema_loss', None)
    valid_loss = obj.get('valid_loss', float('inf'))
    return model, optimizer, iter, args, ema_loss, valid_loss