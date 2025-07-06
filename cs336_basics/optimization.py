import torch
# from torch.optim.optimizer import optimizer
from collections.abc import Callable, Iterable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr = 1e-3, betas = (.9, .999), eps = 1e-8, weight_decay = 1e-2):
        if lr < 0 or betas[0] < 0 or betas[1] < 0 or weight_decay < 0 or eps < 0:
            raise ValueError(f"Invalid, negatove hyperparam") 
        defaults = {'lr' : lr, 'betas' : betas, 'eps' : eps, 'lambda_wd' : weight_decay}
        super().__init__(params,defaults)

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
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))
                t = state.get('t', 1)
                grad = p.grad

                m = beta_1*m + (1-beta_1)*grad
                v = beta_2*v + (1-beta_2)*grad.square()
                alpha_t = lr*math.sqrt(1-math.pow(beta_2,t))/(1-math.pow(beta_1, t))
                p.data -= alpha_t * m.div(v.sqrt() + eps)
                p.data -= lr*lambda_wd*p.data
                state['m'] = m
                state['v'] = v
                state['t'] = t + 1

        return loss

            

def learning_rate_schedule(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int):

    if it < warmup_iters: 
        return (it/warmup_iters)*max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + .5*(1 + math.cos((it - warmup_iters)*math.pi/(cosine_cycle_iters - warmup_iters)))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate
    

def gradient_clipping(params: list[torch.tensor], max_l2_norm, eps = 1e-6):
    for param in params:
        norm = param.grad.norm()
        param.grad.mul_(max_l2_norm/(norm + eps))
    
