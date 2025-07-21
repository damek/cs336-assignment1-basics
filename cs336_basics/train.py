#!/usr/bin/env python3
import os
import sys

# Set cache directories
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_cache'
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
os.environ['XDG_CACHE_HOME'] = '/tmp/cache'

# Create directories
os.makedirs('/tmp/torchinductor_cache', exist_ok=True)
os.makedirs('/tmp/torch_cache', exist_ok=True)
os.makedirs('/tmp/cache', exist_ok=True)

# Patch getpass.getuser to return a dummy username
import getpass
original_getuser = getpass.getuser
def patched_getuser():
    try:
        return original_getuser()
    except KeyError:
        return "cs336user"

getpass.getuser = patched_getuser
import torch
if torch.cuda.is_available(): 
    torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems, stolen from the modded nano gpt repo.
import math
import transformer, optimization, tokenizer_utils
import numpy as np
import wandb, time
import pathlib
import configs
import cProfile
import pstats




def log_layerwise_adamw_updates(optimizer, model, step, log_freq=100):
    """Log per-layer histograms of AdamW update directions: m/(sqrt(v) + eps)"""
    if step % log_freq != 0:
        return
    
    # Create mapping from parameter object to layer name
    param_to_layer = {}
    for name, param in model.named_parameters():
        # Extract clean layer name (e.g., "layers.0.MHA.W_QKV" -> "layers.0.MHA")
        layer_parts = name.split('.')
        if len(layer_parts) >= 3:
            layer_name = '.'.join(layer_parts[:-1])  # Remove the final parameter name
        else:
            layer_name = layer_parts[0] if layer_parts else name
        
        param_to_layer[id(param)] = layer_name
    
    # Collect update directions grouped by layer
    layer_updates = {}
    
    for group in optimizer.param_groups:
        eps = group['eps']  # Get epsilon from optimizer config
        
        for param in group['params']:
            if param.grad is None:
                continue
                
            state = optimizer.state[param]
            if 'm' not in state or 'v' not in state:
                continue
            
            layer_name = param_to_layer.get(id(param), 'unknown')
            
            # Compute the actual update direction: m / (sqrt(v) + eps)
            m = state['m']
            v = state['v']
            update_direction = m / (torch.sqrt(v) + eps)
            
            if layer_name not in layer_updates:
                layer_updates[layer_name] = []
            
            # Collect flattened update directions for this layer
            layer_updates[layer_name].append(update_direction.flatten())
    
    # Create histograms for each layer
    log_dict = {}
    
    for layer_name, updates in layer_updates.items():
        if len(updates) > 0:
            # Concatenate all parameters in this layer
            layer_update_directions = torch.cat(updates)
            
            # Create histogram of update directions
            log_dict[f"adamw_updates/{layer_name}/update_directions"] = wandb.Histogram(layer_update_directions.cpu().numpy())
            
            # Add useful summary statistics
            log_dict[f"adamw_updates/{layer_name}/update_norm"] = layer_update_directions.norm().item()
            log_dict[f"adamw_updates/{layer_name}/update_mean"] = layer_update_directions.mean().item()
            log_dict[f"adamw_updates/{layer_name}/update_std"] = layer_update_directions.std().item()
            log_dict[f"adamw_updates/{layer_name}/update_max"] = layer_update_directions.max().item()
            log_dict[f"adamw_updates/{layer_name}/update_min"] = layer_update_directions.min().item()
            
            # Useful additional metrics
            log_dict[f"adamw_updates/{layer_name}/update_abs_mean"] = layer_update_directions.abs().mean().item()
            log_dict[f"adamw_updates/{layer_name}/num_positive"] = (layer_update_directions > 0).sum().item()
            log_dict[f"adamw_updates/{layer_name}/num_negative"] = (layer_update_directions < 0).sum().item()
    
    if log_dict:
        wandb.log(log_dict, step=step)

@torch.no_grad
def eval(windowed_validation : torch.Tensor, loss_fn, args):
    num_windows = windowed_validation.shape[0]
    nb_batches = math.ceil(num_windows / args.batch_size)
    loss = 0
    total_tokens = 0
    print(f"Evaluating {nb_batches} batches")
    for i in range(nb_batches):
        if i % 100 == 0:
            print(f"Batch {i} of {nb_batches}")
        # make sure to multiply loss by batch size*context_length
        start_window = args.batch_size*i
        end_window = min((i+1)*args.batch_size, num_windows)
        chunk_size = end_window - start_window # could be smaller than batch_size
        batched_windows = windowed_validation[start_window:end_window]
        data = batched_windows[:, :-1].to(args.device)
        targets = batched_windows[:, 1:].to(args.device)
        loss_batch = loss_fn(data, targets)
        num_tokens_in_batch = args.context_length*chunk_size
        loss += loss_batch*num_tokens_in_batch    
        total_tokens += num_tokens_in_batch

    return loss/total_tokens


def save_validation_loss(windowed_validation, loss_fn, args, time_total, iter, best_validation_loss, ema_loss, model, optimizer):
    print("Validating")
    # model.eval()
    valid_loss = eval(windowed_validation, loss_fn, args)
    # model.train()
    wandb.log({
    "Validation loss": valid_loss,
    "wall_time"      : time_total,       
    }, step=iter)

    print("Iteration:", iter, "validation loss:", valid_loss)
    if valid_loss < best_validation_loss:
        best_validation_loss = valid_loss
    ckpt_path = pathlib.Path(f"{args.checkpoint_path}ckpt_best.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)   
    optimization.save_checkpoint(model=model, optimizer=optimizer, iteration = iter, out=ckpt_path, args=args, ema_loss=ema_loss, valid_loss = valid_loss)  

def main():
    cfg, _ = configs.load_cfg() 
    group=cfg.__class__.__name__ 
    purpose = cfg.wandb_project + "_" + cfg.wandb_base_name.split()[0]
    fmt_args  = {**cfg.__dict__, "group": group}
    run_name  = cfg.wandb_base_name.format(**fmt_args) 
    purpose   = f"{cfg.wandb_project}_{run_name}"      
    timestamped_name = f"{run_name}_{time.strftime('%Y%m%d_%H%M')}"
    run_dir   = pathlib.Path(f"checkpoints/{purpose}")
    cfg.checkpoint_path = str(run_dir) + "/"   # ensure trailing slash
    args = cfg
    run = wandb.init(project=args.wandb_project, group=group, name=timestamped_name, config=args)
    wandb.run.log_code(".")
    print("Training with args", args)
    train = torch.as_tensor(np.memmap(args.train_data,dtype=np.uint16,mode="r"), dtype=torch.long, device=args.device)
    validation = torch.as_tensor(np.memmap(args.val_data,dtype=np.uint16,mode="r"), dtype=torch.long, device=args.device)
    args.valid_size = validation.size
    windowed_validation = validation.unfold(0, args.context_length + 1, args.context_length)
    if args.device == "cuda" and torch.cuda.is_available():
        args.device = "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"
    print("device", args.device)
    model = transformer.transformer_lm(vocab_size=args.vocab_size,d_ff=args.d_ff, d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers, context_length=args.context_length, theta=args.rope_theta_parameter, device=args.device, pre_RMS=args.pre_RMS, post_RMS=args.post_RMS, activation=args.activation)
    model.to(args.device)
    # # Compile model (note this changes the names of params to include _orig..., so you need to compile again before loading the checkpoint).
    # Note that compile misbehaves on mps. 
    if args.compile and args.device == "cuda":
        import torch._inductor.config as config
        config.max_autotune = False
        config.force_disable_caches = False  # Keep caching for speed
        config.max_autotune = False
        torch.set_float32_matmul_precision('high')
        
        # torch.backends.cudnn.benchmark = True            # Optimize for fixed input sizes
        # These are the attributes that actually exist:
        if hasattr(config, 'triton'):
            if hasattr(config.triton, 'autotune_pointwise'):
                config.triton.autotune_pointwise = False
        
        backend = "inductor"
        # model = torch.compile(model, backend=backend, mode="reduce-overhead")
        @torch.compile(backend="inductor", mode="reduce-overhead")
        def training_step(model, data, targets):
            loss = transformer.cross_entropy(model(data), targets)
            loss.backward()
            return loss
        @torch.compile(backend="inductor", mode="reduce-overhead")
        def loss_fn(data, targets):
            model.eval()
            with torch.no_grad():
                loss = transformer.cross_entropy(model(data), targets)
            return loss
        
    else: 
        def training_step(model, data, targets):
            loss = transformer.cross_entropy(model(data), targets)
            loss.backward()
            return loss
        def loss_fn(data, targets):
            model.eval()
            with torch.no_grad():
                loss = transformer.cross_entropy(model(data), targets)
            return loss

    optimizer = optimization.AdamW(model.parameters(), betas = args.betas, eps = args.eps, weight_decay=args.weight_decay)
    print("Weight decay", args.weight_decay)
    current_iter = 0
    ema_loss = 0
    best_validation_loss = float('inf')
    if args.resume_from != None: 
        model, optimizer, current_iter, args_old, ema_loss, best_validation_loss = optimization.load_checkpoint(src=args.resume_from, model=model, optimizer=optimizer)
        print(f"Loading from checkpoint: iteration {current_iter}, ema_loss {ema_loss}, best_validation_loss {best_validation_loss}")
        # Test with a tiny subset first
        # model.eval()
    if ema_loss == None:
        ema_loss = 0
    if args.lr_scheduler == "cosine":
        print("Using cosine learning rate scheduler")
        optimizer.set_lr(optimization.learning_rate_schedule(current_iter, args.cosine_decay["max_lr"], args.cosine_decay["min_lr"], args.cosine_decay["warmup_steps"], args.cosine_decay["cosine_cycle_final_iter"]))
    else: 
        print("Using constant learning rate scheduler:lr", args.lr)
        optimizer.set_lr(args.lr)

    # if args.validation_every == None: 
    #     args.validation_every = 100

    lambda_ema = .98
    wandb.watch(model, log="all", log_freq=100)
    time_total = 0
    print("validation_every", args.validation_every)
    print("print_every", args.print_every)

    print("Starting training")
    
    for iter in range(current_iter, args.run_until_step):
        time_start = time.perf_counter()
        data, targets = tokenizer_utils.data_from_gpu_tensor(train, batch_size=args.batch_size, context_length=args.context_length)

        if args.lr_scheduler == "cosine":
            optimizer.set_lr(optimization.learning_rate_schedule(iter, args.cosine_decay["max_lr"], args.cosine_decay["min_lr"], args.cosine_decay["warmup_steps"], args.cosine_decay["cosine_cycle_final_iter"]))

        optimizer.zero_grad()
        loss = training_step(model, data, targets)
        with torch.no_grad():
            print("output_layer.param.grad before", model.output_layer.param.grad)
            model.output_layer.param.grad.data.mul_(iter*args.d_model) # mup?
            print("output_layer.param.grad after", model.output_layer.param.grad)
        if args.grad_clip is not None:
            optimization.gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        ema_loss = (1-lambda_ema) * loss.detach() + lambda_ema*ema_loss
        time_end = time.perf_counter()  
        time_total += time_end - time_start

        time_limit_hours = args.time_limit_hours
        if isinstance(time_limit_hours, list):
            time_limit_hours = time_limit_hours[0]
            
        if time_limit_hours is not None and time_total > time_limit_hours * 3600:
            print(f"Time limit reached {time_total/3600} hours, stopping training")
            save_validation_loss(windowed_validation, loss_fn, args, time_total, iter, best_validation_loss, ema_loss, model, optimizer)
            break
        if iter % args.print_every == 0:
            print(f"Iteration: {iter}, ema loss {ema_loss}, lr {optimizer.param_groups[0]['lr']}")
            wandb.log({
                "EMA train loss": ema_loss,
                "wall_time"      : time_total,       
            }, step=iter)
            with torch.no_grad():
                print("Norm W_QKV", torch.linalg.norm(model.layers[0].MHA.W_QKV.param))
            # log_layerwise_adamw_updates(optimizer, model, iter, log_freq=100)



        if args.validation_every != None and (iter+1) % args.validation_every == 0 or (iter == args.run_until_step - 1 and args.validation_every != None):  
            save_validation_loss(windowed_validation, loss_fn, args, time_total, iter, best_validation_loss, ema_loss, model, optimizer)


        
        if (args.save_freq != None and (iter) % args.save_freq == 0): 
            ckpt_path = pathlib.Path(f"{args.checkpoint_path}ckpt_latest.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)   
            optimization.save_checkpoint(model=model, optimizer=optimizer, iteration = iter, out=ckpt_path, args=args, ema_loss=ema_loss, valid_loss = best_validation_loss)

    ckpt_path = pathlib.Path(f"{args.checkpoint_path}ckpt_latest.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)   
    optimization.save_checkpoint(model=model, optimizer=optimizer, iteration = iter, out=ckpt_path, args=args, ema_loss=ema_loss)

    return model, optimizer, args
                

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    main()
    
    # profiler.disable()
    # profiler.dump_stats('train_profile.prof')
    
    # # Print top results immediately
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative').print_stats(20)