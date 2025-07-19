#!/usr/bin/env python3
import math
import transformer, optimization, tokenizer_utils
import numpy as np
import torch
import wandb, time
import pathlib
import configs
import cProfile
import pstats


@torch.no_grad
def eval(windowed_validation : torch.Tensor, model, args):
    model.eval()
    num_windows = windowed_validation.shape[0]
    nb_batches = math.ceil(num_windows / args.batch_size)
    loss = 0
    total_tokens = 0
    for i in range(nb_batches):
        # make sure to multiply loss by batch size*context_length
        start_window = args.batch_size*i
        end_window = min((i+1)*args.batch_size, num_windows)
        chunk_size = end_window - start_window # could be smaller than batch_size
        batched_windows = windowed_validation[start_window:end_window]
        data = batched_windows[:, :-1].to(args.device)
        targets = batched_windows[:, 1:].to(args.device)
        logits = model.forward(data)
        loss_batch = transformer.cross_entropy(logits, targets)
        num_tokens_in_batch = args.context_length*chunk_size
        loss += loss_batch*num_tokens_in_batch    
        total_tokens += num_tokens_in_batch

    model.train()
    return loss/total_tokens

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
        backend = "inductor"
        model   = torch.compile(model, backend=backend, mode="default")
    optimizer = optimization.AdamW(model.parameters(), betas = args.betas, eps = args.eps, weight_decay=args.weight_decay)
    print("Weight decay", args.weight_decay)
    current_iter = 0
    ema_loss = 0
    best_validation_loss = float('inf')
    if args.resume_from != None: 
        model, optimizer, current_iter, args_old, ema_loss, best_validation_loss = optimization.load_checkpoint(src=args.resume_from, model=model, optimizer=optimizer)
        print(f"Loading from checkpoint: iteration {current_iter}, ema_loss {ema_loss}, best_validation_loss {best_validation_loss}")
        # Test with a tiny subset first
        model.eval()
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
    
    for iter in range(current_iter, args.run_until_step):
        time_start = time.perf_counter()
        data, targets = tokenizer_utils.data_from_gpu_tensor(train, batch_size=args.batch_size, context_length=args.context_length)
        if args.lr_scheduler == "cosine":
            optimizer.set_lr(optimization.learning_rate_schedule(iter, args.cosine_decay["max_lr"], args.cosine_decay["min_lr"], args.cosine_decay["warmup_steps"], args.cosine_decay["cosine_cycle_final_iter"]))
        optimizer.zero_grad()
        loss = transformer.cross_entropy(model.forward(data), targets)
        loss.backward()
        if args.grad_clip is not None:
            optimization.gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()
        ema_loss = (1-lambda_ema) * loss.detach() + lambda_ema*ema_loss
        time_end = time.perf_counter()  
        time_total += time_end - time_start
        if iter % args.print_every == 0:
            print(f"Iteration: {iter}, ema loss {ema_loss}, lr {optimizer.param_groups[0]['lr']}")
            wandb.log({
                "EMA train loss": ema_loss,
                "wall_time"      : time_total,       
            }, step=iter)


        if args.validation_every != None and (iter+1) % args.validation_every == 0 or (iter == args.run_until_step - 1 and args.validation_every != None):  
            print("Validating")
            valid_loss = eval(windowed_validation, model, args)
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

        
        if (args.save_freq != None and (iter) % args.save_freq == 0): 
            ckpt_path = pathlib.Path(f"{args.checkpoint_path}ckpt_latest.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)   
            optimization.save_checkpoint(model=model, optimizer=optimizer, iteration = iter, out=ckpt_path, args=args, ema_loss=ema_loss, valid_loss = best_validation_loss)

    ckpt_path = pathlib.Path(f"{args.checkpoint_path}ckpt_latest.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)   
    optimization.save_checkpoint(model=model, optimizer=optimizer, iteration = iter, out=ckpt_path, args=args, ema_loss=ema_loss)

    return model, optimizer, args
                

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    profiler.dump_stats('train_profile.prof')
    
    # Print top results immediately
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)