#!/usr/bin/env python3
import math
import transformer, optimization, tokenizer_utils
import numpy as np
import torch
import wandb, time
import pathlib
import configs

@torch.no_grad
def eval(windowed_validation : torch.Tensor, model, args):
    valid_size = args.valid_size
    nb_batches = math.ceil(valid_size //(args.batch_size* args.context_length))
    loss = 0
    for i in range(nb_batches):
        # make sure to multiply loss by batch size*context_length
        end_window = min((i+1)*args.batch_size, valid_size)
        chunk_size = end_window - i*args.batch_size # could be smaller than batch_size
        start_window = args.batch_size*i
        data = windowed_validation[ start_window:end_window, :-1].to(args.device)
        targets = windowed_validation[ start_window:end_window, 1:].to(args.device)
        loss_batch = args.context_length*chunk_size*transformer.cross_entropy(model.forward(data), targets)
        loss += loss_batch/valid_size    

    return loss

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
    print("Training with args", args)
    train = np.memmap(args.train_data,dtype=np.uint16,mode="r")    
    validation = np.memmap(args.val_data,dtype=np.uint16,mode="r")
    args.valid_size = validation.size
    validation = torch.as_tensor(validation, dtype=torch.long)
    windowed_validation = validation.unfold(0, args.context_length + 1, args.context_length)
    if args.device == "cuda" and torch.cuda.is_available():
        args.device = "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = "cpu"
    print("device", args.device)
    model = transformer.transformer_lm(vocab_size=args.vocab_size,d_ff=args.d_ff, d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers, context_length=args.context_length, theta=args.rope_theta_parameter, device=args.device, pre_RMS=True, post_RMS=False)
    model.to(args.device)

    # # Compile model (note this changes the names of params to include _orig..., so you need to compile again before loading the checkpoint).
    # Note that compile misbehaves on mps. 
    if args.compile and args.device == "cuda":
        backend = "inductor"
        model   = torch.compile(model, backend=backend, mode="default")
    optimizer = optimization.AdamW(model.parameters(), lr = args.lr, betas = args.betas, eps = args.eps, weight_decay=args.weight_decay)
    current_iter = 0
    ema_loss = 0
    best_validation_loss = float('inf')
    if args.resume_from != None: 
        current_iter, args_old, ema_loss, best_validation_loss = optimization.load_checkpoint(src=args.resume_from, model=model, optimizer=optimizer)
        print(f"Loading from checkpoint: iteration {current_iter}, ema_loss {ema_loss}")
    if ema_loss == None:
        ema_loss = 0
    optimizer.lr = args.lr

    if args.validation_every == None: 
        args.validation_every = 100

    lambda_ema = .98
    wandb.watch(model, log="all", log_freq=100)
    time_total = 0
    print("print_every", args.print_every)
    for iter in range(current_iter, args.num_training_steps + current_iter):
        time_start = time.perf_counter()
        data, targets = tokenizer_utils.data_from_numpy(train, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
        optimizer.zero_grad()
        loss = transformer.cross_entropy(model.forward(data), targets)
        loss.backward()
        optimizer.step()
        ema_loss = (1-lambda_ema) * loss.detach() + lambda_ema*ema_loss
        time_end = time.perf_counter()  
        time_total += time_end - time_start
        if iter % args.print_every == 0:
            print(f"Iteration: {iter}, ema loss {ema_loss}")
            wandb.log({
                "EMA train loss": ema_loss,
                "wall_time"      : time_total,       
            }, step=iter)


        if (iter+1) % args.validation_every == 0:  
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

        
        if args.save_freq != None and (iter) % args.save_freq == 0: 
            ckpt_path = pathlib.Path(f"{args.checkpoint_path}ckpt_latest.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)   
            optimization.save_checkpoint(model=model, optimizer=optimizer, iteration = iter, out=ckpt_path, args=args, ema_loss=ema_loss)

        ckpt_path = pathlib.Path(f"{args.checkpoint_path}ckpt_latest.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)   
        optimization.save_checkpoint(model=model, optimizer=optimizer, iteration = iter, out=ckpt_path, args=args, ema_loss=ema_loss)

    return model, optimizer, args
                

if __name__ == "__main__":
    main()