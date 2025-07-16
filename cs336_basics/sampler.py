import torch
import configs, tokenizer_utils, transformer, optimization
import argparse

# Get command line arguments from user 
## Checkpoint location 
## Prompt 
## Number of tokens to sample
## Temperature
## Top-p

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=str, required=True)
p.add_argument("--prompt", type=str, required=True)
p.add_argument("--max_tokens", type=int, default=100)
p.add_argument("--temperature", type=float, default=1)
p.add_argument("--p", type=float, default=None)
p.add_argument("--vocab_path", type=str, required=True)
p.add_argument("--merges_path", type=str, required=True)
p.add_argument("--device", type=str, default="cuda")
p.add_argument("--num_tokens_per_sample", type=int, default=1)
args = p.parse_args()

## Load model
model, _, _, _, _, _ = optimization.load_checkpoint(args.checkpoint, model=None, optimizer=None)

## Load tokenizer
tokenizer = tokenizer_utils.Tokenizer.from_files(args.vocab_path, args.merges_path)

## Sample from the model 

for i in range(args.num_tokens_per_sample):
    sampled_text = transformer.decode(prompt=args.prompt, model=model, tokenizer=tokenizer, max_tokens=args.max_tokens, temperature=args.temperature, p=args.p, device=args.device)
    print(sampled_text)
    print("-"*100)






