#!/usr/bin/env python3

from huggingface_hub import snapshot_download
from pathlib import Path

REPO_ID   = "damekdavis/TinyStories"   
TARGET    = Path("TinyStories")

def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id     = REPO_ID,
        repo_type   = "dataset",    
        local_dir   = TARGET,
        local_dir_use_symlinks = False   
    )
    print(f"Files are in {TARGET.resolve()}")

if __name__ == "__main__":
    main()
    "huggingface_hub>=0.23.0",     
