from huggingface_hub import snapshot_download

def download_stablesr_testsets():
    # Download the dataset from huggingface
    local_dir = snapshot_download(
        repo_id="Iceclear/StableSR-TestSets",
        repo_type="dataset",
        local_dir="/ceph/data-bk/zl/datasets/StableSR-TestSets",  # Local directory to save the dataset
        local_dir_use_symlinks=False  # Download actual files instead of symlinks
    )
    print(f"Dataset downloaded to: {local_dir}")
    return local_dir

if __name__ == "__main__":
    download_stablesr_testsets()
