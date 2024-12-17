from huggingface_hub import login, create_repo, HfApi

# Login first (this will prompt for your token)
login()

# Initialize the API
api = HfApi()

# 1. Create the repository
repo_id = "JackyZhuo/krea_2M"  # Replace with your desired repo name
# create_repo(
#     repo_id=repo_id,
#     repo_type="dataset",  # Use "dataset" for data files, "model" for model files
#     private=False  # Set to True if you want a private repository
# )

# 2. Upload the file
api.upload_file(
    path_or_fileobj="/goosedata/3_5_million_captioned_2024_10_25.csv",
    path_in_repo="3_5_million_captioned_2024_10_25.csv",
    repo_id=repo_id,
    repo_type="dataset"
)