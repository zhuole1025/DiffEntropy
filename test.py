from huggingface_hub import create_repo

repo_name = "JackyZhuo/enhance-flux"
create_repo(repo_name, private=False)  # Set private=True if you want a private repo

from huggingface_hub import upload_file

# Path to your local file
local_path = "/ceph/data-bk/zl/DiffEntropy/flux/results/1024_1.0_256,128,64_0.4,0.4,0.2_controlnet_2_4_backbone_19_38_snr_uniform_cnet_snr_none_cfg_1.0_wo_shift_lr_1e-5_cap_redux_tiled_multi_degradation_wo_noise/checkpoints/0064000/consolidated.00-of-01.pth"

# Path in the repo where you want to store the file
repo_path = "v1_64k/consolidated.00-of-01.pth"

upload_file(
    path_or_fileobj=local_path,
    path_in_repo=repo_path,
    repo_id=repo_name,
    repo_type="model"
)