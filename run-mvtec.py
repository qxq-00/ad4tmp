import argparse
import os
import subprocess
import sys

from ldm.device_utils import get_torch_device


parser = argparse.ArgumentParser()
parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="whether use ht encoder",
    )
parser.add_argument(
        "--adaptive_mask",
        action="store_true",
        help='whether use adaptive attention reweighting',
    )
parser.add_argument(
        "--gpu_id",
        type=int, default=0,
        help="whether use ht encoder",
    )
parser.add_argument(
        "--sample_name",
        type=str,
        default="all",
        help="Train and generate for a single MVTec class, e.g. bottle",
    )
parser.add_argument(
        "--anomaly_name",
        type=str,
        default="all",
        help="Optionally limit mask training/generation to one anomaly subtype",
    )
opt = parser.parse_args()
python_bin = sys.executable
device = get_torch_device(gpu_id=opt.gpu_id)
use_cuda = device.type == "cuda"
root_dir = opt.data_path
dirs1 = os.listdir(root_dir)

if opt.sample_name != 'all':
    dirs1 = [opt.sample_name]


def run_step(args):
    env = os.environ.copy()
    if use_cuda:
        env["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    print("Running:", " ".join(args))
    subprocess.run(args, check=True, env=env)


for dir1 in dirs1:
    if not os.path.isdir(os.path.join(root_dir, dir1)):
        continue
    dirs = os.listdir(os.path.join(root_dir, dir1, 'test'))
    for dir2 in dirs:
        if dir2 == 'good':
            continue
        if opt.anomaly_name != 'all' and dir2 != opt.anomaly_name:
            continue

        train_mask_cmd = [
            python_bin,
            "train_mask.py",
            f"--mvtec_path={opt.data_path}",
            "--base",
            "configs/latent-diffusion/txt2img-1p4B-finetune.yaml",
            "-t",
            "--actual_resume",
            "./models/ldm/text2img-large/model.ckpt",
            "-n",
            "test",
            "--init_word",
            "crack",
            f"--sample_name={dir1}",
            f"--anomaly_name={dir2}",
        ]
        if use_cuda:
            train_mask_cmd.extend(["--gpus", "0,"])
        run_step(train_mask_cmd)

        run_step([
            python_bin,
            "generate_mask.py",
            f"--data_root={root_dir}",
            f"--sample_name={dir1}",
            f"--anomaly_name={dir2}",
        ])

        generate_with_mask_cmd = [
            python_bin,
            "generate_with_mask.py",
            f"--data_root={opt.data_path}",
            f"--sample_name={dir1}",
            f"--anomaly_name={dir2}",
        ]
        if opt.adaptive_mask:
            generate_with_mask_cmd.append("--adaptive_mask")
        run_step(generate_with_mask_cmd)

# For generating texture anomaly (e.g., color in wood), set 'adaptive_mask' to be True to use
# Adaptive attention reweighting by '--adaptive_mask' in generate_with_mask.py
# For generating structual anomaly (e.g., squeeze in capsule), set 'adaptive_mask' to be False
# by removing '--adaptive_mask' in generate_with_mask.py
