import argparse
import logging
import os
import subprocess

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--script_path", type=str, default=None, help="script path")
# script_path = (
#     "/data/users/zhangjunlei/tyx/reward-by-prm800k/src/download-from-hf-hub.py"
# )
parser.add_argument(
    "--process_wise_args_path",
    type=str,
    default="./process-wise-args.txt",
    help="args path",
)
parser.add_argument("--log_path", type=str, default=None, help="log path")
parser.add_argument("--max_trials", type=int, default=10, help="max trials")
args = parser.parse_args()

utils.init_logging()

logger = logging.getLogger(__name__)


argname = os.path.basename(args.process_wise_args_path).split(".")[0]
with open(args.process_wise_args_path, "r") as args_file:
    process_wise_arg_vals = args_file.readlines()
process_wise_arg_vals = [arg_val.strip() for arg_val in process_wise_arg_vals]

if args.log_path is not None and os.path.exists(args.log_path):
    os.remove(args.log_path)


for arg_val in process_wise_arg_vals:
    trial_cnt = 0
    while True:
        # 运行另一个Python脚本
        result = subprocess.run(
            ["nohup", utils.python_path, args.script_path, f"--{argname}", arg_val],
            # stdout=log_file,
            # stderr=log_file,
            env=os.environ,
        )

        # 如果脚本成功运行，退出循环
        if result.returncode == 0:
            break
        else:
            trial_cnt += 1
            logger.warning(f"Failed to run the script for {trial_cnt} times.")

            if trial_cnt >= args.max_trials:
                raise Exception(f"Failed to run for {args.max_trials} times.")
