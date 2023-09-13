import argparse
import logging
import os
import subprocess
import time

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--script_path", type=str, default=None, help="script path")
# script_path = (
#     "/data/users/zhangjunlei/tyx/reward-by-prm800k/src/download-from-hf-hub.py"
# )
parser.add_argument(
    "--process_wise_args_path",
    type=str,
    default=None,
    help="args path",
)
parser.add_argument("--log_path", type=str, default=None, help="log path")
parser.add_argument("--max_trials", type=int, default=None, help="max trials")
parser.add_argument("--trial_interval", type=int, default=60, help="trial interval")
args = parser.parse_args()

utils.init_logging()

logger = logging.getLogger(__name__)

if args.process_wise_args_path is not None:
    argname = os.path.basename(args.process_wise_args_path).split(".")[0]
    with open(args.process_wise_args_path, "r") as args_file:
        process_wise_arg_vals = args_file.readlines()
    process_wise_arg_vals = [arg_val.strip() for arg_val in process_wise_arg_vals]
else:
    argname = None
    process_wise_arg_vals = [""]

if args.log_path is not None and os.path.exists(args.log_path):
    os.remove(args.log_path)


for arg_val in process_wise_arg_vals:
    trial_cnt = 0
    run_args = ["nohup", utils.python_path, args.script_path]
    if argname is not None:
        logger.info(f"Running script with {argname}={arg_val}")
        run_args += [f"--{argname}", arg_val]
    while True:
        # 运行另一个Python脚本
        result = subprocess.run(
            args=run_args,
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
            time.sleep(args.trial_interval)
            if args.max_trials is not None and trial_cnt >= args.max_trials:
                raise Exception(f"Failed to run for {args.max_trials} times.")
