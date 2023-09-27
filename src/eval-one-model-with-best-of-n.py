import argparse

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--gpu_ids", type=str, required=True)
    args = parser.parse_args()

    utils.set_gpu_ids(args.gpu_ids)

    utils.eval_model_with_best_of_n(
        model_name_or_path=args.model_name_or_path,
        metrics=[metric for metric in utils.all_metrics if metric != "majority_voting"],
        seed=42,
    )
