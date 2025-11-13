import argparse
import shlex
import subprocess
import sys
from typing import List


def build_command_for_user(user_id: int) -> List[str]:
    """Build the llamafactory-cli command for a given user id.

    Mirrors utils/reward_train/eval_rm/user0.yaml, changing only dataset/output paths.
    """
    dataset_name_train = f"user{user_id}_train"
    dataset_name_val = f"user{user_id}_val"
    output_dir = f"saves/mod/user{user_id}_1b/full"

    # Construct the command as a list for subprocess
    cmd: List[str] = [
        "llamafactory-cli",
        "train",
        # method
        "--stage", "rm",
        "--do_train", "true",
        "--finetuning_type", "full",
        # model
        "--model_name_or_path", "meta-llama/Llama-3.2-1B-Instruct",
        "--trust_remote_code", "true",
        # dataset
        "--dataset", dataset_name_train,
        "--dataset_dir", "/mmfs1/gscratch/ark/devinl6/preference/preference-decoding/data/PERSONA_testing",
        "--template", "llama3",
        "--cutoff_len", "2048",
        "--max_samples", "80",
        "--overwrite_cache", "true",
        "--preprocessing_num_workers", "8",
        "--dataloader_num_workers", "4",
        "--save_safetensors", "false",
        # output
        "--output_dir", output_dir,
        "--logging_steps", "10",
        "--save_steps", "500",
        "--plot_loss", "true",
        "--overwrite_output_dir", "true",
        "--save_only_model", "false",
        "--report_to", "wandb",
        # train
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "1.0e-5",
        "--num_train_epochs", "3.0",
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.1",
        "--bf16", "true",
        "--ddp_timeout", "180000000",
        # eval
        "--eval_dataset", dataset_name_val,
        "--per_device_eval_batch_size", "1",
        "--eval_strategy", "steps",
        "--eval_steps", "100",
    ]
    return cmd


 


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Run llamafactory-cli training for a single user.")
    parser.add_argument(
        "--user",
        type=int,
        help="User id to run (e.g., 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing.",
    )

    args = parser.parse_args(argv)
    user_id = args.user

    cmd = build_command_for_user(user_id)
    printable = " ".join(shlex.quote(p) for p in cmd)
    print(f"\n=== user{user_id}: {printable}")
    if args.dry_run:
        return 0
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"user{user_id}: command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


