import argparse
import os
import shlex
import subprocess
import sys
import tempfile

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def main(argv):
    parser = argparse.ArgumentParser(description="Run IPCAE rewards experiment without YAML by passing all args explicitly.")
    parser.add_argument("--k", type=int, default=2, help="Number of features to select (default: 2)")
    parser.add_argument("--input_dim", type=int, default=400, help="Input dimension (fixed to 400 by default)")
    parser.add_argument(
        "--train_path",
        type=str,
        default="/gscratch/ark/devinl6/preference/preference-decoding/rewards_high_var/rewards_persona_testing_train.pt",
        help="Absolute path to unified training rewards file",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="/gscratch/ark/devinl6/preference/preference-decoding/rewards_high_var/rewards_persona_testing_val.pt",
        help="Absolute path to unified validation rewards file",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--blr", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dim_ip", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="./outs/rewards_experiment")
    parser.add_argument("--log_dir", type=str, default="./logs/rewards_experiment")
    parser.add_argument("--every_n_epochs", type=int, default=10)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--save_last", type=int, default=1)
    parser.add_argument("--wandb", type=str, default="high_var_concrete", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--extra", type=str, default="", help="Extra raw args to forward to main_pl.py (in addition to the generated config)")
    parser.add_argument("--keep_config", action="store_true", help="Do not delete the temporary YAML config after run")

    args = parser.parse_args(argv)

    # Build command to run from repo root
    # Build a full config dict equivalent to rewards_k*_o400.yaml
    config = {
        "dataset": "custom",
        "train_path": args.train_path,
        "val_path": args.val_path,
        # Model configuration
        "model": "ConcreteLinear",
        "input_dim": int(args.input_dim),
        "dim_ip": int(args.dim_ip),
        "k": int(args.k),
        # Training configuration
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "blr": float(args.blr),
        # Loss configuration
        "loss_type": "mse",
        "jsd_factor": 0,
        "eeg_factor": 0.0,
        # Temperature and threshold annealing
        "anneal_temp": "exp",
        "temp_base": 10.0,
        "temp_min": 0.01,
        # Training settings
        "straight_through": 1,
        "rao_samples": 1,
        "use_masked_loss": 0,
        "norm_pix_loss": 0,
        # Output configuration
        "output_dir": args.output_dir,
        "log_dir": args.log_dir,
        "save_top_k": int(args.save_top_k),
        "save_last": int(args.save_last),
        "every_n_epochs": int(args.every_n_epochs),
        # Logging
        "wandb": args.wandb,
        "run_name": args.run_name,
        # Other settings
        "gumbel_learn_mode": "logits",
        "seed": int(args.seed),
        "scale_lr_by_batchsize": 0,
    }

    if yaml is None:
        print("PyYAML not available; please install pyyaml to use this runner.", file=sys.stderr)
        return 1

    config_out = getattr(args, "config_out", "")
    if config_out.strip():
        os.makedirs(os.path.dirname(os.path.abspath(config_out)), exist_ok=True)
        with open(config_out, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        tmp_config_path = os.path.abspath(config_out)
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(config, tf, default_flow_style=False)
            tmp_config_path = tf.name

    cmd = [sys.executable, "ipcae/src/main_pl.py", "--config", tmp_config_path]

    if args.extra.strip():
        # Allow power users to pass additional flags transparently
        cmd.extend(shlex.split(args.extra))

    print("Running:")
    print(" ".join(shlex.quote(c) for c in cmd))
    # Ensure we are in repo root even if executed from elsewhere
    repo_root = os.path.dirname(os.path.abspath(__file__))
    try:
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            print(f"Command failed with exit code {proc.returncode}", file=sys.stderr)
            return proc.returncode
        else:
            print(proc.stdout)
            return 0
    finally:
        if not args.keep_config and not config_out.strip():
            try:
                os.remove(tmp_config_path)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


