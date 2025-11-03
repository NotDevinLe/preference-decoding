import torch
import pytorch_lightning as pl
from parse_args import parse_args_and_config, create_output_dirs
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import numpy as np
import random

import utils
from pl_callbacks import (
    SnapshotCallback,
    TemperatureCallback,
    CosineAnnealLRCallback,
    FreezeDistribCallback,
    ThresholdCallback,
    WarmupLRCallback,
    LocalLoggerCallback,
)


from pl_wrappers import (
    PL_CAE_Wrapper,
    PL_CAE_Wrapper_CLAS,
)

import models.cae as cae

MODEL_DICT = {
    "feature_learners": {
        **cae.__dict__,
    }
}


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)


def main(args):
    print(f"Saving checkpoints in {args.output_dir}")
    print(f"Saving logs in {args.log_dir}")

    args.epochs += 1
    # Wandb
    run = None
    logger = None

    if utils.get_rank() == 0:
        print(f"args.wandb={args.wandb},args.wandb_entity={args.wandb_entity}")
        run = wandb.init(
            project=args.wandb,
            entity=args.wandb_entity,
            name=getattr(args, "run_name", None),
            config=vars(args),
            dir=args.output_dir,
            anonymous="allow",
        )
        print(f"run.config={run.config}")
        update_args(args, dict(run.config))
        logger = WandbLogger()

    # batch_size lr scaling trick
    eff_batch_size = (
        args.batch_size * args.accum_iter * args.num_devices * args.num_nodes
    )
    if args.scale_lr_by_batchsize:
        args.lr = args.blr * eff_batch_size / 256 if args.lr is None else args.lr
    else:
        args.lr = args.blr

    callbacks = []

    if bool(args.local_logging):
        callbacks += [LocalLoggerCallback(log_dir=args.log_dir)]
    # Checkpoint callback
    callbacks += [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=args.save_last,
            every_n_epochs=args.every_n_epochs,
        )
    ]
    if bool(args.save_snapshots):
        callbacks += [SnapshotCallback(args.output_dir)]

    if args.anneal_lr == "cosine":
        # Cosine annealing LR callback with warmup
        callbacks += [
            CosineAnnealLRCallback(
                lr=args.lr, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs
            )
        ]
    elif args.anneal_lr == "warmup":
        # Warmup, thereafter fixed lr
        callbacks += [
            WarmupLRCallback(
                lr=args.lr, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs
            )
        ]

    # TODO: add switch_st callback

    datasets = utils.get_dataset(
        args.dataset, 
        args.data_root_dir, 
        train_path=getattr(args, 'train_path', None),
        val_path=getattr(args, 'val_path', None)
    )

    # Infer input_dim from dataset if not explicitly provided
    inferred_input_dim = getattr(args, "input_dim", None)
    try:
        if inferred_input_dim is None and len(datasets) > 0 and len(datasets[0]) > 0:
            sample_x = datasets[0][0][0]
            if isinstance(sample_x, torch.Tensor):
                inferred_input_dim = int(np.prod(tuple(sample_x.shape)))
    except Exception:
        # Fallback to None and let model defaults handle if inference fails
        inferred_input_dim = getattr(args, "input_dim", None)

    args.test_batch_size = args.batch_size
    if args.full_batch:
        args.batch_size = len(datasets[0])

    # Define model
    if args.model in MODEL_DICT["feature_learners"]:
        # add temp annealing callback if CAE
        if args.anneal_temp:
            callbacks += [
                TemperatureCallback(
                    num_epochs=args.epochs,
                    temp_base=args.temp_base,
                    temp_min=args.temp_min,
                    stop_anneal=args.stop_anneal,
                    # warmup_epochs=args.freeze_epochs,
                    mode=args.anneal_temp,
                )
            ]
        if args.anneal_thresh:
            callbacks += [
                ThresholdCallback(
                    num_epochs=args.epochs,
                    temp_base=args.thresh_base,
                    temp_min=args.thresh_min,
                    stop_anneal=args.stop_anneal,
                    # warmup_epochs=args.freeze_epochs,
                    mode=args.anneal_thresh,
                )
            ]
        model = MODEL_DICT["feature_learners"][args.model](
            input_dim=inferred_input_dim,
            dim_ip=args.dim_ip,
            mask_ratio=args.mask_ratio,
            k=args.k,
            pi_dropout=args.pi_dropout,
            IP_dropout=args.IP_dropout,
            pi_layer_norm=args.pi_layer_norm,
            gumbel_learn_mode=args.gumbel_learn_mode,
            IP_initialization=args.IP_initialization,
            IP_weights=args.IP_weights,
            IP_bias=bool(int(args.IP_bias)),
            no_gumbel_noise=args.no_gumbel_noise,
            marginal_initialization=args.marginal_initialization,
        )

        if args.model.startswith("clas"):
            pl_model = PL_CAE_Wrapper_CLAS(model=model, args=args, datasets=datasets)
        else:
            pl_model = PL_CAE_Wrapper(model=model, args=args, datasets=datasets)

    else:
        raise NotImplementedError("Model not found")

    if run is not None and bool(args.watch_grads):
        wandb.watch(model, log="all", log_freq=20)

    # callback for freezing the gumbel distribution
    if args.freeze_distrib or args.freeze_epochs:
        callbacks += [
            FreezeDistribCallback(model, args.freeze_distrib, args.freeze_epochs)
        ]

    # Define trainer
    distributed = args.num_devices > 1 or args.num_nodes > 1
    (
        print(
            f"Starting distributed training using {args.num_devices} devices on {args.num_nodes} node(s)"
        )
        if distributed
        else print("Starting single-device training")
    )

    if run is not None and utils.get_rank() == 0:
        run.log({"num_parameters": utils.get_num_parameters(model)})

    trainer = pl.Trainer(
        # Distributed kwargs
        accelerator=args.accelerator,
        devices=(
            [i for i in range(args.num_devices)]
            if args.accelerator == "gpu"
            else args.num_devices
        ),
        num_nodes=args.num_nodes,
        strategy=args.strategy if distributed else "auto",
        precision=args.precision,
        # Training args
        max_epochs=args.epochs,
        gradient_clip_val=args.clip_grad,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0 if bool(args.save_snapshots) else 2,
        benchmark=True,
    )
    trainer.validate(pl_model)  # Validate once before training
    trainer.fit_loop.epoch_progress.current.processed = 1
    trainer.fit(pl_model)

    if args.save_top_k > 0:
        try:
            trainer.test(ckpt_path="best")
        except ValueError as e:
            print(f"Warning: Could not test with 'best' checkpoint: {e}")
            print("Testing with the last checkpoint instead...")
            trainer.test(ckpt_path="last")

    # Extract selected features from the learned model parameters
    print("\n" + "="*60)
    print("EXTRACTING SELECTED FEATURES")
    print("="*60)
    
    try:
        model = pl_model.model
        gd = model.gumbel_distrib
        gd.eval()           # disables dropout/layernorm stats updates
        model.eval()

        with torch.no_grad():
            L = gd.get_logits()          # (K, D)
            idx = L.argmax(dim=1)        # (K,)
        selected = idx.cpu().tolist()

        print(f"Selected features (K={len(selected)}): {selected}")
    except Exception as e:
        print(f"Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

    wandb.finish()

if __name__ == "__main__":
    # parse args
    args = parse_args_and_config()
    create_output_dirs(
        args, is_main_process=utils.get_rank() == 0
    )  # TODO: revisit main process
    # A100 specific setting
    if args.matmul_precision:
        torch.set_float32_matmul_precision(args.matmul_precision)
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # run
    main(args)
