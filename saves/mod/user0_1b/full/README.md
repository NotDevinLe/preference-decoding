---
library_name: transformers
license: other
base_model: meta-llama/Llama-3.2-1B-Instruct
tags:
- llama-factory
- full
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: full
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# full

This model is a fine-tuned version of [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on the user0_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0001
- Accuracy: 1.0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.0           | 1.25  | 100  | 0.0014          | 1.0      |
| 0.0           | 2.5   | 200  | 0.0001          | 1.0      |


### Framework versions

- Transformers 4.52.4
- Pytorch 2.7.0+cu126
- Datasets 3.6.0
- Tokenizers 0.21.1
