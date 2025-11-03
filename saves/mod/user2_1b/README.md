---
library_name: peft
license: other
base_model: meta-llama/Llama-3.2-1B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: user2_1b
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# user2_1b

This model is a fine-tuned version of [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on the user2_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6590
- Accuracy: 0.9

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.0033        | 1.25  | 100  | 1.2063          | 0.85     |
| 0.0           | 2.5   | 200  | 0.7336          | 0.9      |
| 0.0           | 3.75  | 300  | 0.6621          | 0.9      |
| 0.0           | 5.0   | 400  | 0.6590          | 0.9      |


### Framework versions

- PEFT 0.15.2
- Transformers 4.52.4
- Pytorch 2.7.0+cu126
- Datasets 3.6.0
- Tokenizers 0.21.1