# Start Training - Multi-Terminal Setup

Follow these steps to start training with each server in its own terminal:

## Terminal 1: VLLM Server
```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000 --gpu-memory-utilization 0.6
```
**Status**: Wait for "Application startup complete" message

## Terminal 2: Learner Server  
```bash
python -m gumbel.core.learner_server --config gumbel/configs/experiment.yaml
```
**Status**: Wait for "Uvicorn running on http://0.0.0.0:8002"

## Terminal 3: Collector Server
```bash  
python -m gumbel.core.collector_server --config gumbel/configs/experiment.yaml
```
**Status**: Wait for "Uvicorn running on http://0.0.0.0:8001"

## Terminal 4: Start Training (Coordinator)
```bash
python -m gumbel.core.coordinator --config gumbel/configs/experiment.yaml
```
**This terminal will show the training progress and run the actual training loop!**

---

## 🎯 What Each Server Does:

- **VLLM**: Serves the language model for scoring text
- **Learner**: Handles model training and parameter updates  
- **Collector**: Samples data and computes drift rewards
- **Coordinator**: **RUNS THE TRAINING** - coordinates data flow between servers

## 📊 Training Output:
The coordinator will show training progress like:
```
🎉 BATCH: 32 samples | reward∈[-0.123,0.456]
Step 10 | Loss: 0.1234
Step 20 | Loss: 0.1156
...
Training completed at step 1000
```

## 🛑 To Stop:
- Ctrl+C in the Coordinator terminal (Terminal 4) first
- Then Ctrl+C in the other terminals

## ⚙️ Current Config:
- **1000 attributes** (d=1000)  
- **1000 training steps**
- **4 users × 8 samples per batch**
- **Checkpoints saved every 500 steps**