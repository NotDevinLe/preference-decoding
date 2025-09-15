#!/bin/bash
# Multi-terminal server startup helper
# This script prints commands to run in separate terminals

echo "================================================================"
echo "🚀 GUMBEL DISTRIBUTED TRAINING - MULTI-TERMINAL SETUP"
echo "================================================================"
echo ""
echo "Open 4 terminals and run these commands in order:"
echo ""

echo "📟 TERMINAL 1 - VLLM Server:"
echo "vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000 --gpu-memory-utilization 0.6"
echo ""

echo "🧠 TERMINAL 2 - Learner Server:"  
echo "python -m gumbel.core.learner_server --config gumbel/configs/experiment.yaml"
echo ""

echo "📊 TERMINAL 3 - Collector Server:"
echo "python -m gumbel.core.collector_server --config gumbel/configs/experiment.yaml"
echo ""

echo "🎯 TERMINAL 4 - Start Training (Coordinator):"
echo "python -m gumbel.core.coordinator --config gumbel/configs/experiment.yaml"
echo ""

echo "================================================================"
echo "📋 STARTUP ORDER:"
echo "1. Start Terminal 1 (VLLM) first - wait for 'startup complete'"
echo "2. Start Terminal 2 (Learner) - wait for 'Uvicorn running'"  
echo "3. Start Terminal 3 (Collector) - wait for 'Uvicorn running'"
echo "4. Start Terminal 4 (Coordinator) - THIS RUNS THE TRAINING!"
echo ""
echo "🎉 Training will start automatically when Coordinator connects!"
echo "================================================================"