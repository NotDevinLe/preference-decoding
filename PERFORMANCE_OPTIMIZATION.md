# Performance Optimization Guide

## The Problem: 2 prompts/second with 1000 attributes

### Root Cause Analysis
With **1000 attributes** and **sequential processing**:
- Old approach: 1 base call + 1000 attribute calls = **1001 sequential API calls**
- Each call ~100-200ms = **100+ seconds per batch**
- Result: **~2 prompts/second** (very slow!)

### Solution: Full Concurrency
New approach fires all **(d+1) × B requests simultaneously**:
- 8 users × 4 samples = 32 samples
- 32 samples × 1001 prompts = **32,032 concurrent requests**
- Expected speedup: **100-500x faster!**

## Performance Optimizations Applied

### ✅ v3 Updates (2025-09-14)
- **Full Concurrency**: `compute_drift_rewards()` now fires all requests in one wave
- **Batch Processing**: All (d+1)*B requests processed simultaneously
- **Request Mapping**: Efficient reconstruction of base and attribute scores
- **Performance Monitoring**: Added timing logs to track request throughput

## Expected Performance Improvements

### Before (Sequential)
```
1000 attributes × 32 samples = 32,000 requests
Processing: Sequential (1 → 2 → 3 → ... → 32,000)
Time per batch: ~100+ seconds
Throughput: ~2 prompts/second
```

### After (Concurrent)
```
1000 attributes × 32 samples = 32,000 requests  
Processing: Concurrent (all fired simultaneously)
Time per batch: ~2-10 seconds (depending on VLLM server capacity)
Throughput: ~50-300 prompts/second
```

## Tuning Parameters

### 1. Concurrency Limits
```bash
# Increase collector concurrency (default: 256)
export COLLECTOR_CONCURRENCY=1024

# Start collector
gumbel-collector --your-args-here
```

### 2. VLLM Server Optimization
```bash
# Increase VLLM max parallel requests
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --port 8000 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.9
```

### 3. Batch Size Tuning
Start small and increase gradually:
```bash
# Small batches (good for testing)
python -m gumbel.tests.test_collector --users-per-batch 2 --samples-per-user 2

# Medium batches (balanced)
python -m gumbel.tests.test_collector --users-per-batch 8 --samples-per-user 4

# Large batches (high throughput)
python -m gumbel.tests.test_collector --users-per-batch 16 --samples-per-user 8
```

### 4. Network Optimization
```bash
# Use local VLLM server for best performance
--vllm-server-url http://localhost:8000

# For remote servers, consider:
# - Higher timeout values
# - Lower concurrency limits
# - Smaller batch sizes
```

## Performance Monitoring

### Real-time Monitoring
The optimized version now prints timing information:
```
Firing 32032 concurrent requests (32 samples × 1001 prompts)...
Completed 32032 requests in 3.45s (9286.7 req/sec)
```

### Test Performance
```bash
# Run performance test
python -m gumbel.tests.test_collector \
  --users-per-batch 8 \
  --samples-per-user 4 \
  --max-batches 10

# Monitor output for:
# - Samples/sec throughput  
# - Batch completion time
# - Request success rate
```

## Troubleshooting Performance Issues

### Issue 1: Still Slow After Optimization
**Symptoms**: Still getting 2-10 prompts/second
**Causes**: 
- VLLM server bottleneck
- Network latency
- Memory constraints

**Solutions**:
```bash
# Check VLLM server logs for errors
# Increase VLLM max parallel processing
# Use faster GPU/more GPU memory
# Reduce batch size temporarily
```

### Issue 2: Timeout Errors
**Symptoms**: aiohttp timeout errors, 504 Gateway Timeout
**Causes**: Too many concurrent requests overwhelming VLLM

**Solutions**:
```bash
# Reduce collector concurrency
export COLLECTOR_CONCURRENCY=128

# Increase timeouts
# Check VLLM server capacity
```

### Issue 3: Memory Errors
**Symptoms**: CUDA OOM, system memory issues
**Causes**: Too large batches, high concurrency

**Solutions**:
```bash
# Reduce batch sizes
--users-per-batch 4 --samples-per-user 2

# Reduce GPU memory utilization
vllm serve ... --gpu-memory-utilization 0.7
```

## Expected Results

### Target Performance (1000 attributes)
- **Small batches** (8 samples): 50-100 prompts/second
- **Medium batches** (32 samples): 100-300 prompts/second  
- **Large batches** (64+ samples): 200-500 prompts/second

### Scaling with Attributes
- **100 attributes**: 10x faster than 1000 attributes
- **50 attributes**: 20x faster than 1000 attributes
- **10 attributes**: 100x faster than 1000 attributes

## Performance Testing Commands

```bash
# Quick performance test
python -m gumbel.tests.test_collector --max-batches 5

# Stress test with large batches
python -m gumbel.tests.test_collector \
  --users-per-batch 16 \
  --samples-per-user 8 \
  --max-batches 20

# Sustained throughput test
python -m gumbel.tests.test_collector \
  --users-per-batch 8 \
  --samples-per-user 4 \
  --max-batches 100 \
  --batch-interval 1.0
```

The optimization should give you **50-100x speedup** from 2 prompts/second to 100-300 prompts/second! 🚀