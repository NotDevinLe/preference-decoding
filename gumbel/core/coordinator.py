#!/usr/bin/env python3
"""
Coordinator: Start and manage both collector and learner servers.
- Pure-async HTTP with aiohttp (keep-alive + connection pooling)
- Backpressure via asyncio.Queue and a semaphore for HTTP concurrency
- Replay buffer support with fresh+replay mixing
- Optional live monitoring (matplotlib) and W&B logging
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import random
import argparse
import logging
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from collections import deque


try:
    import uvloop  # type: ignore
    UVLOOP = True
except Exception:
    UVLOOP = False

import aiohttp
from aiohttp import ClientSession, TCPConnector, ClientTimeout

# Optional W&B
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


# ============ Coordinator ============

class ServerCoordinator:
    """Coordinates collector and learner servers using async HTTP."""

    def __init__(
        self,
        collector_url: str,
        learner_url: str,
        queue_size: int = 100,
        replay_buffer_size: int = 10_000,
        replay_ratio: float = 0.3,
        enable_monitoring: bool = True,
        enable_wandb: bool = False,
        plot_update_interval: float = 10.0,
        timeouts: Dict[str, float] | None = None,
        http_concurrency: int = 128,
        http_total_timeout: float | None = None,
    ):
        # Endpoints
        self.collector_url = collector_url.rstrip("/")
        self.learner_url = learner_url.rstrip("/")

        # Timeouts
        self.timeouts = timeouts or {
            "server_health_check": 120.0,
            "server_startup_wait": 300.0,
            "get_params": 120.0,
            "generate_batch": 180.0,
            "train_step": 120.0,
            "server_status": 120.0,
            "system_check": 150.0,
        }
        self.http_total_timeout = http_total_timeout or self.timeouts.get("system_check", 150.0)

        # Async queue + control
        self.batch_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self.training_active: bool = False
        self.producer_task: Optional[asyncio.Task] = None
        self.consumer_task: Optional[asyncio.Task] = None

        # Replay buffer
        self.replay_buffer: deque[Dict[str, Any]] = deque(maxlen=replay_buffer_size)
        self.replay_ratio: float = replay_ratio

        # Monitoring
        self.enable_monitoring: bool = enable_monitoring
        self.enable_wandb: bool = enable_wandb and WANDB_AVAILABLE
        self.plot_update_interval: float = plot_update_interval
        self.start_time: Optional[float] = None
        self.metrics: Dict[str, List[float]] = {
            "timestamps": [],
            "steps": [],
            "losses": [],
            "reward_signals": [],
            "active_attributes": [],
            "temperatures": [],
            "queue_sizes": [],
            "replay_buffer_sizes": [],
        }
        self.monitoring_thread: Optional[threading.Thread] = None
        self.wandb_run = None

        # Training parameters (tune as desired)
        self.users_per_batch: int = 4
        self.samples_per_user: int = 8

        # HTTP client and limits
        self.http: Optional[ClientSession] = None
        self.http_concurrency = http_concurrency
        self.http_sem = asyncio.Semaphore(self.http_concurrency)

        logging.info("ServerCoordinator initialized")
        logging.info(f"Collector: {self.collector_url}")
        logging.info(f"Learner:   {self.learner_url}")
        logging.info(f"Queue size: {queue_size}")
        logging.info(f"🔄 REPLAY BUFFER: Enabled with {replay_buffer_size} max samples (mixing ratio: {replay_ratio:.1%})")

    # ---------- HTTP helpers ----------

    async def _ensure_http(self) -> None:
        if self.http is None or self.http.closed:
            timeout = ClientTimeout(total=self.http_total_timeout)
            connector = TCPConnector(limit=self.http_concurrency, limit_per_host=self.http_concurrency)
            self.http = ClientSession(timeout=timeout, connector=connector, raise_for_status=False)

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_payload: Any | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        retry_backoff_base: float = 0.5,
    ) -> Tuple[int, Any | None]:
        """Generic JSON request with simple exponential backoff."""
        await self._ensure_http()
        assert self.http is not None

        for attempt in range(1, max_retries + 1):
            try:
                async with self.http_sem:
                    if method.upper() == "GET":
                        async with self.http.get(url, timeout=timeout) as resp:
                            status = resp.status
                            data = await self._safe_json(resp)
                    elif method.upper() == "POST":
                        async with self.http.post(url, json=json_payload, timeout=timeout) as resp:
                            status = resp.status
                            data = await self._safe_json(resp)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                if 200 <= status < 300:
                    return status, data

                # Non-2xx: sometimes retries help (e.g., 429/5xx)
                logging.debug(f"{method} {url} -> HTTP {status}, attempt {attempt}/{max_retries}")
            except Exception as e:
                logging.debug(f"{method} {url} failed on attempt {attempt}/{max_retries}: {e}")

            if attempt < max_retries:
                sleep_s = retry_backoff_base * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
                await asyncio.sleep(sleep_s)

        # Final failure
        return 0, None

    @staticmethod
    async def _safe_json(resp: aiohttp.ClientResponse) -> Any | None:
        try:
            return await resp.json()
        except Exception:
            try:
                txt = await resp.text()
                logging.debug(f"Non-JSON response: {txt[:200]}...")
            except Exception:
                pass
            return None

    # ---------- Server readiness ----------

    async def wait_for_server(self, url: str, name: str, max_wait: float | None = None) -> bool:
        max_wait = max_wait or self.timeouts["server_startup_wait"]
        logging.info(f"Waiting for {name} server at {url} (timeout: {max_wait}s)...")
        start = time.time()

        while time.time() - start < max_wait:
            status, data = self._ensure_future(await self._request_json(
                "GET",
                f"{url}/health",
                timeout=self.timeouts["server_health_check"],
                max_retries=1,
            ))
            if status == 200:
                logging.info(f"{name} server ready: {data}")
                return True
            await asyncio.sleep(1.0)

        logging.error(f"{name} server failed to start within {max_wait}s")
        return False

    @staticmethod
    def _ensure_future(result: Tuple[int, Any | None]) -> Tuple[int, Any | None]:
        # Small helper for readability; returns as-is.
        return result

    async def connect_to_servers(self) -> None:
        await self._ensure_http()
        logging.info("=== Connecting to Existing Servers ===")
        if not await self.wait_for_server(self.collector_url, "Collector"):
            raise RuntimeError(f"Collector not available at {self.collector_url}")
        if not await self.wait_for_server(self.learner_url, "Learner"):
            raise RuntimeError(f"Learner not available at {self.learner_url}")
        logging.info("=== Connected to both servers ===")

    # ---------- Replay buffer ----------

    def add_to_replay_buffer(self, R: List[List[float]], user_data: Dict[str, Any]) -> None:
        batch_size = len(R)
        prompts = user_data.get("prompts", [])
        outputs = user_data.get("outputs", [])
        user_ids = user_data.get("user_ids", [])

        for i in range(batch_size):
            sample = {
                "reward_vector": R[i],
                "prompt": prompts[i] if i < len(prompts) else "",
                "output": outputs[i] if i < len(outputs) else "",
                "user_id": user_ids[i] if i < len(user_ids) else f"u{i}",
            }
            self.replay_buffer.append(sample)

    def sample_replay_data(self, target_batch_size: int) -> Optional[Dict[str, Any]]:
        if not self.replay_buffer:
            logging.debug("🚫 REPLAY BUFFER: Empty, no replay data available")
            return None
        replay_size = min(int(target_batch_size * self.replay_ratio), len(self.replay_buffer))
        if replay_size <= 0:
            logging.debug(f"🚫 REPLAY BUFFER: Calculated replay size {replay_size} <= 0, skipping replay")
            return None

        logging.debug(f"🎲 REPLAY BUFFER: Sampling {replay_size} from {len(self.replay_buffer)} available samples")
        replay_samples = random.sample(list(self.replay_buffer), replay_size)
        
        R = [s["reward_vector"] for s in replay_samples]
        user_data = {
            "prompts": [s["prompt"] for s in replay_samples],
            "outputs": [s["output"] for s in replay_samples],
            "user_ids": [s["user_id"] for s in replay_samples],
        }
        return {"R": R, "user_data": user_data, "success": True}

    def mix_fresh_and_replay(self, fresh: Dict[str, Any], replay: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if replay is None:
            return fresh

        combined_R = fresh["R"] + replay["R"]
        combined_user_data = {
            "prompts": fresh["user_data"]["prompts"] + replay["user_data"]["prompts"],
            "outputs": fresh["user_data"]["outputs"] + replay["user_data"]["outputs"],
            "user_ids": fresh["user_data"]["user_ids"] + replay["user_data"]["user_ids"],
        }
        return {"R": combined_R, "user_data": combined_user_data, "success": True}

    def get_replay_stats(self) -> Dict[str, Any]:
        if not self.replay_buffer:
            return {
                "size": 0,
                "max_size": self.replay_buffer.maxlen,
                "utilization": 0.0,
                "replay_ratio": self.replay_ratio,
                "unique_users": 0,
            }
        unique_users = len({s["user_id"] for s in self.replay_buffer})
        return {
            "size": len(self.replay_buffer),
            "max_size": self.replay_buffer.maxlen,
            "utilization": len(self.replay_buffer) / self.replay_buffer.maxlen,
            "replay_ratio": self.replay_ratio,
            "unique_users": unique_users,
        }

    # ---------- Server RPCs ----------

    async def get_learner_params(self) -> Dict[str, Any] | None:
        status, data = await self._request_json(
            "GET", f"{self.learner_url}/get_params", timeout=self.timeouts["get_params"]
        )
        if status == 200:
            return data
        logging.error(f"get_learner_params failed: HTTP {status}")
        return None

    async def call_collector_generate_batch(self, behavior_logits: List[float], tau: float) -> Dict[str, Any] | None:
        payload = {
            "users_per_batch": self.users_per_batch,
            "samples_per_user": self.samples_per_user,
            "behavior_logits": behavior_logits,  # keep if collector expects; otherwise remove
            "tau": tau,
        }
        status, data = await self._request_json(
            "POST", f"{self.collector_url}/generate_batch",
            json_payload=payload, timeout=self.timeouts["generate_batch"]
        )
        if status == 200:
            return data
        if status == 0:
            logging.error(f"🔥 COLLECTOR CONNECTION FAILED: HTTP {status} - collector server may be down or unreachable")
        else:
            logging.error(f"🔥 COLLECTOR ERROR: HTTP {status} - {data}")
        return None

    async def call_learner_train_step(self, batch_data: Dict[str, Any]) -> Dict[str, Any] | None:
        # Format request for learner's expected structure
        learner_request = {
            "m_hard": [],  # Empty since we're not using hard masks anymore
            "R": batch_data.get("R", []),
            "user_data": batch_data.get("user_data", {}),
            "success": batch_data.get("success", True),
            "error": batch_data.get("error", None)
        }
        
        status, data = await self._request_json(
            "POST", f"{self.learner_url}/train_step",
            json_payload=learner_request, timeout=self.timeouts["train_step"]
        )
        if status == 200:
            return data
        logging.error(f"learner.train_step failed: HTTP {status}")
        return None

    async def get_status(self, which: str) -> Dict[str, Any] | None:
        base = self.collector_url if which == "collector" else self.learner_url
        status, data = await self._request_json(
            "GET", f"{base}/status", timeout=self.timeouts["server_status"]
        )
        return data if status == 200 else None

    # ---------- Producer/Consumer ----------

    async def producer_loop(self) -> None:
        logging.info("Producer loop started")
        while self.training_active:
            try:
                # Pull behavior policy from learner
                params = await self.get_learner_params()
                if not params or not params.get("success"):
                    logging.warning("get_learner_params() failed; retrying")
                    await asyncio.sleep(1.0)
                    continue

                behavior_logits = params.get("mask_logits", [])
                tau = params.get("tau", 1.0)

                # Ask collector for a fresh batch
                fresh = await self.call_collector_generate_batch(behavior_logits, tau)
                if not fresh or not fresh.get("success"):
                    logging.warning("collector.generate_batch failed; retrying")
                    await asyncio.sleep(1.0)
                    continue

                # Log batch stats
                R = fresh.get("R", [])
                batch_size = len(R)
                if R:
                    try:
                        rmin = min(min(row) for row in R)
                        rmax = max(max(row) for row in R)
                        logging.info(f"🎉 BATCH: {batch_size} samples | reward∈[{rmin:.3f},{rmax:.3f}]")
                    except Exception:
                        logging.info(f"🎉 BATCH: {batch_size} samples")
                else:
                    logging.info(f"🎉 BATCH: {batch_size} samples")

                # Add to replay buffer
                self.add_to_replay_buffer(R, fresh.get("user_data", {}))
                logging.info(f"📦 REPLAY BUFFER: Added {batch_size} samples (buffer size: {len(self.replay_buffer)}/{self.replay_buffer.maxlen})")

                # Mix with replay
                replay = self.sample_replay_data(len(R))
                if replay:
                    replay_size = len(replay["R"])
                    logging.info(f"🔄 REPLAY BUFFER: Sampling {replay_size} replay samples for mixing (ratio: {self.replay_ratio})")
                    mixed = self.mix_fresh_and_replay(fresh, replay)
                    logging.info(f"🎯 MIXED BATCH: {len(fresh['R'])} fresh + {replay_size} replay = {len(mixed['R'])} total samples")
                else:
                    logging.info(f"⚡ FRESH ONLY: No replay data available, using {batch_size} fresh samples only")
                    mixed = fresh

                # Enqueue (blocks if queue full)
                await self.batch_queue.put(mixed)

                # Debug queue/replay utilization
                stats = self.get_replay_stats()
                utilization_pct = 100 * stats['utilization']
                logging.debug(
                    f"Queued mixed batch | queue={self.batch_queue.qsize()} | replay buffer: {stats['size']}/{stats['max_size']} ({utilization_pct:.1f}% full)"
                )

            except Exception as e:
                logging.error(f"Producer error: {e}")
                await asyncio.sleep(1.0)

        logging.info("Producer loop stopped")

    async def consumer_loop(self) -> None:
        logging.info("Consumer loop started")
        local_step = 0
        while self.training_active:
            try:
                batch = await self.batch_queue.get()

                result = await self.call_learner_train_step(batch)
                if result and result.get("success"):
                    local_step = result.get("step", local_step + 1)

                    # Update metrics with actual values
                    await self.update_metrics(
                        step=local_step,
                        loss=result.get("loss"),
                        reward_signal=result.get("reward_signal"),
                        active_attributes=result.get("active_attributes"),
                        temperature=None,  # polled periodically below
                    )

                    if local_step % 10 == 0:
                        logging.info(f"Trained step {local_step} | queue={self.batch_queue.qsize()}")
                else:
                    logging.warning(f"Training step {local_step} failed")

                self.batch_queue.task_done()

            except Exception as e:
                logging.error(f"Consumer error: {e}")
                await asyncio.sleep(1.0)

        logging.info(f"Consumer loop stopped at step {local_step}")

    # ---------- Monitoring ----------

    async def update_metrics(
        self,
        step: int,
        loss: float | None = None,
        reward_signal: float | None = None,
        active_attributes: float | None = None,
        temperature: float | None = None,
    ) -> None:
        if not self.enable_monitoring:
            return

        now = time.time()
        if self.start_time is None:
            self.start_time = now
        t = now - self.start_time

        self.metrics["timestamps"].append(t)
        self.metrics["steps"].append(float(step))
        self.metrics["losses"].append(float(loss or 0.0))
        self.metrics["reward_signals"].append(float(reward_signal or 0.0))
        self.metrics["active_attributes"].append(float(active_attributes or 0.0))
        self.metrics["temperatures"].append(float(temperature or 1.0))
        self.metrics["queue_sizes"].append(float(self.batch_queue.qsize()))
        self.metrics["replay_buffer_sizes"].append(float(len(self.replay_buffer)))

        if self.enable_wandb and self.wandb_run:
            self.wandb_run.log(
                {
                    "step": step,
                    "loss": loss or 0.0,
                    "reward_signal": reward_signal or 0.0,
                    "active_attributes": active_attributes or 0.0,
                    "temperature": temperature or 1.0,
                    "queue_size": self.batch_queue.qsize(),
                    "replay_buffer_size": len(self.replay_buffer),
                    "timestamp": t,
                },
                step=step,
            )

    def setup_monitoring(self) -> None:
        if not self.enable_monitoring:
            return

        if self.enable_wandb:
            try:
                self.wandb_run = wandb.init(
                    project="distributed-sparse-attributes",
                    name=f"coordinator-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    config={
                        "users_per_batch": self.users_per_batch,
                        "samples_per_user": self.samples_per_user,
                        "replay_buffer_size": self.replay_buffer.maxlen,
                        "replay_ratio": self.replay_ratio,
                        "collector_url": self.collector_url,
                        "learner_url": self.learner_url,
                    },
                )
                logging.info("W&B init ok")
            except Exception as e:
                logging.error(f"W&B init failed: {e}")
                self.enable_wandb = False



    def start_monitoring_thread(self) -> None:
        if not self.enable_monitoring:
            return

        def _loop():
            while self.training_active:
                try:
                    time.sleep(1.0)
                except Exception as e:
                    logging.debug(f"Monitoring thread error: {e}")
                    time.sleep(5.0)

        self.monitoring_thread = threading.Thread(target=_loop, daemon=True)
        self.monitoring_thread.start()
        logging.info("Monitoring thread started")

    def save_monitoring_results(self) -> None:
        if not self.enable_monitoring or not self.metrics["timestamps"]:
            return
        try:
            metrics_file = f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=2)
            logging.info(f"Saved metrics: {metrics_file}")

        except Exception as e:
            logging.error(f"Failed to save monitoring results: {e}")

    # ---------- Top-level training ----------

    async def start_async_training(self, max_steps: int = 1000, log_freq: int = 20, checkpoint_freq: int = 100) -> bool:
        logging.info(f"Starting async training: max_steps={max_steps}")
        self.setup_monitoring()
        self.start_monitoring_thread()

        self.training_active = True
        self.producer_task = asyncio.create_task(self.producer_loop())
        self.consumer_task = asyncio.create_task(self.consumer_loop())

        step_seen = 0
        try:
            while step_seen < max_steps and self.training_active:
                await asyncio.sleep(5.0)

                # Poll learner status & params to track temperature and active features
                status = await self.get_status("learner")
                if status:
                    current_step = int(status.get("current_step", step_seen))
                    step_seen = max(step_seen, current_step)
                    active_features = float(status.get("active_features", 0.0))

                    params = await self.get_learner_params()
                    tau = float(params.get("tau", 1.0)) if params and params.get("success") else 1.0

                    # Update metrics; loss/reward_signal unknown here (set to 0.0)
                    await self.update_metrics(
                        step=step_seen,
                        loss=None,
                        reward_signal=None,
                        active_attributes=active_features,
                        temperature=tau,
                    )

                    if step_seen % log_freq == 0:
                        rep = self.get_replay_stats()
                        replay_status = f"📊 REPLAY BUFFER: {rep['size']}/{rep['max_size']} ({100*rep['utilization']:.1f}% full, {rep['unique_users']} unique users)"
                        logging.info(
                            f"Step {step_seen} | active={active_features:.1f} | tau={tau:.3f} | "
                            f"queue={self.batch_queue.qsize()} | {replay_status}"
                        )

            # Stop loops
            self.training_active = False

            # Drain tasks
            if self.producer_task:
                await self.producer_task
            if self.consumer_task:
                await self.consumer_task

            self.save_monitoring_results()
            if self.enable_wandb and self.wandb_run:
                try:
                    self.wandb_run.finish()
                except Exception:
                    pass

            logging.info(f"Training completed at step {step_seen}")
            return True

        except Exception as e:
            logging.error(f"Training error: {e}")
            self.training_active = False
            return False

    async def run_training(self, max_steps: int = 1000, log_freq: int = 20, checkpoint_freq: int = 100) -> bool:
        try:
            await self.connect_to_servers()
            success = await self.start_async_training(max_steps, log_freq, checkpoint_freq)
            return success
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            return False
        except Exception as e:
            logging.error(f"run_training failed: {e}")
            return False
        finally:
            self.training_active = False
            # Close HTTP session
            try:
                if self.http and not self.http.closed:
                    await self.http.close()
            except Exception:
                pass
            # Join monitor thread (daemon; should exit automatically)


# ============ CLI / Main ============

def main():
    parser = argparse.ArgumentParser(description="Distributed Sparse Attribute Learning - Coordinator")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file path")

    # Server overrides
    parser.add_argument("--collector-url", type=str, help="Collector server URL (override)")
    parser.add_argument("--learner-url", type=str, help="Learner server URL (override)")

    # Training overrides
    parser.add_argument("--max-steps", type=int, help="Max training steps (override)")
    parser.add_argument("--log-freq", type=int, help="Logging frequency (override)")
    parser.add_argument("--checkpoint-freq", type=int, help="Checkpoint frequency (override)")

    # Monitoring overrides
    parser.add_argument("--enable-wandb-coordinator", action="store_true", help="Enable W&B on coordinator")
    parser.add_argument("--disable-monitoring", action="store_true", help="Disable monitoring")
    parser.add_argument("--plot-update-interval", type=float, help="Plot update interval (s)")
    parser.add_argument("--log-level", type=str, help="Log level")

    args = parser.parse_args()

    # Load external config (same interface you used)
    try:
        from ..utils.config_loader import load_config, ConfigLoader
        config = load_config(args.config)
        ConfigLoader.print_config_summary(config)
        print()
    except Exception as e:
        print(f"Error loading config '{args.config}': {e}")
        print("Ensure the config file exists and is valid JSON.")
        sys.exit(1)

    # Logging
    log_level = args.log_level or config["monitoring"]["log_level"]
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Apply coordinator config + overrides
    coord_cfg = ConfigLoader.get_coordinator_config(config)
    if args.collector_url:
        coord_cfg["collector_url"] = args.collector_url
    if args.learner_url:
        coord_cfg["learner_url"] = args.learner_url
    if args.max_steps is not None:
        coord_cfg["max_steps"] = args.max_steps
    if args.log_freq is not None:
        coord_cfg["log_freq"] = args.log_freq
    if args.checkpoint_freq is not None:
        coord_cfg["checkpoint_freq"] = args.checkpoint_freq
    if args.plot_update_interval is not None:
        coord_cfg["plot_update_interval"] = args.plot_update_interval
    if args.enable_wandb_coordinator:
        coord_cfg["enable_wandb"] = True
    if args.disable_monitoring:
        coord_cfg["enable_monitoring"] = False

    # Optional uvloop
    if UVLOOP:
        try:
            uvloop.install()
        except Exception:
            pass

    # Build coordinator
    coordinator = ServerCoordinator(
        collector_url=coord_cfg["collector_url"],
        learner_url=coord_cfg["learner_url"],
        queue_size=coord_cfg["queue_size"],
        replay_buffer_size=coord_cfg["replay_buffer_size"],
        replay_ratio=coord_cfg["replay_ratio"],
        enable_monitoring=coord_cfg["enable_monitoring"],
        enable_wandb=coord_cfg["enable_wandb"],
        plot_update_interval=coord_cfg["plot_update_interval"],
        timeouts=coord_cfg["timeouts"],
        http_concurrency=coord_cfg.get("http_concurrency", 128),
        http_total_timeout=coord_cfg.get("http_total_timeout", None),
    )

    logging.info("=== Distributed Sparse Attribute Learning ===")
    logging.info(f"Config:   {args.config}")
    logging.info(f"Dataset:  {config['data']['dataset_path']}")
    logging.info(f"Model:    {config['model']['d']} attrs -> {config['model']['k']} comps")
    logging.info(f"Collector URL: {coord_cfg['collector_url']}")
    logging.info(f"Learner   URL: {coord_cfg['learner_url']}")
    logging.info(f"Training steps: {coord_cfg['max_steps']}")

    async def _run():
        return await coordinator.run_training(
            max_steps=coord_cfg["max_steps"],
            log_freq=coord_cfg["log_freq"],
            checkpoint_freq=coord_cfg["checkpoint_freq"],
        )

    try:
        ok = asyncio.run(_run())
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        logging.info("Coordinator interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
