#!/usr/bin/env python3
"""
Experiment Runner for Gumbel Distributed Training
Starts all components from a single config file and manages the experiment lifecycle.
"""

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add gumbel to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gumbel.utils.config_loader import load_config, ConfigLoader


class ExperimentRunner:
    """Manages the lifecycle of a distributed training experiment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n🛑 Received signal {signum}, shutting down experiment...")
        self.stop_all()
        sys.exit(0)
    
    def start_vllm_server(self) -> bool:
        """Start VLLM server if configured."""
        if not self.config.get("vllm", {}).get("start_server", True):
            logging.info("VLLM server startup disabled in config")
            return True
        
        vllm_config = self.config["vllm"]
        model_name = vllm_config["model_name"]
        port = vllm_config["server_url"].split(":")[-1]
        gpu_memory_util = vllm_config.get("gpu_memory_util", 0.6)
        
        cmd = [
            "vllm", "serve", model_name,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_util),
            "--disable-log-requests"  # Reduce log noise
        ]
        
        logging.info(f"🚀 Starting VLLM server: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes["vllm"] = process
            
            # Wait for VLLM to start (basic check)
            time.sleep(10)
            if process.poll() is None:
                logging.info("✅ VLLM server started successfully")
                return True
            else:
                stdout, stderr = process.communicate()
                logging.error(f"❌ VLLM server failed to start: {stderr}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Failed to start VLLM server: {e}")
            return False
    
    def start_learner_server(self) -> bool:
        """Start learner server."""
        cmd = [
            sys.executable, "-m", "gumbel.core.learner_server",
            "--config", str(self.config_path)
        ]
        
        logging.info(f"🧠 Starting learner server: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            self.processes["learner"] = process
            
            # Wait a bit for startup
            time.sleep(3)
            if process.poll() is None:
                logging.info("✅ Learner server started successfully")
                return True
            else:
                stdout, _ = process.communicate()
                logging.error(f"❌ Learner server failed to start: {stdout}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Failed to start learner server: {e}")
            return False
    
    def start_collector_server(self) -> bool:
        """Start collector server."""
        cmd = [
            sys.executable, "-m", "gumbel.core.collector_server",
            "--config", str(self.config_path)
        ]
        
        logging.info(f"📊 Starting collector server: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            self.processes["collector"] = process
            
            # Wait a bit for startup
            time.sleep(3)
            if process.poll() is None:
                logging.info("✅ Collector server started successfully")
                return True
            else:
                stdout, _ = process.communicate()
                logging.error(f"❌ Collector server failed to start: {stdout}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Failed to start collector server: {e}")
            return False
    
    def start_coordinator(self) -> bool:
        """Start coordinator."""
        cmd = [
            sys.executable, "-m", "gumbel.core.coordinator",
            "--config", str(self.config_path)
        ]
        
        logging.info(f"🎯 Starting coordinator: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            self.processes["coordinator"] = process
            
            logging.info("✅ Coordinator started successfully")
            return True
                
        except Exception as e:
            logging.error(f"❌ Failed to start coordinator: {e}")
            return False
    
    def check_processes(self) -> Dict[str, bool]:
        """Check which processes are still running."""
        status = {}
        for name, process in self.processes.items():
            if process.poll() is None:
                status[name] = True
            else:
                status[name] = False
                # Log any errors from failed processes
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    if stderr:
                        logging.error(f"❌ {name} failed: {stderr}")
                except:
                    pass
        return status
    
    def stop_all(self):
        """Stop all processes gracefully."""
        logging.info("🛑 Stopping all processes...")
        
        # Stop in reverse order (coordinator first, VLLM last)
        stop_order = ["coordinator", "collector", "learner", "vllm"]
        
        for name in stop_order:
            if name in self.processes:
                process = self.processes[name]
                if process.poll() is None:  # Still running
                    logging.info(f"Stopping {name}...")
                    try:
                        process.terminate()
                        process.wait(timeout=10)
                        logging.info(f"✅ {name} stopped")
                    except subprocess.TimeoutExpired:
                        logging.warning(f"⚠️  {name} didn't stop gracefully, forcing...")
                        process.kill()
                        process.wait()
                        logging.info(f"🔪 {name} force stopped")
                    except Exception as e:
                        logging.error(f"❌ Error stopping {name}: {e}")
        
        self.processes.clear()
        self.running = False
    
    def run_experiment(self, config_path: str) -> bool:
        """Run the complete experiment."""
        self.config_path = Path(config_path)
        
        logging.info("=" * 60)
        logging.info(f"🚀 STARTING EXPERIMENT: {self.config.get('experiment_name', 'Unnamed')}")
        logging.info("=" * 60)
        
        ConfigLoader.print_config_summary(self.config)
        
        try:
            # Start servers in order
            if not self.start_vllm_server():
                return False
            
            if not self.start_learner_server():
                return False
            
            if not self.start_collector_server():
                return False
            
            # Small delay before starting coordinator
            time.sleep(2)
            
            if not self.start_coordinator():
                return False
            
            self.running = True
            
            # Monitor the experiment
            logging.info("🔍 Monitoring experiment progress...")
            
            coordinator_process = self.processes["coordinator"]
            
            # Wait for coordinator to finish or fail
            while coordinator_process.poll() is None:
                time.sleep(5)
                
                # Check all processes
                status = self.check_processes()
                if not all(status.values()):
                    failed = [name for name, running in status.items() if not running]
                    logging.error(f"❌ Processes failed: {failed}")
                    break
            
            # Get coordinator exit code
            coordinator_exit_code = coordinator_process.poll()
            
            if coordinator_exit_code == 0:
                logging.info("🎉 Experiment completed successfully!")
                return True
            else:
                # Try to get output from coordinator
                try:
                    stdout, _ = coordinator_process.communicate(timeout=1)
                    if stdout:
                        logging.error(f"Coordinator output: {stdout[-500:]}")  # Last 500 chars
                except:
                    pass
                
                logging.error(f"❌ Experiment failed (coordinator exit code: {coordinator_exit_code})")
                return False
                
        except KeyboardInterrupt:
            logging.info("🛑 Experiment interrupted by user")
            return False
        except Exception as e:
            logging.error(f"❌ Experiment failed with error: {e}")
            return False
        finally:
            self.stop_all()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Gumbel distributed training experiment")
    parser.add_argument("config", help="Path to experiment config file (YAML or JSON)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    try:
        # Load and validate config
        config = load_config(args.config)
        
        if args.dry_run:
            ConfigLoader.print_config_summary(config)
            return
        
        # Run experiment
        runner = ExperimentRunner(config)
        success = runner.run_experiment(args.config)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logging.error(f"❌ Failed to run experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()