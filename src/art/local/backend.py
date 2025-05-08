from datetime import datetime
import asyncio
import atexit
import json
import math
from art.utils.old_benchmarking.calculate_step_metrics import calculate_step_std_dev
from art.utils.output_dirs import (
    get_default_art_path,
    get_model_dir,
    get_trajectories_split_dir,
)
from art.utils.trajectory_logging import serialize_trajectory_groups
from mp_actors import move_to_child_process
import multiprocessing as mp
import numpy as np
import os
import polars as pl
import psutil
import signal
import subprocess
import sys
import threading
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from tqdm import auto as tqdm
from typing import AsyncIterator, cast, List
import wandb
from wandb.sdk.wandb_run import Run

# Track all backend instances to ensure cleanup on exit and signals
_backend_instances = []

# Track the resource tracker processes
_resource_tracker_processes: List[int] = []

# Function to find and kill resource tracker processes
def _kill_resource_trackers():
    # First try the ones we've tracked
    global _resource_tracker_processes
    for pid in _resource_tracker_processes:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"Killed tracked resource tracker process {pid}")
        except Exception:
            pass
    
    # Try three approaches to find resource trackers
    try:
        # 1. Use psutil to find direct children
        try:
            current_process = psutil.Process(os.getpid())
            for child in current_process.children(recursive=True):
                try:
                    cmdline = " ".join(child.cmdline())
                    if "resource_tracker" in cmdline:
                        print(f"Killing resource tracker child process (psutil): {child.pid}")
                        child.kill()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error using psutil to find resource trackers: {e}")
            
        # 2. Use ps command to find related processes
        try:
            ps_output = subprocess.check_output(["ps", "-ef"], text=True)
            parent_pid = os.getpid()
            for line in ps_output.splitlines():
                if str(parent_pid) in line and "resource_tracker" in line:
                    try:
                        parts = line.split()
                        tracker_pid = int(parts[1])
                        print(f"Killing resource tracker process (ps): {tracker_pid}")
                        os.kill(tracker_pid, signal.SIGKILL)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error using ps to find resource trackers: {e}")
            
        # 3. Use multiprocessing module's active_children
        try:
            import multiprocessing as mp
            for child in mp.active_children():
                try:
                    if "resource_tracker" in child.name:
                        print(f"Killing resource tracker process (mp): {child.pid}")
                        child.terminate()
                        child.join(timeout=0.5)
                        if child.is_alive():
                            child.kill()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error using mp.active_children to find resource trackers: {e}")
            
    except Exception as e:
        print(f"Error during resource tracker cleanup: {e}")
        
    # Last resort - try to kill the module itself
    try:
        import multiprocessing.resource_tracker as rt
        if hasattr(rt, "_resource_tracker") and rt._resource_tracker is not None:
            print("Shutting down resource tracker module")
            # Try to use its own shutdown
            if hasattr(rt, "_resource_tracker") and hasattr(rt._resource_tracker, "close"):
                rt._resource_tracker.close()
            # Make sure it doesn't restart
            rt._resource_tracker = None
    except Exception as e:
        print(f"Error shutting down resource tracker module: {e}")

# Set up a flag to track if we've done a final cleanup
_final_cleanup_done = False

# Always start the watchdog at import time
threading.Timer(30, lambda: os._exit(0)).start()

# Also hook into asyncio.run to ensure we start an earlier watchdog when it completes
original_asyncio_run = asyncio.run
def _patched_asyncio_run(main, *, debug=None):
    try:
        result = original_asyncio_run(main, debug=debug)
        # Start watchdog when asyncio.run completes
        _cleanup_all_backends()
        _start_exit_watchdog()
        return result
    except Exception as e:
        # Also ensure cleanup on exception
        _cleanup_all_backends()
        _start_exit_watchdog()
        raise e
# Replace asyncio.run with our patched version
asyncio.run = _patched_asyncio_run

# Function to clean up all backends on exit
def _cleanup_all_backends():
    global _final_cleanup_done
    
    if _final_cleanup_done:
        return
    
    print(f"Cleaning up {len(_backend_instances)} backend instances...")
    
    # First, try to clean up each backend
    for backend in list(_backend_instances):
        if hasattr(backend, "_cleanup_sync"):
            try:
                backend._cleanup_sync()
            except Exception as e:
                print(f"Error cleaning up backend: {e}")
    
    # Force cleanup of resource tracker and model-service processes
    try:
        # Kill model-service processes
        subprocess.run(["pkill", "-9", "model-service"], check=False)
        # Kill resource trackers
        _kill_resource_trackers()
    except Exception as e:
        print(f"Error during final process cleanup: {e}")
    
    # Clear the global list
    _backend_instances.clear()
    
    # Try to cancel any remaining asyncio tasks
    try:
        import asyncio
        if hasattr(asyncio, 'all_tasks'):
            tasks = asyncio.all_tasks()
            for task in tasks:
                if not task.done() and not task.cancelled():
                    task.cancel()
    except Exception as e:
        print(f"Error cancelling asyncio tasks: {e}")
    
    # Try to shutdown multiprocessing module gracefully 
    try:
        import multiprocessing as mp
        if hasattr(mp, 'get_context'):
            ctx = mp.get_context()
            if hasattr(ctx, '_close_process_pool'):
                ctx._close_process_pool()
    except Exception as e:
        print(f"Error shutting down multiprocessing: {e}")
        
    # Last resort - try to kill known multiprocessing internals directly
    try:
        # Try to kill the semaphore tracker
        try:
            import multiprocessing.semaphore_tracker as st
            if hasattr(st, '_semaphore_tracker') and st._semaphore_tracker is not None:
                if hasattr(st._semaphore_tracker, 'stop'):
                    st._semaphore_tracker.stop()
                st._semaphore_tracker = None
                print("Shutdown semaphore tracker")
        except Exception:
            pass
            
        # Try to kill the forkserver
        try:
            import multiprocessing.forkserver as fs
            if hasattr(fs, '_forkserver') and fs._forkserver is not None:
                if hasattr(fs._forkserver, 'close'):
                    fs._forkserver.close()
                fs._forkserver = None
                print("Shutdown fork server")
        except Exception:
            pass
    except Exception:
        pass
    
    _final_cleanup_done = True

# Add a watchdog thread that will force exit if the process is hanging
def _force_exit_thread():
    import time
    import os
    import sys
    
    # Wait for a timeout period after which we assume the process is hanging
    time.sleep(5)  # Give normal cleanup 5 seconds to complete
    
    try:
        # If we're still running after timeout, something is hung
        print("\n=== WATCHDOG: Process appears to be hanging, forcing exit ===")
        # Simply force exit the process
        print("WATCHDOG: Invoking os._exit() to forcefully terminate the process")
        os._exit(0)
    except Exception as e:
        print(f"Error in watchdog thread: {e}")
        os._exit(1)

# Setup for the watchdog
_exit_watchdog = None

def _start_exit_watchdog():
    global _exit_watchdog
    if _exit_watchdog is None or not _exit_watchdog.is_alive():
        _exit_watchdog = threading.Thread(
            target=_force_exit_thread,
            daemon=True,
            name="ExitWatchdog"
        )
        _exit_watchdog.start()
        print("Started exit watchdog thread")

# Register cleanup on normal exit
atexit.register(_cleanup_all_backends)

# Special safety mechanism to ensure we don't hang on exit
original_exit = sys.exit
def _safe_exit(code=0):
    # Trigger cleanup
    _cleanup_all_backends()
    # Start the watchdog to ensure we really exit
    _start_exit_watchdog()
    # Call original exit
    original_exit(code)
# Replace the builtin exit
sys.exit = _safe_exit

# Set up signal handlers for SIGINT (Ctrl+C) and SIGTERM
def _signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, cleaning up...")
    try:
        # Kill all model-service processes immediately
        subprocess.run(["pkill", "-9", "model-service"], check=False)
        
        # Kill resource trackers
        _kill_resource_trackers()
        
        # Then try the more graceful cleanup
        _cleanup_all_backends()
        
        # Start watchdog to ensure we really exit
        _start_exit_watchdog()
    except Exception as e:
        print(f"Error during signal cleanup: {e}")
    finally:
        # Re-raise the signal to allow normal Python signal handling to continue
        # (this ensures the default handler can still terminate the process)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

# Register signal handlers
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, _signal_handler)

from .. import dev
from ..backend import Backend
from ..model import Model, TrainableModel
from .service import ModelService
from ..trajectories import Trajectory, TrajectoryGroup
from ..types import Message, TrainConfig
from ..utils import format_message
from .pack import (
    packed_tensors_from_tokenized_results,
    packed_tensors_to_dir,
    PackedTensors,
    plot_packed_tensors,
)
from .tokenize import tokenize_trajectory_groups
from .checkpoints import (
    delete_checkpoints,
    get_step,
)
from art.utils.s3 import pull_model_from_s3, push_model_to_s3


class LocalBackend(Backend):
    def __init__(self, *, in_process: bool = False, path: str | None = None) -> None:
        """
        Initializes a local, directory-based Backend interface at the given path.

        Note:
            The local Backend uses Weights & Biases for training monitoring.
            If you don't have a W&B account, you can create one at https://wandb.ai.

        Args:
            in_process: Whether to run the local service in-process.
            path: The path to the local directory. Defaults to "{repo_root}/.art".
        """
        self._in_process = in_process
        self._path = path or get_default_art_path()
        os.makedirs(self._path, exist_ok=True)

        # Other initialization
        self._services: dict[str, ModelService] = {}
        self._tokenizers: dict[str, "PreTrainedTokenizerBase"] = {}
        self._wandb_runs: dict[str, Run] = {}
        
        # Register this instance for global cleanup
        global _backend_instances
        _backend_instances.append(self)

    async def register(
        self,
        model: Model,
    ) -> None:
        """
        Registers a model with the local Backend for logging and/or training.

        Args:
            model: An art.Model instance.
        """
        output_dir = get_model_dir(model=model, art_path=self._path)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/model.json", "w") as f:
            json.dump(model.model_dump(), f)

    async def _get_service(self, model: TrainableModel) -> ModelService:
        if model.name not in self._services:
            config = dev.get_model_config(
                base_model=model.base_model,
                output_dir=get_model_dir(model=model, art_path=self._path),
                config=model._internal_config,
            )
            self._services[model.name] = ModelService(
                host="localhost",
                port=8089 + len(self._services),
                model_name=model.name,
                base_model=model.base_model,
                config=config,
                output_dir=get_model_dir(model=model, art_path=self._path),
            )
            if not self._in_process:
                # Kill all "model-service" processes to free up GPU memory
                subprocess.run(["pkill", "-9", "model-service"])
                # To enable sleep mode, import peft before unsloth
                # Unsloth will issue warnings, but everything appears to be okay
                if config.get("engine_args", {}).get("enable_sleep_mode", False):
                    os.environ["IMPORT_PEFT"] = "1"
                # When moving the service to a child process, import unsloth
                # early to maximize optimizations
                os.environ["IMPORT_UNSLOTH"] = "1"
                
                # Find any resource tracker processes that may exist before creating a new one
                self._track_resource_trackers()
                
                self._services[model.name] = move_to_child_process(
                    self._services[model.name],
                    process_name="model-service",
                )
                
                # Find and track new resource tracker processes that were created
                self._track_resource_trackers()
                
        return self._services[model.name]
        
    def _track_resource_trackers(self) -> None:
        """Find and track resource tracker processes to ensure they get cleaned up."""
        global _resource_tracker_processes
        
        try:
            # Method 1: Find resource tracker processes using psutil
            try:
                current_process = psutil.Process(os.getpid())
                for child in current_process.children(recursive=True):
                    try:
                        cmdline = " ".join(child.cmdline())
                        if "resource_tracker" in cmdline and child.pid not in _resource_tracker_processes:
                            _resource_tracker_processes.append(child.pid)
                            print(f"Tracking resource tracker process: {child.pid} ({cmdline})")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error tracking resource trackers with psutil: {e}")
                
            # Method 2: Use ps command to find related processes
            try:
                ps_output = subprocess.check_output(["ps", "-ef"], text=True)
                parent_pid = os.getpid()
                for line in ps_output.splitlines():
                    if str(parent_pid) in line and "resource_tracker" in line:
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                tracker_pid = int(parts[1])
                                if tracker_pid not in _resource_tracker_processes:
                                    _resource_tracker_processes.append(tracker_pid)
                                    print(f"Tracking resource tracker process (ps): {tracker_pid}")
                        except Exception:
                            pass
            except Exception as e:
                print(f"Error tracking resource trackers with ps: {e}")
                
        except Exception as e:
            print(f"Error tracking resource trackers: {e}")

    def _get_packed_tensors(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        plot_tensors: bool,
    ) -> PackedTensors | None:
        if not model.base_model in self._tokenizers:
            self._tokenizers[model.base_model] = AutoTokenizer.from_pretrained(
                model.base_model
            )
        tokenizer = self._tokenizers[model.base_model]
        tokenized_results = list(
            tokenize_trajectory_groups(
                tokenizer,
                trajectory_groups,
            )
        )
        if not tokenized_results:
            return None
        max_tokens = max(len(result.tokens) for result in tokenized_results)
        # Round up max_tokens to the nearest multiple of 2048
        sequence_length = math.ceil(max_tokens / 2048) * 2048
        packed_tensors = packed_tensors_from_tokenized_results(
            tokenized_results,
            sequence_length,
            pad_token_id=tokenizer.eos_token_id,  # type: ignore
        )
        # If all logprobs are NaN then there is no suitable data for tuning
        if np.isnan(packed_tensors["logprobs"]).all():
            print(
                "There are no assistant logprobs to train on. Did you forget to include at least one Choice in Trajectory.messages_and_choices?"
            )
            return None
        if plot_tensors:
            plot_packed_tensors(packed_tensors)
        else:
            print(
                f"Packed {len(tokenized_results)} trajectories into {packed_tensors['tokens'].shape[0]} sequences of length {packed_tensors['tokens'].shape[1]}"
            )
        return packed_tensors

    async def _get_step(self, model: TrainableModel) -> int:
        return self.__get_step(model)

    def __get_step(self, model: TrainableModel) -> int:
        return get_step(get_model_dir(model=model, art_path=self._path))

    async def _delete_checkpoints(
        self,
        model: TrainableModel,
        benchmark: str,
        benchmark_smoothing: float,
    ) -> None:
        output_dir = get_model_dir(model=model, art_path=self._path)
        # Keep the latest step
        steps_to_keep = [get_step(output_dir)]
        try:
            best_step = (
                pl.read_ndjson(f"{output_dir}/history.jsonl")
                .drop_nulls(subset=[benchmark])
                .group_by("step")
                .mean()
                .with_columns(pl.col(benchmark).ewm_mean(alpha=benchmark_smoothing))
                .sort(benchmark)
                .select(pl.col("step").last())
                .item()
            )
            steps_to_keep.append(best_step)
        except FileNotFoundError:
            pass
        except pl.exceptions.ColumnNotFoundError:
            print(f'No "{benchmark}" metric found in history')
        delete_checkpoints(output_dir, steps_to_keep)

    async def _prepare_backend_for_training(
        self,
        model: TrainableModel,
        config: dev.OpenAIServerConfig | None = None,
    ) -> tuple[str, str]:
        service = await self._get_service(model)
        await service.start_openai_server(config=config)
        server_args = (config or {}).get("server_args", {})

        base_url = f"http://{server_args.get('host', '0.0.0.0')}:{server_args.get('port', 8000)}/v1"
        api_key = server_args.get("api_key", None) or "default"

        return base_url, api_key

    async def _log(
        self,
        model: Model,
        trajectory_groups: list[TrajectoryGroup],
        split: str = "val",
    ) -> None:
        # Save logs for trajectory groups
        parent_dir = get_trajectories_split_dir(
            get_model_dir(model=model, art_path=self._path), split
        )
        os.makedirs(parent_dir, exist_ok=True)

        # Get the file name for the current iteration, or default to 0 for non-trainable models
        iteration = self.__get_step(model) if isinstance(model, TrainableModel) else 0
        file_name = f"{iteration:04d}.yaml"

        # Write the logs to the file
        with open(f"{parent_dir}/{file_name}", "w") as f:
            f.write(serialize_trajectory_groups(trajectory_groups))

        # Collect all metrics (including reward) across all trajectories
        all_metrics: dict[str, list[float]] = {"reward": [], "exception_rate": []}

        for group in trajectory_groups:
            for trajectory in group:
                if isinstance(trajectory, BaseException):
                    all_metrics["exception_rate"].append(1)
                    continue
                else:
                    all_metrics["exception_rate"].append(0)
                # Add reward metric
                all_metrics["reward"].append(trajectory.reward)

                # Collect other custom metrics
                for metric, value in trajectory.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(float(value))

        # Calculate averages for all metrics
        averages = {}
        for metric, values in all_metrics.items():
            if len(values) > 0:
                averages[metric] = sum(values) / len(values)

        # Calculate average standard deviation of rewards within groups
        averages["reward_std_dev"] = calculate_step_std_dev(trajectory_groups)

        if isinstance(model, TrainableModel):
            self._log_metrics(model, averages, split)

    def _trajectory_log(self, trajectory: Trajectory) -> str:
        """Format a trajectory into a readable log string."""
        header = f"reward: {trajectory.reward} {' '.join(f'{k}: {v}' for k, v in trajectory.metrics.items())}\n\n"
        formatted_messages = []
        for message_or_choice in trajectory.messages_and_choices:
            if isinstance(message_or_choice, dict):
                message = message_or_choice
            else:
                message = cast(Message, message_or_choice.message.model_dump())
            formatted_messages.append(format_message(message))
        return header + "\n".join(formatted_messages)

    async def _train_model(
        self,
        model: TrainableModel,
        trajectory_groups: list[TrajectoryGroup],
        config: TrainConfig,
        dev_config: dev.TrainConfig,
    ) -> AsyncIterator[dict[str, float]]:
        service = await self._get_service(model)
        await self._log(model, trajectory_groups, "train")
        packed_tensors = self._get_packed_tensors(
            model, trajectory_groups, plot_tensors=False
        )
        if packed_tensors is None:
            print(
                "Skipping tuning as there is no suitable data. "
                "This can happen when all the trajectories in the same group "
                "have the same reward and thus no advantage to train on."
            )
            return
        disk_packed_tensors = packed_tensors_to_dir(
            packed_tensors, f"{get_model_dir(model=model, art_path=self._path)}/tensors"
        )
        results: list[dict[str, float]] = []
        num_gradient_steps = disk_packed_tensors["num_sequences"]
        pbar = tqdm.tqdm(total=num_gradient_steps, desc="train")
        async for result in service.train(disk_packed_tensors, config, dev_config):
            results.append(result)
            yield {**result, "num_gradient_steps": num_gradient_steps}
            pbar.update(1)
            pbar.set_postfix(result)
        pbar.close()
        data = {
            k: sum(d.get(k, 0) for d in results) / sum(1 for d in results if k in d)
            for k in {k for d in results for k in d}
        }
        self._log_metrics(model, data, "train", step_offset=-1)

    def _log_metrics(
        self,
        model: TrainableModel,
        metrics: dict[str, float],
        split: str,
        step_offset: int = 0,
    ) -> None:
        # Add namespacing if needed
        metrics = (
            {f"{split}/{metric}": value for metric, value in metrics.items()}
            if split
            else metrics
        )
        step = (
            self.__get_step(model) if isinstance(model, TrainableModel) else 0
        ) + step_offset

        # If we have a W&B run, log the data there
        if run := self._get_wandb_run(model):
            run.log(
                metrics,
                step=step,
            )

    def _get_wandb_run(self, model: TrainableModel) -> Run | None:
        if "WANDB_API_KEY" not in os.environ:
            return None
        if (
            model.name not in self._wandb_runs
            or self._wandb_runs[model.name]._is_finished
        ):
            run = wandb.init(
                project=model.project,
                name=model.name,
                id=model.name,
                resume="allow",
            )
            self._wandb_runs[model.name] = run
            print(f"Wandb run initialized! You can view it at {run.url}")
        return self._wandb_runs[model.name]

    # ------------------------------------------------------------------
    # Experimental support for S3
    # ------------------------------------------------------------------

    async def _experimental_pull_from_s3(
        self,
        model: Model,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        """Download the model directory from S3 into local Backend storage. Right now this can be used to pull trajectory logs for processing or model checkpoints."""
        await pull_model_from_s3(
            model_name=model.name,
            project=model.project,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
            art_path=self._path,
        )

    async def _experimental_push_to_s3(
        self,
        model: Model,
        s3_bucket: str | None = None,
        prefix: str | None = None,
        verbose: bool = False,
        delete: bool = False,
    ) -> None:
        """Upload the model directory from local storage to S3."""
        await push_model_to_s3(
            model_name=model.name,
            project=model.project,
            s3_bucket=s3_bucket,
            prefix=prefix,
            verbose=verbose,
            delete=delete,
            art_path=self._path,
        )

    async def down(self) -> None:
        """
        Shut down the LocalBackend by terminating all services and child processes.
        
        This method ensures that all model services are stopped and child processes 
        are terminated properly, preventing hanging when a script finishes executing.
        """
        print("Shutting down LocalBackend...")
        
        # First try to gracefully stop OpenAI servers if possible
        try:
            for service_name, service in list(self._services.items()):
                if hasattr(service, "stop_openai_server"):
                    try:
                        await service.stop_openai_server()
                        print(f"Stopped OpenAI server for {service_name}")
                    except Exception as e:
                        print(f"Error stopping OpenAI server for {service_name}: {e}")
        except Exception as e:
            print(f"Error shutting down services: {e}")
            
        # Then use the sync cleanup for the rest
        self._cleanup_sync()
        
        print("LocalBackend shut down successfully")
    
    def _cleanup_sync(self) -> None:
        """
        Synchronous cleanup method that can be called from signal handlers or at exit.
        """
        print("Performing synchronous cleanup of LocalBackend...")
        
        # We need to be very careful with service references as they might have unpicklable objects
        # Don't interact with service objects directly - they can cause pickling errors
        
        # Kill model-service processes directly with pkill
        # This is safer than trying to access service objects which might have unpicklable queues
        try:
            subprocess.run(["pkill", "-9", "model-service"], check=False)
        except Exception as e:
            print(f"Error killing model-service processes: {e}")
        
        # Kill resource tracker processes
        try:
            _kill_resource_trackers()
        except Exception as e:
            print(f"Error killing resource trackers: {e}")
            
        # Attempt to clean up asyncio threads
        try:
            import asyncio
            # Try to cancel any asyncio tasks
            for task in asyncio.all_tasks() if hasattr(asyncio, 'all_tasks') else []:
                if not task.done():
                    task.cancel()
        except Exception as e:
            print(f"Error canceling asyncio tasks: {e}")
        
        # Close wandb runs
        try:
            for run in list(self._wandb_runs.values()):
                try:
                    run.finish()
                except Exception:
                    pass
        except Exception as e:
            print(f"Error finishing wandb runs: {e}")
        
        # Clear references (make a copy to avoid modifying during iteration)
        try:
            self._services = {}
            self._wandb_runs = {}
        except Exception as e:
            print(f"Error clearing references: {e}")
        
        # Remove self from global instances list
        try:
            global _backend_instances
            if self in _backend_instances:
                _backend_instances.remove(self)
        except Exception as e:
            print(f"Error removing from global instances: {e}")
            
        print("LocalBackend cleanup completed")
    
    def __del__(self) -> None:
        """
        Destructor that ensures cleanup happens automatically when the object is garbage collected.
        
        This helps prevent hanging when scripts finish without explicitly calling down().
        """
        try:
            self._cleanup_sync()
        except Exception:
            # Avoid errors during interpreter shutdown
            pass