import random
import asyncio
from dotenv import load_dotenv
import os
import signal
import subprocess
import threading
import time
import sys

import art
from rollout import rollout
from art.local.backend import LocalBackend


load_dotenv()

random.seed(42)

# Debug function to print thread and asyncio task info
def print_thread_info():
    print("\n--- THREAD DEBUG INFO ---")
    main_thread = threading.main_thread()
    print(f"Main thread: {main_thread.name}, alive: {main_thread.is_alive()}")
    
    print("\nAll threads:")
    for thread in threading.enumerate():
        print(f"  - {thread.name}, daemon: {thread.daemon}, alive: {thread.is_alive()}")
    
    # Try to get asyncio task info
    try:
        import asyncio
        tasks = asyncio.all_tasks() if hasattr(asyncio, 'all_tasks') else []
        
        print("\nActive asyncio tasks:")
        if not tasks:
            print("  No active asyncio tasks found")
        
        for i, task in enumerate(tasks):
            print(f"  Task {i+1}: {task}")
            print(f"    - Done: {task.done()}")
            print(f"    - Cancelled: {task.cancelled()}")
            try:
                # Get the current stack for this task (where it's blocked)
                task_stack = task.get_stack() if hasattr(task, 'get_stack') else []
                print(f"    - Stack frames: {len(task_stack)}")
                if task_stack:
                    import traceback
                    formatted_stack = ''.join(traceback.format_stack(task_stack[-1]))
                    print(f"    - Top of stack: {formatted_stack.strip()}")
            except Exception as e:
                print(f"    - Error getting stack: {e}")
    except Exception as e:
        print(f"Error getting asyncio task info: {e}")
    
    print("\nActive child processes:")
    try:
        ps_output = subprocess.check_output(["ps", "-ef"], text=True)
        parent_pid = os.getpid()
        print(f"Parent PID: {parent_pid}")
        
        for line in ps_output.splitlines():
            if "model-service" in line or "resource_tracker" in line or f"{parent_pid}" in line:
                print(f"  {line.strip()}")
    except Exception as e:
        print(f"Error getting process info: {e}")
    
    # Try to get open file descriptors
    try:
        fd_dir = f"/proc/{os.getpid()}/fd"
        if os.path.exists(fd_dir):
            print("\nOpen file descriptors:")
            fds = os.listdir(fd_dir)
            print(f"  Total open FDs: {len(fds)}")
            
            for fd in fds[:20]:  # Limit to first 20 to avoid too much output
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                    print(f"  {fd} -> {target}")
                except Exception:
                    pass
                    
            if len(fds) > 20:
                print(f"  ... and {len(fds) - 20} more")
    except Exception as e:
        print(f"Error getting file descriptor info: {e}")
    
    # Check multiprocessing module's context
    try:
        import multiprocessing as mp
        print("\nMultiprocessing info:")
        print(f"  Start method: {mp.get_start_method(allow_none=True)}")
        
        # Try to access and print active children
        try:
            active = mp.active_children()
            print(f"  Active children: {len(active)}")
            for child in active:
                print(f"    - {child.name}, daemon: {child.daemon}, pid: {child.pid}, alive: {child.is_alive()}")
        except Exception as e:
            print(f"  Error getting active children: {e}")
    except Exception as e:
        print(f"Error getting multiprocessing info: {e}")
        
    print("------------------------\n")

# Background thread to periodically print debug info
def background_debug_monitor():
    print("Starting background debug monitor")
    try:
        count = 0
        while True:
            time.sleep(10)  # Print debug info every 10 seconds
            count += 1
            print(f"\n==== PERIODIC DEBUG INFO (#{count}) ====")
            print_thread_info()
    except Exception as e:
        print(f"Background debug monitor error: {e}")

async def main():
    print("Starting test script")
    # run from the root of the repo
    backend = LocalBackend(path="examples/tic_tac_toe/.art")
    print("Backend created")

    model = art.TrainableModel(
        name="test-exit",
        project="tic-tac-toe-test",
        base_model="Qwen/Qwen2.5-3B-Instruct",
    )
    print("Model created")
    await model.register(backend)
    print("Model registered")

    # Just run one training iteration for quick testing
    train_groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(
                rollout(model, 0, is_validation=False) for _ in range(10)  # Reduce to 10 for faster testing
            )
            for _ in range(1)
        ),
        pbar_desc="gather",
    )
    print("Gathered training data")
    await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-4))
    print("Training completed")

    # Print debug info before exit
    print_thread_info()
    
    # Don't call down() - this is the scenario we want to test
    print("Main function completed without explicit cleanup")


if __name__ == "__main__":
    print("Script starting")
    
    # Start background monitoring
    monitor_thread = threading.Thread(
        target=background_debug_monitor, 
        daemon=True,
        name="BackgroundMonitor"
    )
    monitor_thread.start()
    
    try:
        asyncio.run(main())
        print("asyncio.run completed - script should exit naturally")
        print("Waiting 5 seconds to monitor post-completion state...")
        time.sleep(5)  # Allow time to show the final state
        
        print("Debug after waiting:")
        print_thread_info()
        
        print("Process still running after completion. Manually forcing exit...")
        # Try to manually identify specific issues
        subprocess.run(["pkill", "-9", "model-service"], check=False)
        
        # Look for resource tracker processes directly related to us
        ps_output = subprocess.check_output(["ps", "-ef"], text=True)
        parent_pid = os.getpid()
        for line in ps_output.splitlines():
            if f"{parent_pid}" in line and "resource_tracker" in line:
                print(f"Found resource tracker: {line}")
                try:
                    parts = line.split()
                    tracker_pid = int(parts[1])
                    print(f"Killing resource tracker PID: {tracker_pid}")
                    os.kill(tracker_pid, signal.SIGKILL)
                except Exception as e:
                    print(f"Error killing resource tracker: {e}")
        
        print("Forcing exit with os._exit(0)")
        os._exit(0)  # Force exit
        
    except KeyboardInterrupt:
        print("\nScript interrupted with KeyboardInterrupt")
        print_thread_info()
        sys.exit(1)
    except Exception as e:
        print(f"\nScript failed with error: {e}")
        print_thread_info()
        sys.exit(1)