import random
import asyncio
from dotenv import load_dotenv

import art
from rollout import rollout
from art.local.backend import LocalBackend


load_dotenv()

random.seed(42)

DESTROY_AFTER_RUN = False


import threading
import subprocess
import os
import time

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
            if "model-service" in line or "python" in line:
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
        
    print("------------------------\n")

async def main():
    print("Starting main function")
    # run from the root of the repo
    backend = LocalBackend(path="examples/tic_tac_toe/.art")
    print("LocalBackend created")

    model = art.TrainableModel(
        name="001-script",
        project="tic-tac-toe-local",
        base_model="Qwen/Qwen2.5-3B-Instruct",
    )
    print("TrainableModel created")
    await model.register(backend)
    print("Model registered with backend")

    for i in range(await model.get_step(), 100):
        print(f"\n--- Starting iteration {i} ---")
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, i, is_validation=False) for _ in range(200)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        print(f"Gathered trajectory groups for iteration {i}")
        await model.delete_checkpoints()
        print(f"Deleted checkpoints for iteration {i}")
        await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-4))
        print(f"Completed training for iteration {i}")

    print("\nTraining loop completed")

    # Debug info before cleanup
    print_thread_info()
    
    if DESTROY_AFTER_RUN:
        print("Calling backend.down()")
        await backend.down()
        print("backend.down() completed")
    
    # Debug info after cleanup
    print_thread_info()
    
    print("Main function completed")
    
    # Schedule additional debug info after a delay
    threading.Timer(5, print_thread_info).start()


# Background thread to periodically print debug info
def background_debug_monitor():
    print("Starting background debug monitor")
    try:
        count = 0
        while True:
            time.sleep(30)  # Print debug info every 30 seconds
            count += 1
            print(f"\n==== PERIODIC DEBUG INFO (#{count}) ====")
            print_thread_info()
    except Exception as e:
        print(f"Background debug monitor error: {e}")

if __name__ == "__main__":
    print("Script started")
    
    # Start background monitoring in a daemon thread
    monitor_thread = threading.Thread(
        target=background_debug_monitor, 
        daemon=True,
        name="BackgroundMonitor"
    )
    monitor_thread.start()
    
    try:
        asyncio.run(main())
        print("asyncio.run completed")
        
        # Additional debug info after main completes
        print_thread_info()
        
        print("Will print additional debug info in 5 seconds if script hasn't exited...")
        
        # Wait a bit to see if we hang
        for i in range(1, 4):
            time.sleep(5)
            print(f"Still running after {i*5} seconds, printing more debug info...")
            print_thread_info()
            
    except KeyboardInterrupt:
        print("\nScript interrupted with KeyboardInterrupt")
        print_thread_info()
    except Exception as e:
        print(f"\nScript failed with error: {e}")
        print_thread_info()
    
    print("End of script reached")
