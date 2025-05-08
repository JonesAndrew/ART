#!/usr/bin/env python
"""
Final test script demonstrating automatic cleanup when the backend goes out of scope.
"""

import asyncio
import art
from art.local.backend import LocalBackend
from rollout import rollout

async def main():
    print("Creating LocalBackend...")
    backend = LocalBackend()
    
    print("Creating TrainableModel...")
    model = art.TrainableModel(
        name="auto-cleanup-test",
        project="tic-tac-toe-test",
        base_model="Qwen/Qwen2.5-3B-Instruct",
    )
    
    print("Registering model...")
    await model.register(backend)
    
    # Do multiple training iterations
    for i in range(2):
        print(f"\n--- Training iteration {i+1}/2 ---")
        
        print("Creating training data...")
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, i, is_validation=False) for _ in range(5)  # Just 5 examples
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        
        print(f"Training model (iteration {i+1})...")
        await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-4))
        print(f"Completed training iteration {i+1}")
    
    print("\nAll training completed successfully.")
    print("Exiting function - backend will go out of scope and trigger cleanup automatically")
    # We don't call backend.down() - it should clean up automatically when it goes out of scope

if __name__ == "__main__":
    print("Script started")
    asyncio.run(main())
    print("Script completed - should exit naturally without hanging")