#!/usr/bin/env python
"""
Simple test script with no debugging to show the fixed behavior.
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
        name="simple-test",
        project="tic-tac-toe-test",
        base_model="Qwen/Qwen2.5-3B-Instruct",
    )
    
    print("Registering model...")
    await model.register(backend)
    
    print("Creating minimal training data...")
    train_groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(
                rollout(model, 0, is_validation=False) for _ in range(5)  # Just 5 examples
            )
            for _ in range(1)
        ),
        pbar_desc="gather",
    )
    
    print("Training model...")
    await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-4))
    
    print("Training complete!")
    # Not calling backend.down() - should still exit cleanly

if __name__ == "__main__":
    print("Script started")
    asyncio.run(main())
    print("Script completed - should exit naturally")