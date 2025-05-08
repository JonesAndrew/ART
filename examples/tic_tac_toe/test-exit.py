import random
import asyncio
from dotenv import load_dotenv

import art
from rollout import rollout
from art.local.backend import LocalBackend


load_dotenv()

random.seed(42)

# No cleanup called explicitly
DESTROY_AFTER_RUN = False


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

    # Don't call down() - this is the scenario we want to test
    print("Main function completed without explicit cleanup")


if __name__ == "__main__":
    print("Script starting")
    asyncio.run(main())
    print("asyncio.run completed - script should exit naturally")