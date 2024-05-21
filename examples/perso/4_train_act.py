"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch
import wandb

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy

# Create a directory to store the training checkpoint.
output_directory = Path("outputs/train/act_aloha_sim_insertion_human")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 20000
device = torch.device("cuda")
log_freq = 250

# Set up the dataset.
delta_timestamps = {
    # Load the previous image and state at -0.1 seconds before current frame,
    # then load current image and state corresponding to 0.0 second.
    # "observation.images.top": [-0.1, 0.0],
    # "observation.images.top": [0.0],
    # "observation.state": [-0.1, 0.0],
    # "observation.state": [0.0],
    # Load the previous action (-0.1), the next action to be executed (0.0),
    # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    # used to supervise the policy.
    # "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    "action": [i / 50 for i in range(100)],

}
dataset = LeRobotDataset("lerobot/aloha_sim_insertion_human", delta_timestamps=delta_timestamps)

print(f"Dataset has {len(dataset)} samples.")

# Set up the the policy.
# Policies are initialized with a configuration class, in this case `DiffusionConfig`.
# For this example, no arguments need to be passed because the defaults are set up for PushT.
# If you're doing something different, you will likely need to change at least some of the defaults.
cfg = ACTConfig()
policy = ACTPolicy(cfg, dataset_stats=dataset.stats)
policy.train()
policy.to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4000, gamma=0.5)

# Create dataloader for offline training.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
)

print("Begin training.")

# Set up wandb logging.
run = wandb.init(project="lerobot",
                 config=cfg.__dict__,
                 job_type="train",
                 name="act_aloha_sim_insertion_human") 

# Run training loop.
step = 0
done = False
while not done:
    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        # print(batch.keys())
        # print("observation.state", batch["observation.state"].shape)
        # print(batch["observation.state"][0])
        # print("observation.images.top", batch["observation.images.top"].shape)
        # print("action", batch["action"].shape)
        batch["observation.state"] = batch["observation.state"].squeeze(1)
        # print("observation.state", batch["observation.state"].shape)
        
        output_dict = policy.forward(batch)
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"loss": loss.item()})

        scheduler.step()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

# Save a policy checkpoint.
policy.save_pretrained(output_directory)

print("Training complete.")
run.finish()