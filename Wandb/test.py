import wandb
import numpy as np

if __name__ == "__main__":
    wandb.init(project="hello", entity="carlylezqy")
    wandb.log({"loss": 0.5})
    examples = []
    for i in range(3):
        pixels = np.random.randint(low=0, high=256, size=(100, 100, 3))
        image = wandb.Image(pixels, caption=f"random field {i}")
        examples.append(image)

    wandb.log({"examples": examples})