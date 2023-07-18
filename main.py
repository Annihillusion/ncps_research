import torch
import torch.nn as nn
import numpy as np
import gym
import argparse
import sys

from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from ncps.datasets.torch import AtariCloningDataset

from models import ConvLTC
from utils import *

BATCH_SIZE = 64
NUM_EPOCHS = 50

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_neurons', default=64, type=int)
    parser.add_argument('--connect_policy', default='ncp', type=str)
    args = parser.parse_args()

    # Names of log and saved_model
    PATH = f'log/{args.num_neurons}neurons_{args.connect_policy}_noRecur.txt'
    MODEL_NAME = f'saved_model/{args.num_neurons}neurons_{args.connect_policy}_noRecur.pkl'
    sys.stdout = open(PATH, 'w')

    env = gym.make("ALE/Breakout-v5")
    env = wrap_deepmind(env)

    root_dir = 'E:/breakout'
    train_ds = AtariCloningDataset("breakout", split="train", root_dir=root_dir)
    val_ds = AtariCloningDataset("breakout", split="val", root_dir=root_dir)
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLTC(n_neurons=args.num_neurons, n_actions=env.action_space.n, connect_policy=args.connect_policy).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # save_wiring(model)

    max_return = 0
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, trainloader, criterion, optimizer)

        # Evaluate
        val_loss, val_acc = eval(model, valloader, criterion)
        print(f"Epoch {epoch+1}, train_loss={train_loss:0.4g}, val_loss={val_loss:0.4g}, val_acc={100*val_acc:0.2f}%")

        # Apply model in real environment
        returns = run_closed_loop(model, env, num_episodes=10)
        print(f"Mean return {np.mean(returns)} (n={len(returns)})")
        if np.mean(returns) > max_return:
            max_return = np.mean(returns)
            torch.save(model.state_dict(), MODEL_NAME)