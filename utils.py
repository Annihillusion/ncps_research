# Copyright 2022 Mathias Lechner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np

from tqdm import tqdm

def eval(model, valloader, criterion):
    losses, accs = [], []
    model.eval()
    device = next(model.parameters()).device  # get device the model is located on
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)  # move data to same device as the model
            labels = labels.to(device)

            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
            labels = labels.view(-1, *labels.shape[2:])  # flatten
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(-1) == labels).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


def train_one_epoch(model, trainloader, criterion, optimizer):
    running_loss = 0.0
    total = len(trainloader)
    pbar = tqdm(total)
    model.train()
    device = next(model.parameters()).device  # get device the model is located on
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)  # move data to same device as the model
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, hx = model(inputs)
        labels = labels.view(-1, *labels.shape[2:])  # flatten
        outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        pbar.set_description(f"loss={running_loss / (i + 1):0.4g}")
        pbar.update(1)
    pbar.close()
    return running_loss / total


def run_closed_loop(model, env, num_episodes=None):
    model.eval()
    obs = env.reset()
    device = next(model.parameters()).device
    hx = None  # Hidden state of the RNN
    returns = []
    total_reward = 0
    with torch.no_grad():
        while True:
            # PyTorch require channel first images -> transpose data
            obs = np.transpose(obs, [2, 0, 1]).astype(np.float32) / 255
            # add batch and time dimension (with a single element in each)
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
            pred, hx = model(obs, hx)
            # remove time and batch dimension -> then argmax
            action = pred.squeeze(0).squeeze(0).argmax().item()
            obs, r, done, _ = env.step(action)
            total_reward += r
            if done:
                obs = env.reset()
                hx = None  # Reset hidden state of the RNN
                returns.append(total_reward)
                total_reward = 0
                if num_episodes is not None:
                    # Count down the number of episodes
                    num_episodes = num_episodes - 1
                    if num_episodes == 0:
                        return returns
