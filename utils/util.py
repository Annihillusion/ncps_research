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
import networkx as nx
import matplotlib.pyplot as plt
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


def get_graph(self, include_sensory_neurons=True):
        if not self.is_built():
            raise ValueError(
                "Wiring is not built yet.\n"
                "This is probably because the input shape is not known yet.\n"
                "Consider calling the model.build(...) method using the shape of the inputs."
            )
        # Only import networkx package if we really need it
        import networkx as nx

        DG = nx.DiGraph()
        for i in range(self.units):
            neuron_type = self.get_type_of_neuron(i)
            DG.add_node("neuron_{:d}".format(i), neuron_type=neuron_type)
        if include_sensory_neurons == True:
            for i in range(self.input_dim):
                DG.add_node("sensory_{:d}".format(i), neuron_type="sensory")

        erev = self.adjacency_matrix
        sensory_erev = self.sensory_adjacency_matrix

        if include_sensory_neurons == True:
            for src in range(self.input_dim):
                for dest in range(self.units):
                    if self.sensory_adjacency_matrix[src, dest] != 0:
                        polarity = (
                            "excitatory" if sensory_erev[src, dest] >= 0.0 else "inhibitory"
                        )
                        DG.add_edge(
                            "sensory_{:d}".format(src),
                            "neuron_{:d}".format(dest),
                            polarity=polarity,
                        )

        for src in range(self.units):
            for dest in range(self.units):
                if self.adjacency_matrix[src, dest] != 0:
                    polarity = "excitatory" if erev[src, dest] >= 0.0 else "inhibitory"
                    DG.add_edge(
                        "neuron_{:d}".format(src),
                        "neuron_{:d}".format(dest),
                        polarity=polarity,
                    )
        return DG

def draw_networks(wiring):
    layer_size = [len(wiring.get_neurons_of_layer(i)) for i in range(wiring.num_layers)]
    id = iter(np.arange(wiring.units)[::-1])
    layer_color = ['blue', 'grey', 'orange']
    synapse_colors = {"excitatory": "green", "inhibitory": "red"}
    layers = [[f'neuron_{next(id)}' for _ in range(size)] for size in layer_size]
    G = nx.Graph()
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    color = [layer_color[data["layer"]] for v, data in G.nodes(data=True)]
    pos = nx.multipartite_layout(G, subset_key="layer")
    # plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_color=color, with_labels=False)

    G = get_graph(wiring, include_sensory_neurons=False)
    for node1, node2, data in G.edges(data=True):
            polarity = data["polarity"]
            edge_color = synapse_colors[polarity]
            nx.draw_networkx_edges(G, pos, [(node1, node2)], edge_color=edge_color)

    plt.axis("equal")
    plt.show()