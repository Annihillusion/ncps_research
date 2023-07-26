import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.models import ConvLTC
from utils.util import draw_networks

NUM_NEURONS = 12
MODEl_PATH = f'saved_model/{NUM_NEURONS}neurons_ncp_50epochs.pkl'

if __name__ == '__main__':
    model = ConvLTC(NUM_NEURONS)
    model.load_state_dict(torch.load(MODEl_PATH, map_location=torch.device('cpu')))
    wiring = model.rnn._wiring

    #adj = np.genfromtxt(f"wiring/{NUM_NEURONS}adj.csv", delimiter=',')
    adj = wiring.adjacency_matrix
    #wiring.adjacency_matrix = adj
    # adj_sen = wiring.sensory_adjacency_matrix
    weight = np.abs(model.rnn.rnn_cell.w.detach().numpy()*adj)
    weight = np.flipud(weight)
    weight = np.fliplr(weight)

    draw_networks(wiring)

    fig, ax = plt.subplots()
    pos = ax.imshow(weight)
    fig.colorbar(pos, ax=ax)
    plt.savefig(f'img/32to8/{wiring.units}_weights.png')
    plt.show()



