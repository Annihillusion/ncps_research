import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.models import ConvLTC
from utils.util import draw_networks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_neurons', default=8, type=int)
    parser.add_argument('--suffix', default='', type=str)
    args = parser.parse_args()

    NUM_NEURONS = args.num_neurons
    MODEl_PATH = f'saved_model/{NUM_NEURONS}neurons_ncp{args.suffix}.pkl'
    ACT_PATH = f'record/{NUM_NEURONS}act{args.suffix}.csv'
    # FIG_PATH = f'img/network{args.suffix}/{NUM_NEURONS}_network{args.suffix}'
    FIG_PATH = '.'

    model = ConvLTC(NUM_NEURONS)
    model.load_state_dict(torch.load(MODEl_PATH, map_location=torch.device('cpu')))
    wiring = model.rnn._wiring

    #adj = np.genfromtxt(f"wiring/{NUM_NEURONS}adj.csv", delimiter=',')
    adj = wiring.adjacency_matrix
    #wiring.adjacency_matrix = adj
    # adj_sen = wiring.sensory_adjacency_matrix
    weight = np.abs(model.rnn.rnn_cell.w.detach().numpy()*adj)
    # weight = np.flipud(weight)
    # weight = np.fliplr(weight)

    activation = np.genfromtxt(ACT_PATH, delimiter=',')
    draw_networks(wiring, weight, activation, 0, FIG_PATH)
    plt.clf()
    draw_networks(wiring, weight, activation, 1, FIG_PATH)

    # fig, ax = plt.subplots()
    # pos = ax.imshow(weight)
    # fig.colorbar(pos, ax=ax)
    # plt.savefig()
    #plt.show()



