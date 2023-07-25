import torch
import itertools
import matplotlib.pyplot as plt
from utils.models import ConvLTC
from utils.util import draw_networks

NUM_NEURONS = 12
MODEl_PATH = 'saved_model/12neurons_ncp_50epochs.pkl'

if __name__ == '__main__':
    model = ConvLTC(NUM_NEURONS)
    model.load_state_dict(torch.load(MODEl_PATH, map_location=torch.device('cpu')))
    wiring = model.rnn._wiring

    adj = wiring.adjacency_matrix
    adj_sen = wiring.sensory_adjacency_matrix

    draw_networks(wiring)
    # wiring.draw_graph(layout='kamada')
    # fig, ax = plt.subplots()
    # ax.imshow(adj)
    # plt.show()
    print(0)

