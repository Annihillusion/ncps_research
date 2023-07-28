import gym
import torch
import numpy as np
import argparse
from utils.models import ConvLTC
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind


def run_closed_loop(model, env, num_episodes=None, prefix='.'):
    model.eval()
    obs = env.reset()
    device = next(model.parameters()).device
    hx = None  # Hidden state of the RNN
    returns = []
    total_reward = 0
    act_dict = {0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT'}
    count = 0
    activation_collector = []
    action_collector = []

    with torch.no_grad():
        while True:
            # PyTorch require channel first images -> transpose data
            obs = np.transpose(obs, [2, 0, 1]).astype(np.float32) / 255
            # add batch and time dimension (with a single element in each)
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
            obs_cache = obs.squeeze(0)
            pred, hx = model(obs, hx)
            # remove time and batch dimension -> then argmax
            action = pred.squeeze(0).squeeze(0).argmax().item()
            obs, r, done, info = env.step(action)

            activation_collector.append(hx[0].numpy())
            action_collector.append(action)

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
                        # np.savetxt(f'{prefix}_activation.csv', np.array(activation_collector), delimiter=',')
                        # np.savetxt(f'16neuron_feature_{id}.csv', np.array(feature_collector), delimiter=',')
                        # np.savetxt(f'{prefix}_action.csv', np.array(action_collector), delimiter=',')
                        return np.array(activation_collector), np.array(action_collector), np.mean(returns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_neurons', default=8, type=int)
    parser.add_argument('--suffix', default='', type=str)
    args = parser.parse_args()

    NUM_NEURONS = args.num_neurons
    suffix = args.suffix
    MODEL_PATH = f'saved_model/{NUM_NEURONS}neurons_ncp{suffix}.pkl'
    RECORD_PREFIX = f'record/{NUM_NEURONS}neurons_ncp{suffix}'

    env = gym.make("ALE/Breakout-v5")
    env = wrap_deepmind(env)
    device = torch.device("cpu")
    model = ConvLTC(n_neurons=NUM_NEURONS, n_actions=env.action_space.n).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    activation, action, reward = run_closed_loop(model, env, num_episodes=5, prefix=RECORD_PREFIX)

    collecter = [[], [], [], []]
    for i, item in enumerate(activation):
        collecter[int(action[i])].append(item)
    matrix4 = np.mean(collecter[3], axis=-2)

    if collecter[2] == []:
        min = matrix4.min()
        matrix3 = np.ones(args.num_neurons) * min
    else:
        matrix3 = np.mean(collecter[2], axis=-2)
    mat = np.array([matrix3, matrix4])
    np.savetxt(f'record/{NUM_NEURONS}act{suffix}.csv', mat, delimiter=',')
    print(0)
