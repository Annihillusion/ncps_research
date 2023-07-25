import gym

from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from utils.util import *
from utils.models import ConvLTC

NUM_NEURONS = 8

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    # env = gym.make("ALE/Breakout-v5")
    env = wrap_deepmind(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLTC(n_neurons=NUM_NEURONS, n_actions=env.action_space.n)
    model.load_state_dict(torch.load('saved_model/8neurons_ncp_50epochs.pkl'))
    model = model.to(device)
    run_closed_loop(model, env)
