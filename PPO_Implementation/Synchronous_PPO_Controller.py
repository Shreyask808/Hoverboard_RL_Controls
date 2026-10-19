from dataclasses import dataclass
import torch
from torch.distributions import Normal
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Actor_Critic_Neural_Networks.Actor_Critic_NNs import Policy_Net, Value_Function
from Hoverboard_Environment.Hbd_Env import Hoverboard
from PPO_Implementation.PPO_Rollout_Function import Rollout, Advantage, Batch_Rollout

# =================================================================================================================================================================================================================
# Parallel Workers definition
Hbd_path = "/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_2.xml"
N_workers = 3
input_dim = 8

@dataclass
class Batch:
    obs: np.ndarray
    advantage: np.ndarray
    targets: np.ndarray
    log_probs: np.ndarray
    action: np.ndarray
    episode_length: np.ndarray

def worker_fn(worker_id,xml_path,Policy_Net,Value_Function,worker_input,worker_output,discount_factor):
    torch.set_num_threads(1)
    env = Hoverboard(xml_path)
    cpu = torch.device("cpu")
    while True:
        n_episodes = worker_input.get()
        if n_episodes is None:
            break
        worker_output.put(Batch_Rollout(env,Policy_Net,Value_Function,n_episodes,discount_factor,cpu))

# =================================================================================================================================================================================================================
# PPO Gradient Step

# =================================================================================================================================================================================================================
# Synchronous PPO Architecture Definition
epochs = 10
batchsize = 10
iterations = 100
discount_factor = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":
    mp.set_start_method("spawn")

    Actor = Policy_Net(input_dim,64,64,2).to(device)                                                            # Stays in GPU if available
    Critic = Value_Function(input_dim,64,64,64).to(device)                                                      # Stays in GPU if available

    Actor_optim  = optim.Adam(Actor.parameters(),lr=3e-4)
    Critic_optim = optim.Adam(Critic.parameters(),lr=1e-4)

    shared_actor = Policy_Net(input_dim, 64, 64,2)                                                              # Stays in the shared memory space in CPU
    shared_actor.share_memory() 
    shared_critic = Value_Function(input_dim, 64, 64, 64)                                                           # Stays in the shared memory space in CPU
    shared_critic.share_memory()

    worker_inputs = [mp.Queue() for _ in range(N_workers)]
    worker_outputs = mp.Queue()

    process = [mp.Process(target=worker_fn,args=(i,Hbd_path,shared_actor,shared_critic,worker_inputs[i],worker_outputs,discount_factor)) for i in range(N_workers)]
    for p in process:
        p.start()

    for itr in range(iterations):
        shared_critic.load_state_dict(Critic.state_dict())  
        shared_actor.load_state_dict(Actor.state_dict())

        for q in worker_inputs:
            q.put(batchsize)
        results = [worker_outputs.get() for _ in range(N_workers)]
        replay_buffer = Batch(*[np.concatenate(x) for x in zip(*results)])

    for q in worker_inputs:
        q.put(None)
    for p in process:
        p.join()