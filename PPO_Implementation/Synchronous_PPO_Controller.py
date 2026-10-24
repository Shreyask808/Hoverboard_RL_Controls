from dataclasses import dataclass
import torch
import time
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Actor_Critic_Neural_Networks.Actor_Critic_NNs import Policy_Net, Value_Function
from Hoverboard_Environment.Hbd_Env import Hoverboard
from PPO_Implementation.PPO_Rollout_Function import Batch_Rollout

# =================================================================================================================================================================================================================
# Parallel Workers definition
Hbd_path = "/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/Hoverboard_MJCF_Files/hoverboard_4.xml"
N_workers = 5
input_dim = 12

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
## PPO Dataset
class PPO_Dataset(Dataset):
    def __init__(self,replay_buffer,device):
        self.obs = torch.tensor(replay_buffer.obs, dtype=torch.float32, device=device)
        self.advantage = torch.tensor(replay_buffer.advantage, dtype=torch.float32, device=device)
        self.targets = torch.tensor(replay_buffer.targets, dtype=torch.float32, device=device)
        self.log_prob = torch.tensor(replay_buffer.log_probs, dtype=torch.float32, device=device)
        self.action = torch.tensor(replay_buffer.action, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.advantage)

    def __getitem__(self, index):
        return self.obs[index], self.advantage[index], self.targets[index], self.log_prob[index], self.action[index]
    
def PPO_Update(Actor, Critic, train_loader, Actor_optim, Critic_optim,epochs, clip, device, target_kl=0.01):
    ds = train_loader.dataset
    for epoch in range(epochs):
        for obs_b, adv_b, tgt_b, old_lp_b, act_b in train_loader:
            mean, std = Actor(obs_b)
            new_lp = Normal(mean, std).log_prob(act_b).sum(dim=1)
            ratio = torch.exp(new_lp - old_lp_b)
            L = torch.min(ratio * adv_b, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_b)

            Actor_optim.zero_grad()
            (-L.mean()).backward()
            torch.nn.utils.clip_grad_norm_(Actor.parameters(), 0.5)
            Actor_optim.step()

            Critic_optim.zero_grad()
            F.mse_loss(Critic(obs_b), tgt_b).backward()
            torch.nn.utils.clip_grad_norm_(Critic.parameters(), 0.5)
            Critic_optim.step()

        with torch.no_grad():
            mean, std = Actor(ds.obs)
            log_ratio = Normal(mean, std).log_prob(ds.action).sum(dim=1) - ds.log_prob
            kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()

        if kl > 1.5 * target_kl:
            break

    return kl, epoch + 1
        
# =================================================================================================================================================================================================================
# Synchronous PPO Architecture Definition
epochs = 25
batchsize = 10
iterations = 1200
discount_factor = 0.999
clip = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t0 = time.time()

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
        adv = replay_buffer.advantage
        replay_buffer.advantage = (adv - adv.mean()) / (adv.std() + 1e-8)
        buffer_dataset = PPO_Dataset(replay_buffer,device)
        target_steps = 170                                   # steps per iteration that worked early
        n_mb_per_epoch = max(1, target_steps // epochs)      # about 17
        mb_size = max(256, len(adv) // n_mb_per_epoch)
        train_loader = DataLoader(buffer_dataset, batch_size = mb_size, shuffle=True)
        kl, epoch = PPO_Update(Actor,Critic,train_loader,Actor_optim,Critic_optim,epochs,clip,device)

        print(f"{itr+1}. Average Episode Length for all workers - {replay_buffer.episode_length.mean()}, KL = {kl}, Epoch = {epoch}")

    for q in worker_inputs:
        q.put(None)
    for p in process:
        p.join()

    tf = time.time()
    torch.save(Actor.state_dict(),"/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/PPO_Implementation/PPO_attempt_5_Actor_10W.pth")
    torch.save(Critic.state_dict(),"/mnt/c/Users/admin/Documents/Github/Hoverboard_RL_Controls/PPO_Implementation/PPO_attempt_5_Critic_10W.pth")

    wall_clock_time_min = np.floor((tf -t0)/60)
    wall_clock_time_sec = (tf - t0)%60
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("")
    print("Weights saved successfully")
    print("")
    print(f"Wall Clock Time - {wall_clock_time_min} [min] {wall_clock_time_sec} [sec]")
    print("")
    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")