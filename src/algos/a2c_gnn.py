from collections import namedtuple
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

from src.algos.layers import GNNCritic, GNNActor
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T, scale_factor, json_file=None, observe_od_prices=False):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        self.observe_od_prices = observe_od_prices
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs):
        acc, time, dacc, demand = obs

        # Current availability at t+1
        current_avb = torch.tensor([acc[n][self.env.time + 1] * self.s for n in self.env.region]).view(1, 1, self.env.nregion).float()

        # Future availability from t+2 to t+T
        future_avb = torch.tensor([[(acc[n][self.env.time + 1] + self.env.dacc[n][t])* self.s for n in self.env.region] for t in range(self.env.time + 1, self.env.time + self.T + 1)]).view(1, self.T, self.env.nregion).float()

        # Queue length
        queue_len = torch.tensor([len(self.env.queue[n]) * self.s for n in self.env.region]).view(1, 1, self.env.nregion).float()
        
        # Current demand at t
        current_demand = torch.tensor([sum([(demand[i, j][time])* self.s for j in self.env.region]) for i in self.env.region]).view(1, 1, self.env.nregion).float()

        if self.observe_od_prices:
            # Current OD prices at t - shape [nregion, nregion] for OD-specific prices
            current_price = torch.tensor([[(self.env.price[i, j][time])* self.s for j in self.env.region] for i in self.env.region]).view(1, self.env.nregion, self.env.nregion).float()
            
            x = torch.cat((current_avb, future_avb, queue_len, current_demand, current_price), dim=1).squeeze(0).view(1 + self.T + 1 + 1 + self.env.nregion, self.env.nregion).T
        else:
            # Current price at t
            current_price = torch.tensor([sum([(self.env.price[i, j][time])* self.s for j in self.env.region]) for i in self.env.region]).view(1, 1, self.env.nregion).float()                   

            x = torch.cat((current_avb, future_avb, queue_len, current_demand, current_price), dim=1).squeeze(0).view(1 + self.T + 1 + 1 + 1, self.env.nregion).T
              
        if self.json_file is not None:
            edge_index = torch.vstack(
                (
                    torch.tensor(
                        [edge["i"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                    torch.tensor(
                        [edge["j"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                )
            ).long()
        else:
            edge_index = torch.cat(
                (
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                ),
                dim=0,
            ).long()
        data = Data(x, edge_index)
        return data

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(
        self,
        env,
        input_size,
        device,
        hidden_size,
        T,
        mode,
        gamma,
        p_lr,   # actor learning rate
        q_lr,   # critic learning rate
        actor_clip,    # gradient clipping value for actor
        critic_clip,   # gradient clipping value for critic
        scale_factor,
        json_file = None,
        observe_od_prices = False,
        od_price_actions = False,  # whether to use OD-based price actions
        reward_scale = 0.00005,  # reward scaling factor
        eps=np.finfo(np.float32).eps.item(),
        job_id = None  # unique identifier for this model instance
    ):
        
        super(A2C, self).__init__()
        self.env = env
        self.act_dim = env.nregion
        self.mode = mode
        self.od_price_actions = od_price_actions
        self.job_id = job_id
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.actor = GNNActor(self.input_size, self.hidden_size, act_dim=self.act_dim, mode=mode, od_price_actions=od_price_actions)
        self.critic = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.obs_parser = GNNParser(self.env, T=T, json_file=json_file, scale_factor=scale_factor, observe_od_prices=observe_od_prices)

        self.p_lr = p_lr
        self.q_lr = q_lr
        self.optimizers = self.configure_optimizers()

        self.gamma = gamma
        self.reward_scale = reward_scale
        self.actor_clip = actor_clip
        self.critic_clip = critic_clip

        # Action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.concentration_history = []  # Track concentration parameters per step
        
        self.to(self.device)

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state
    
    # Combines select action and forward steps of actor and critic
    def select_action(self, obs, deterministic=False, return_concentration=False, log_step=False, step_num=None):
        state = self.parse_obs(obs).to(self.device)
        a, logprob, concentration = self.actor(state, deterministic=deterministic)
        value = self.critic(state)
        
        # Only save actions for training (when not deterministic)
        if not deterministic:
            if isinstance(concentration, dict):
                self.last_concentration = {k: v.detach() for k, v in concentration.items()}
                self.concentration_history.append({k: v.detach().cpu().numpy() for k, v in concentration.items()})
            else:
                self.last_concentration = concentration.detach()
                self.concentration_history.append(concentration.detach().cpu().numpy())
            self.last_action = a.detach()
            self.last_value = value.detach()
            self.last_log_prob = logprob.detach() if logprob is not None else None
            self.saved_actions.append(SavedAction(logprob, value))
    
        action_np = a.detach().cpu().numpy()
        
        # Handle different action shapes based on mode and od_price_actions:
        # Mode 0: [nregion] -> 1D array
        # Mode 1 origin: [nregion] -> 1D array
        # Mode 1 OD: [nregion, nregion] -> 2D array
        # Mode 2 origin: [nregion, 2] -> 2D array [[price, reb], ...]
        # Mode 2 OD: [nregion, nregion+1] -> 2D array [[OD prices..., reb], ...]
        if self.mode == 0:
            action_array = action_np
        elif self.mode == 1:
            if self.od_price_actions:
                action_array = action_np  # [nregion, nregion]
            else:
                action_array = action_np.squeeze(-1) if action_np.ndim > 1 and action_np.shape[-1] == 1 else action_np  # [nregion]
        else:
            action_array = action_np  # [nregion, 2] or [nregion, nregion+1]
        
        # Return the action and optionally concentration parameter and log probability
        if return_concentration:
            if isinstance(concentration, dict):
                concentration_value = {k: v.detach().cpu().numpy() for k, v in concentration.items()}
            else:
                concentration_value = concentration.detach().cpu().numpy()
            logprob_value = logprob.item() if logprob is not None else None
            return action_array, concentration_value, logprob_value
        else:
            return action_array
    
    def training_step(self, update_actor=True):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # Don't normalize rewards - use raw rewards
        scaled_rewards = self.rewards
        
        # calculate the true value using scaled rewards
        for r in scaled_rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device) / self.reward_scale
        
        # Calculate advantages
        advantages = []
        
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            advantages.append(advantage)
        
        # Calculate advantage statistics for logging
        advantages_tensor = torch.tensor(advantages)
        adv_mean = advantages_tensor.mean().item()
        adv_std = advantages_tensor.std().item()
        
        # Calculate losses using advantages
        for i, ((log_prob, value), R) in enumerate(zip(saved_actions, returns)):
            # Policy loss is just the advantage term
            policy_loss = -log_prob * advantages[i]
            policy_losses.append(policy_loss)
            
            # Value loss unchanged
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))
        
        # Actor loss is the policy loss
        a_loss = torch.stack(policy_losses).mean()
        
        # take gradient steps for actor only if update_actor=True
        if update_actor:
            self.optimizers['a_optimizer'].zero_grad()
            a_loss.backward()
            
            # Gradient clipping for actor
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip)
            self.optimizers['a_optimizer'].step()
        else:
            # During warmup, don't update actor
            actor_grad_norm = torch.tensor(0.0)
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).mean()
        v_loss.backward()
        
        # Gradient clipping for critic
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip)
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        del self.concentration_history[:]  # Clear concentration history

        # Return metrics for logging
        return {
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_loss": a_loss.item(),
            "critic_loss": v_loss.item(),
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
        }

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c_optimizer"] = torch.optim.Adam(critic_params, lr=self.q_lr)
        return optimizers

    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["model"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)

    def test_agent(self, test_episodes, env, cplexpath, directory):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            obs = env.reset()
            action_rl = [0]*env.nregion
            actions = []
            done = False
            while not done:

                if env.mode == 0:
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()
                    eps_reward += paxreward
            
                    action_rl = self.select_action(obs, deterministic=True)  # Choose an action
                    actions.append(action_rl)

                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}

                    # solve minimum rebalancing distance problem (Step 3 in paper)
                    rebAction = solveRebFlow(env, desiredAcc)

                    # Take rebalancing action in environment
                    _, rebreward, done, _, system_info, _, _ = env.reb_step(rebAction)
                    eps_reward += rebreward
                    
                elif env.mode == 1:
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                    eps_reward += paxreward
                    action_rl = self.select_action(obs, deterministic=True)  # Choose an action
                    env.matching_update()

                elif env.mode == 2:
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                    eps_reward += paxreward

                    action_rl = self.select_action(obs, deterministic=True)  # Choose an action)

                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    desiredAcc = {
                        env.region[i]: int(
                            action_rl[i][-1] * dictsum(env.acc, env.time + 1))
                        for i in range(len(env.region))
                    }

                    # solve minimum rebalancing distance problem (Step 3 in paper)
                    rebAction = solveRebFlow(env, desiredAcc)

                    # Take rebalancing action in environment
                    _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
                    eps_reward += rebreward 
                else:
                    raise ValueError("Only mode 0, 1, and 2 are allowed")  
                   
                eps_served_demand += info["served_demand"]
                eps_rebalancing_cost += info["rebalancing_cost"]
                
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
        )