import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Beta
from torch_geometric.nn import GCNConv

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, in_channels, hidden_size, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = torch.sum(x, dim=1)
        x = self.lin3(x)
        x = x.squeeze(0)
        return x

class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size, act_dim, mode, od_price_actions=False):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.od_price_actions = od_price_actions
        self.conv1 = GCNConv(in_channels, in_channels)
        
        # Calculate output dimensions based on mode and OD pricing
        if mode == 0:
            # Mode 0: rebalancing only — single MLP
            self.lin1 = nn.Linear(in_channels, hidden_size)
            self.lin2 = nn.Linear(hidden_size, hidden_size)
            self.lin3 = nn.Linear(hidden_size, 1)
        elif mode == 1:
            # Mode 1: pricing only — single MLP
            self.lin1 = nn.Linear(in_channels, hidden_size)
            self.lin2 = nn.Linear(hidden_size, hidden_size)
            if od_price_actions:
                self.lin3 = nn.Linear(hidden_size, 2 * act_dim)  # 2 Beta params × N destinations
            else:
                self.lin3 = nn.Linear(hidden_size, 2)  # 2 Beta params
        else:
            # Mode 2: pricing + rebalancing — shared MLP
            self.lin1 = nn.Linear(in_channels, hidden_size)
            self.lin2 = nn.Linear(hidden_size, hidden_size)
            if od_price_actions:
                self.lin3 = nn.Linear(hidden_size, 2 * act_dim + 1)  # 2 Beta params × N destinations + 1 Dirichlet param
            else:
                self.lin3 = nn.Linear(hidden_size, 3)  # 2 Beta params + 1 Dirichlet param

    def forward(self, data, deterministic=False):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = x.reshape(-1, self.act_dim, self.in_channels)
        if self.mode == 0:
            # Single MLP
            x = F.leaky_relu(self.lin1(x))
            x = F.leaky_relu(self.lin2(x))
            x = F.softplus(self.lin3(x))
            assert x.shape == (1, self.act_dim, 1), f"Mode 0: Expected shape (1, {self.act_dim}, 1), got {x.shape}"
            concentration = x.squeeze(-1) + 0.1
            assert concentration.shape == (1, self.act_dim), f"Mode 0: Expected concentration shape (1, {self.act_dim}), got {concentration.shape}"
        elif self.mode == 1:
            # Single MLP
            x = F.leaky_relu(self.lin1(x))
            x = F.leaky_relu(self.lin2(x))
            x = F.softplus(self.lin3(x))
            if self.od_price_actions:
                assert x.shape == (1, self.act_dim, 2 * self.act_dim), f"Mode 1 OD: Expected shape (1, {self.act_dim}, {2*self.act_dim}), got {x.shape}"
                concentration = x.view(1, self.act_dim, self.act_dim, 2) + 1
            else:
                assert x.shape == (1, self.act_dim, 2), f"Mode 1: Expected shape (1, {self.act_dim}, 2), got {x.shape}"
                concentration = x + 1
        else:
            # Shared MLP for pricing and rebalancing
            x = F.leaky_relu(self.lin1(x))
            x = F.leaky_relu(self.lin2(x))
            x = F.softplus(self.lin3(x))
            if self.od_price_actions:
                assert x.shape == (1, self.act_dim, 2 * self.act_dim + 1), f"Mode 2 OD: Expected shape (1, {self.act_dim}, {2*self.act_dim+1}), got {x.shape}"
                beta_params = x[:, :, :2*self.act_dim].view(1, self.act_dim, self.act_dim, 2) + 1
                dirichlet_param = x[:, :, 2*self.act_dim:].unsqueeze(-1) + 0.1
                concentration = {'beta': beta_params, 'dirichlet': dirichlet_param}
            else:
                assert x.shape == (1, self.act_dim, 3), f"Mode 2: Expected shape (1, {self.act_dim}, 3), got {x.shape}"
                concentration = torch.cat([x[:,:,:2] + 1, x[:,:,2:] + 0.1], dim=-1)  # [1, nregion, 3]
    
        if deterministic:
            if self.mode == 0:
                action = concentration / concentration.sum()
                action = action.squeeze(0)
            elif self.mode == 1:
                if self.od_price_actions:
                    action_o = concentration[:,:,:,0] / (concentration[:,:,:,0] + concentration[:,:,:,1])
                    action_o[action_o<0] = 0
                    action = action_o.squeeze(0)
                else:
                    action_o = concentration[:,:,0] / (concentration[:,:,0] + concentration[:,:,1])
                    action_o[action_o<0] = 0
                    action = action_o.squeeze(0)
            else:
                if self.od_price_actions:
                    beta_conc = concentration['beta']
                    action_o = beta_conc[:,:,:,0] / (beta_conc[:,:,:,0] + beta_conc[:,:,:,1])
                    action_o[action_o<0] = 0
                    dirichlet_conc = concentration['dirichlet']
                    action_reb = dirichlet_conc.squeeze(-1) / dirichlet_conc.squeeze(-1).sum()
                    action = torch.cat((action_o.squeeze(0), action_reb.squeeze(0).unsqueeze(-1)), dim=-1)
                else:
                    action_o = concentration[:,:,0] / (concentration[:,:,0] + concentration[:,:,1])
                    action_o[action_o<0] = 0
                    action_reb = concentration[:,:,2] / concentration[:,:,2].sum()
                    action = torch.stack((action_o.squeeze(0), action_reb.squeeze(0)), dim=-1)
            log_prob = None
        else:
            if self.mode == 0:
                m = Dirichlet(concentration)
                action = m.rsample()
                log_prob = m.log_prob(action)
                action = action.squeeze(0)
                assert action.shape == (self.act_dim,), f"Mode 0: Expected action shape ({self.act_dim},), got {action.shape}"
                assert log_prob.shape == (1,), f"Mode 0: Expected log_prob shape (1,), got {log_prob.shape}"
            elif self.mode == 1:
                if self.od_price_actions:
                    m_o = Beta(concentration[:,:,:,0], concentration[:,:,:,1])
                    action_o = m_o.rsample()
                    log_prob = m_o.log_prob(action_o).sum(dim=[1, 2])
                    action = action_o.squeeze(0)
                    assert action.shape == (self.act_dim, self.act_dim), f"Mode 1 OD: Expected action shape ({self.act_dim}, {self.act_dim}), got {action.shape}"
                    assert log_prob.shape == (1,), f"Mode 1 OD: Expected log_prob shape (1,), got {log_prob.shape}"
                else:
                    m_o = Beta(concentration[:,:,0], concentration[:,:,1])
                    action_o = m_o.rsample()
                    log_prob = m_o.log_prob(action_o).sum(dim=-1)
                    action = action_o.squeeze(0)
                    assert action.shape == (self.act_dim,), f"Mode 1: Expected action shape ({self.act_dim},), got {action.shape}"
                    assert log_prob.shape == (1,), f"Mode 1: Expected log_prob shape (1,), got {log_prob.shape}"
            else:
                if self.od_price_actions:
                    beta_conc = concentration['beta']
                    dirichlet_conc = concentration['dirichlet']
                    m_o = Beta(beta_conc[:,:,:,0], beta_conc[:,:,:,1])
                    action_o = m_o.rsample()
                    m_reb = Dirichlet(dirichlet_conc.squeeze(-1))
                    action_reb = m_reb.rsample()
                    log_prob = m_o.log_prob(action_o).sum(dim=[1, 2]) + m_reb.log_prob(action_reb)
                    action = torch.cat((action_o.squeeze(0), action_reb.squeeze(0).unsqueeze(-1)), dim=-1)
                    assert action.shape == (self.act_dim, self.act_dim + 1), f"Mode 2 OD: Expected action shape ({self.act_dim}, {self.act_dim + 1}), got {action.shape}"
                    assert log_prob.shape == (1,), f"Mode 2 OD: Expected log_prob shape (1,), got {log_prob.shape}"
                else:
                    m_o = Beta(concentration[:,:,0], concentration[:,:,1])
                    action_o = m_o.rsample()
                    m_reb = Dirichlet(concentration[:,:,2])
                    action_reb = m_reb.rsample()
                    log_prob = m_o.log_prob(action_o).sum(dim=-1) + m_reb.log_prob(action_reb)
                    action = torch.stack((action_o.squeeze(0), action_reb.squeeze(0)), dim=-1)
                    assert action.shape == (self.act_dim, 2), f"Mode 2: Expected action shape ({self.act_dim}, 2), got {action.shape}"
                    assert log_prob.shape == (1,), f"Mode 2: Expected log_prob shape (1,), got {log_prob.shape}"
            
        return action, log_prob, concentration