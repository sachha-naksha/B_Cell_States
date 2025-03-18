import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
    
    def get_action(self, state, deterministic=False):
        # Convert state to tensor if not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Get action logits
        logits = self.forward(state)
        
        # Convert to probabilities
        probs = torch.sigmoid(logits)
        
        if deterministic:
            # Deterministic action (threshold at 0.5)
            action = (probs > 0.5).float().detach().numpy()
        else:
            # Stochastic action (sample from Bernoulli)
            dist = torch.distributions.Bernoulli(probs)
            action = dist.sample().detach().numpy()
        
        # Get log probabilities
        log_probs = dist.log_prob(torch.tensor(action).float()).sum()
        
        return action, log_probs.item()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        # Convert state to tensor if not already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        return self.network(state).squeeze()

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
    
    def get_action(self, state, deterministic=False):
        return self.policy.get_action(state, deterministic)
    
    def update(self, states, actions, log_probs, returns, advantages):
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Get current log probs
        logits = self.policy(states)
        probs = torch.sigmoid(logits)
        dist = torch.distributions.Bernoulli(probs)
        current_log_probs = dist.log_prob(actions).sum(1)
        
        # Compute ratio and clipped objective
        ratio = torch.exp(current_log_probs - old_log_probs)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        # Compute value loss
        value_pred = self.value(states)
        value_loss = F.mse_loss(value_pred, returns)
        
        # Compute entropy (for exploration)
        entropy = -(probs * torch.log(probs + 1e-10) + (1 - probs) * torch.log(1 - probs + 1e-10)).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])