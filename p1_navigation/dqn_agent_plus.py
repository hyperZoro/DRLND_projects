import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
ALPHA = 1               # prioritised experience replay scaling factor
EPSILON = 0.001         # shift to TD error in prioritised experience replay
BETA = 1                # the weight power parameter in prioirtised experience replay

#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=0, is_prioritised_replay=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.is_prioritised_replay = is_prioritised_replay

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, ALPHA, EPSILON, BETA)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        p = np.abs(TD_Error(self.qnetwork_local, state, action, reward, next_state, GAMMA)) + EPSILON
        w = Priority_Weight(p, BATCH_SIZE, BETA)
        self.memory.add(state, action, reward, next_state, done, p, w)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
                self.memory.update_priority(self.qnetwork_local)      

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, priorities, weights = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        for param, weight in zip(self.qnetwork_local.parameters(), weights):
            param._grad.data *= weight
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
                       

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha = 1, epsilon = 0.001, beta = 1):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): prioritised experience replay scaling factor - alpha = 0 means equal probability
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority", "weight"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
    
    def add(self, state, action, reward, next_state, done, priority=0, weight=0):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority, weight)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory, according to the priority"""
        experiences = random.choices(self.memory, k=self.batch_size, weights = self.get_sample_prob())

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack([e.weight for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, priorities, weights)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def get_sample_prob(self):
        prb = np.array( [m.priority**self.alpha for m in self.memory] )
        return prb / np.sum(prb)   
      
    def update_priority(self, Q):
        for m in self.memory:
            m = m._replace(priority = self.epsilon + np.abs(TD_Error(Q, m.state, m.action, m.reward, m.next_state, GAMMA)))
            m = m._replace(weight = Priority_Weight(m.priority, self.batch_size, self.beta))

def TD_Error(Q, state, action, reward, next_state, gamma):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
    return reward + gamma * Q(next_state).detach().max(1)[0].unsqueeze(1) - Q(state).detach().numpy()[0,action]

def Priority_Weight(p, N, beta):
    return (1 / p / N)**beta
