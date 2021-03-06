import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_model import Actor, Critic
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4        # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

LEARN_FREQ = 1
LEARN_NUM = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device used: ", device)

class DDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(2*state_size, 2*action_size, random_seed).to(device)
        self.critic_target = Critic(2*state_size, 2*action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def soft_update(self, tau):
        self._soft_update(self.critic_local, self.critic_target, tau)
        self._soft_update(self.actor_local, self.actor_target, tau)        


    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ??_target = ??*??_local + (1 - ??)*??_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class MADDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.maddpg_agent = [DDPGAgent(state_size,action_size,random_seed), 
                             DDPGAgent(state_size,action_size,random_seed)]
        self.num_agent = len(self.maddpg_agent)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done, t):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and t % LEARN_FREQ == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        action = []
        for i in range(self.num_agent):
            action.append(self.maddpg_agent[i].act(state[i]))
        return np.array(action)


    def reset(self):
        for ag in self.maddpg_agent:
            ag.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + ?? * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        states_both = torch.hstack([states[0], states[1]])
        actions_both = torch.hstack([actions[0], actions[1]])
        next_states_both = torch.hstack([next_states[0], next_states[1]])


        for i in range(self.num_agent):
            target_next_actions_both = []
            local_actions_pred_both = []
            #with torch.no_grad():
            for j in range(self.num_agent):
                ag = self.maddpg_agent[j]
                target_next_actions_both.append(ag.actor_target(next_states[j]))
                local_actions_pred_both.append(ag.actor_local(states[j]))
            target_next_actions_both = torch.hstack(target_next_actions_both)
            local_actions_pred_both = torch.hstack(local_actions_pred_both)

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            ag = self.maddpg_agent[i]
            Q_targets_next = ag.critic_target(next_states_both, target_next_actions_both)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards[i] + (gamma * Q_targets_next * (1 - dones[i]))
            # Compute critic loss
            Q_expected = ag.critic_local(states_both, actions_both)
            critic_loss = F.huber_loss(Q_expected, Q_targets)
            # Minimize the loss
            ag.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(ag.critic_local.parameters(), 1)
            ag.critic_optimizer.step()
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actor_loss = -ag.critic_local(states_both, local_actions_pred_both).mean()
            # Minimize the loss
            ag.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(ag.actor_local.parameters(), 1)
            ag.actor_optimizer.step()
            # ----------------------- update target networks ----------------------- #
            ag.soft_update(TAU)
             



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.2, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.size = size                
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.rollaxis(np.dstack([e.state for e in experiences if e is not None]),2,1)).float().to(device)
        actions = torch.from_numpy(np.rollaxis(np.dstack([e.action for e in experiences if e is not None]),2,1)).float().to(device)
        rewards = torch.from_numpy(np.moveaxis(np.dstack([e.reward for e in experiences if e is not None]),0,-1)).float().to(device)
        next_states = torch.from_numpy(np.rollaxis(np.dstack([e.next_state for e in experiences if e is not None]),2,1)).float().to(device)
        dones = torch.from_numpy(np.moveaxis(np.dstack([e.done for e in experiences if e is not None]),0,-1).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)