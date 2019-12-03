import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # actor learning rate
LR_CRITIC = 1e-3        # critic learning rate
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 10        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OU_Noise:
    """
    Ornstein-Uhlenbeck process
    """
    def __init__(self, size, seed, mu=0, sigma=0.2, dt=1, theta=0.15):
        """
        Initialize parameters and noise process
        :param size: length of random vector
        :param seed: random seed
        :param mu: mean value
        :param sigma: standard deviation
        :param dt: length of a timestep
        :param theta: inverse of a time decay constant
        """
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.dt = dt
        self.theta = theta
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        reset internal state to mean
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        update internal state and return it as a sample
        """
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.state += dx
        return self.state

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
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

        # Actor Network with target net
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network with target net
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Ornstein-Uhlenbeck noise
        self.noise = OU_Noise(action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def reset(self):
        self.noise.reset()
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """
        Returns actions for given state as per current policy.
        :param state: current state
        :param add_noise: whether to add Ornstein-Uhlenbeck noise
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # add OU_noise to action to explore
        if add_noise:
            action_values += self.noise.sample()

        return np.clip(action_values, -1, 1)

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.

        :param experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ------------------- update critic ------------------- #
        # get predicted next state, actions and Q values from target network
        actions_next = self.actor_target(next_states)
        Qtargets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states 
        Qtargets = rewards + (gamma * Qtargets_next * (1 - dones))

        # Get expected Q values from local model
        Qexpected = self.critic_local(states, actions)
        # calculate the batch loss
        critic_loss = F.mse_loss(Qexpected, Qtargets)

        # minimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()        # backward pass
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  #gradient clipping
        self.critic_optimizer.step()   # perform a single optimization step (parameter update)

        # ------------------- update actor ------------------- #
        # compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # minimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()  # backward pass
        self.actor_optimizer.step()  # perform a single optimization step (parameter update)

        # ------------------- update target network ------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model (PyTorch model): weights will be copied from
        :param target_model (PyTorch model): weights will be copied to
        :param tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.

        :param action_size (int): dimension of each action
        :param buffer_size (int): maximum size of buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)