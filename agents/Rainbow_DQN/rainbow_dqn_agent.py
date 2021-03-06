import torch
from torch.functional import Tensor
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import pandas as pd
import numpy as np
from agents.Rainbow_DQN.per import PrioritizedReplayBuffer
from agents.Rainbow_DQN.rainbow_network import RainbowNetwork
from agents.Rainbow_DQN.replay_memory import ReplayBuffer
from environment import TSEnvironment
from typing import Deque, Dict, List, Tuple

class RainbowDQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        env: TSEnvironment,
        memory_size: int = 40000,
        batch_size: int = 32,
        target_update: int = 250,
        gamma: float = 1,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 600.0,
        atom_size: int = 301,
        # N-step Learning
        n_step: int = 1,
    ):

        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        action_dim = env.action_space.n
        obs_dim = env.n_observation_space*env.window
        

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma

        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            memory_size,
            obs_space=obs_dim,
            output=action_dim, batch_size=batch_size, alpha=alpha
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                memory_size,
                obs_space=obs_dim,
                output=action_dim,
                batch_size=batch_size, n_step=n_step, gamma=gamma
            )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = RainbowNetwork(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = RainbowNetwork(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: Tensor) -> np.ndarray:
        """Select an action from the input state."""

        if isinstance(state, pd.DataFrame):
            state = state.to_numpy()
            state = torch.from_numpy(state)
        # NoisyNet: no epsilon greedy action selection
        state = state.unsqueeze(0)
        self.dqn.eval()
        selected_action = self.dqn(state).argmax()
        selected_action = selected_action
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
       #self.dqn.train()
        return selected_action

    def step(self, action: np.ndarray, num_episode: int) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done = self.env.step(action, num_episode)

        if next_state is not None and isinstance(next_state, pd.DataFrame):
            next_state = self.env.transform_data_for_nn(next_state)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        self.dqn.train()

        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train(self, num_episodes: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False
        
        self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        for episode in range(num_episodes):
            self.env.reset()
            for index, values in enumerate(self.env.iter_dataset(train=True)):
                state = self.env.get_state(index + self.env.window)
                if state is not None and isinstance(state, pd.DataFrame):
                    state = self.env.transform_data_for_nn(state)

                action = self.select_action(state)
                next_state, reward, done = self.step(action, episode)

                state = next_state
                score += reward
            
                # NoisyNet: removed decrease of epsilon
            
                # PER: increase beta
                fraction = min(index / len(self.env.dataset.dataset_train), 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # if episode ends
                if done:
                    scores.append(score)
                    score = 0

                # if training is ready
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    losses.append(loss)
                    update_cnt += 1
                
                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

            #self.env.reset_cron_iter()

            if episode % 5 == 0 and episode > 0:
                self.test()

            with open('rewards.txt', 'a') as fout:
                fout.write(f"Episode rewards: {scores[-1]}\n")
            with open('losses.txt', 'a') as fout:
                fout.write(f"{sum(losses)}\n")
            

    def test(self):
        """Test the agent."""
        self.dqn.eval()

        self.env.reset(index=len(self.env.dataset.dataset_train))
        done = False
        rewards, actions = [], []
        
        for index, values in enumerate(self.env.iter_dataset(train=False), 
                                        start=len(self.env.dataset.dataset_train)):
                state = self.env.get_state(index, is_test=True)
                state_transformed = self.env.transform_data_for_nn(state, mode='val')
                action = self.select_action(state_transformed)
                next_state, reward, done = self.env.step(action, is_test=True)

                rewards.append(reward)    
                actions.append(action)
                with open('test.txt', 'a') as fout:
                    fout.write(f"{values[1]['date']}: {action} {reward}\n")
        
        with open('rewards_test.txt', 'a') as fout:
            fout.write(f"Episode rewards: {sum(rewards)}\n")
        with open('actions_test.txt', 'a') as fout:
            fout.write(f"Episode actions: {actions}\n")

        self.dqn.train()


    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)


        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())