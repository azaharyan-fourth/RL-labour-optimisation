from agents.DQN.replay_memory import Transition, ReplayMemory
from agents.DQN.segment_tree import MinSegmentTree, SumSegmentTree
import numpy as np
from typing import List
import random

class PrioritizedExperienceReplay(ReplayMemory):
    def __init__(self, capacity, alpha: float = 0.6):

        super().__init__(capacity)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def push(self, *args):
        super().push(*args)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.capacity

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size, beta: float = 0.4):
        """Sample a batch of experiences."""
        assert beta > 0
        
        indices = self._sample_proportional(batch_size)
        
        sample = [self.memory[i] for i in indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            batch = sample,
            weights=weights,
            indices=indices,
        )

    def _sample_proportional(self, batch_size) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
    
        