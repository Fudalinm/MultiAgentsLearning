import numpy as np
from collections import deque


class ReplayBuffer():
    def __init__(self, max_size=int(1e5)):
        self.buffer = deque([], max_size)
        self.episode_rewards = []

        # Wykorzystujemy tę klasę, żeby pamiętać, ile dostali"śmy nagród w danym
        # epizodzie. To infromacja tylko dla nas, żeby wiedzieć, czy agent
        # radzi sobie coraz lepiej.
        self.reward_sum = 0

    def __len__(self):
        """
        Zwraca liczbę elementów w naszej historii.
        """
        return len(self.buffer)

    def add(self, entry):
        """
        Dodajemy nową krotkę do historii.
        """
        self.buffer.append(entry)
        self.update_reward_sum(entry)

    def update_reward_sum(self, entry):
        _, _, reward, _, done = entry
        self.reward_sum += reward
        if done:
            self.episode_rewards += [self.reward_sum]
            self.reward_sum = 0

    def get_batch(self, batch_size=32):
        """
        Bierzemy batcha wydarzeń z przeszłości
        """
        indices = np.random.choice(
            len(self.buffer), size=batch_size, replace=False)

        history_states = [self.buffer[idx] for idx in indices]
        history_states = list(zip(*history_states))
        batch = [np.array(item) for item in history_states]
        return batch