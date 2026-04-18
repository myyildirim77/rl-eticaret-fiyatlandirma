"""
Eğitim döngüsü — Q-Learning ve SARSA
"""

import numpy as np
from src.environment import PricingEnv, N_STATES, STATE_IDX
from src.agents import QLearningAgent, SARSAAgent

STEPS_PER_EPISODE = 20


def train(agent, episodes: int = 2000, seed: int = 42):
    env = PricingEnv(seed=seed)
    rng = np.random.default_rng(seed + 1)
    rewards_log   = []
    epsilon_log   = []

    for _ in range(episodes):
        state      = env.reset()
        ep_reward  = 0.0

        if isinstance(agent, SARSAAgent):
            action = agent.select_action(state, rng)

        for _ in range(STEPS_PER_EPISODE):
            if isinstance(agent, SARSAAgent):
                next_state, reward = env.step(action)
                next_action        = agent.select_action(next_state, rng)
                agent.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
            else:
                action             = agent.select_action(state, rng)
                next_state, reward = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state

            ep_reward += reward

        agent.decay_epsilon()
        rewards_log.append(ep_reward)
        epsilon_log.append(agent.epsilon)

    return np.array(rewards_log), np.array(epsilon_log)
