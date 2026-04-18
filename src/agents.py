"""
Q-Learning ve SARSA Ajanları
"""

import numpy as np
from src.environment import PricingEnv, N_STATES, N_ACTIONS, STATE_IDX


class QLearningAgent:
    """Off-policy TD kontrolü."""

    name = "Q-Learning"

    def __init__(self, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.Q         = np.zeros((N_STATES, N_ACTIONS))
        self.epsilon   = eps_start

    # ── epsilon-greedy aksiyon seçimi ──────────────────────
    def select_action(self, state, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(N_ACTIONS))
        return int(np.argmax(self.Q[STATE_IDX[state]]))

    # ── Q-Learning güncelleme: max Q(s') ───────────────────
    def update(self, s, a, r, s_next):
        si  = STATE_IDX[s]
        nsi = STATE_IDX[s_next]
        td_target = r + self.gamma * np.max(self.Q[nsi])
        self.Q[si, a] += self.alpha * (td_target - self.Q[si, a])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    @property
    def policy(self):
        return {s: int(np.argmax(self.Q[STATE_IDX[s]])) for s in STATE_IDX}


class SARSAAgent(QLearningAgent):
    """On-policy TD kontrolü."""

    name = "SARSA"

    # ── SARSA güncelleme: gerçek Q(s', a') ─────────────────
    def update(self, s, a, r, s_next, a_next=None):
        si  = STATE_IDX[s]
        nsi = STATE_IDX[s_next]
        q_next = self.Q[nsi, a_next] if a_next is not None else np.max(self.Q[nsi])
        td_target = r + self.gamma * q_next
        self.Q[si, a] += self.alpha * (td_target - self.Q[si, a])
