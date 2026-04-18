"""
E-Ticaret Dinamik Fiyatlandırma — RL Ortamı
============================================
State  : (Rakip Fiyatı [0-2], Stok Seviyesi [0-1])
Action : 0=Fiyat Düşür  1=Sabit Tut  2=Fiyat Artır
Reward : İşlem kârı (₺)
"""

import numpy as np

RAKIP_LABELS  = ["Rakip Ucuz", "Rakip Aynı", "Rakip Pahalı"]
STOK_LABELS   = ["Stok Az",    "Stok Çok"]
ACTION_LABELS = ["Fiyat Düşür", "Sabit Tut", "Fiyat Artır"]

STATES    = [(r, s) for r in range(3) for s in range(2)]
N_STATES  = len(STATES)   # 6
N_ACTIONS = 3
STATE_IDX = {s: i for i, s in enumerate(STATES)}


class PricingEnv:
    """Basit stokastik e-ticaret fiyatlandırma ortamı."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = self._random_state()
        return self.state

    def step(self, action: int):
        reward    = self._reward(self.state, action)
        self.state = self._random_state()
        return self.state, reward

    # ── iç yardımcılar ──────────────────────────────────────
    def _random_state(self):
        return (int(self.rng.integers(0, 3)),
                int(self.rng.integers(0, 2)))

    def _reward(self, state, action) -> float:
        rakip, stok = state
        noise = self.rng.normal(0, 3)

        base = {
            0: [15,  5, -5],
            1: [-3, 10,  8],
            2: [-15, 3, 20],
        }[rakip][action]

        stok_penalty = {0: [-8, 2, -5], 1: [0, 0, 0]}[stok][action]
        return float(base + stok_penalty + noise)
