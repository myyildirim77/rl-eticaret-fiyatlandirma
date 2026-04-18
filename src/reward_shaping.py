"""
Reward Shaping Analizi
======================
Neden bu reward fonksiyonu?
Farklı reward tasarımlarının öğrenmeye etkisini karşılaştırır.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.environment import (STATES, STATE_IDX, N_STATES, N_ACTIONS,
                               RAKIP_LABELS, STOK_LABELS, ACTION_LABELS)
from src.agents import QLearningAgent
from src.train  import train
from src.visualize import (_apply_theme, smooth,
                            BG, SURFACE, BORDER, FG, MUTED,
                            C_QL, C_SA, C_ACC, C_POS, C_NEG)


# ── Alternatif reward fonksiyonları ──────────────────────────────────────────
class PricingEnvV1:
    """Basit reward: sadece aksiyon-rakip uyumuna bak, stok yok."""
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = (int(self.rng.integers(3)), int(self.rng.integers(2)))
        return self.state

    def step(self, action):
        rakip, _ = self.state
        reward = {0: [10,  0, -10],
                  1: [ 0, 10,   0],
                  2: [-10, 0,  10]}[rakip][action]
        reward += self.rng.normal(0, 2)
        self.state = (int(self.rng.integers(3)), int(self.rng.integers(2)))
        return self.state, float(reward)


class PricingEnvV2:
    """Aşırı gürültülü reward: öğrenmeyi zorlaştırır."""
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = (int(self.rng.integers(3)), int(self.rng.integers(2)))
        return self.state

    def step(self, action):
        rakip, stok = self.state
        base = {0: [15, 5, -5], 1: [-3, 10, 8], 2: [-15, 3, 20]}[rakip][action]
        reward = base + self.rng.normal(0, 20)   # çok yüksek gürültü
        self.state = (int(self.rng.integers(3)), int(self.rng.integers(2)))
        return self.state, float(reward)


class PricingEnvV3:
    """Bizim tasarımımız (stok cezası + dengeli gürültü)."""
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = (int(self.rng.integers(3)), int(self.rng.integers(2)))
        return self.state

    def step(self, action):
        rakip, stok = self.state
        base   = {0: [15, 5, -5], 1: [-3, 10, 8], 2: [-15, 3, 20]}[rakip][action]
        stok_p = {0: [-8, 2, -5], 1: [0, 0, 0]}[stok][action]
        reward = base + stok_p + self.rng.normal(0, 3)
        self.state = (int(self.rng.integers(3)), int(self.rng.integers(2)))
        return self.state, float(reward)


def _train_custom_env(EnvClass, episodes=1000, seed=42):
    """Özel env ile Q-Learning eğitimi."""
    agent = QLearningAgent()
    env   = EnvClass(seed=seed)
    rng   = np.random.default_rng(seed + 1)
    rewards = []

    for _ in range(episodes):
        state = env.reset()
        ep_r  = 0.0
        for _ in range(20):
            si     = STATE_IDX[state]
            action = (int(rng.integers(N_ACTIONS))
                      if rng.random() < agent.epsilon
                      else int(np.argmax(agent.Q[si])))
            ns, r  = env.step(action)
            nsi    = STATE_IDX[ns]
            agent.Q[si, action] += agent.alpha * (
                r + agent.gamma * np.max(agent.Q[nsi]) - agent.Q[si, action]
            )
            state = ns
            ep_r += r
        agent.decay_epsilon()
        rewards.append(ep_r)

    return agent, np.array(rewards)


def run_reward_shaping_analysis(save_dir="assets"):
    _apply_theme()
    Path(save_dir).mkdir(exist_ok=True)

    SEEDS    = [42, 7, 13]
    EPISODES = 1000

    print("  Reward versiyonları karşılaştırılıyor ...")
    envs = [
        ("V1 — Basit (stok yok)",      PricingEnvV1, C_NEG),
        ("V2 — Gürültülü (σ=20)",      PricingEnvV2, C_ACC),
        ("V3 — Bizim Tasarım (σ=3)",   PricingEnvV3, C_QL),
    ]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    fig.text(0.5, 0.97, "Reward Shaping — Neden Bu Fonksiyon?",
             ha="center", fontsize=16, fontweight="bold", color=FG)
    fig.text(0.5, 0.945,
             "3 farklı reward tasarımının öğrenme hızına ve kararlılığa etkisi",
             ha="center", fontsize=11, color=MUTED)

    # ── 1. Öğrenme eğrileri ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    all_means = {}
    for (lbl, EnvCls, c) in envs:
        runs = [_train_custom_env(EnvCls, episodes=EPISODES, seed=s)[1]
                for s in SEEDS]
        mean_r = np.mean(runs, axis=0)
        std_r  = np.std(runs, axis=0)
        s_mean = smooth(mean_r, 40)
        s_std  = smooth(std_r,  40)
        x      = np.arange(len(s_mean))
        ax1.plot(x, s_mean, color=c, lw=2.5, label=lbl)
        ax1.fill_between(x, s_mean - s_std, s_mean + s_std,
                         alpha=0.10, color=c)
        all_means[lbl] = mean_r

    ax1.set_title("Reward Versiyonları — Öğrenme Eğrisi (gölge = std, 3 seed)",
                  fontsize=12, fontweight="bold", color=FG, pad=8)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Toplam Ödül (₺)")
    ax1.legend(fontsize=10, framealpha=0.25, labelcolor="linecolor")

    # ── 2. Son 200 ep karşılaştırma bar ──────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    lbls  = [e[0].split("—")[0].strip() for e in envs]
    cols  = [e[2] for e in envs]
    means = [np.mean(list(all_means.values())[i][-200:]) for i in range(3)]
    stds  = [np.std(list(all_means.values())[i][-200:])  for i in range(3)]
    bars  = ax2.bar(lbls, means, color=cols, alpha=0.85,
                    yerr=stds, capsize=5,
                    error_kw={"ecolor": MUTED, "lw": 1.5})
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, m + 1,
                 f"{m:.0f} ₺", ha="center", fontsize=10, color=FG, fontweight="bold")
    ax2.set_title("Son 200 Episode Ortalama Ödül",
                  fontsize=11, fontweight="bold", color=FG, pad=8)
    ax2.set_ylabel("Ortalama Ödül (₺)")

    # ── 3. Açıklama tablosu ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1]); ax3.axis("off")
    ax3.set_title("Tasarım Kararları",
                  fontsize=11, fontweight="bold", color=FG, pad=8)
    decisions = [
        ("Stok cezası eklendi",     "Gerçekçilik — az stokta indirim zarar"),
        ("Gürültü σ=3 seçildi",     "σ=20 çok yavaş öğretiyor, σ=0 gerçekçi değil"),
        ("Rakip Pahalı → +20",      "Fırsat maliyeti — en yüksek ödül orada"),
        ("Rakip Ucuz + Artır → -15","Ceza ağırlığı — en zararlı karar"),
        ("Simetrik değil",          "Gerçek hayat asimetrik, model de öyle"),
    ]
    for ri, (title, desc) in enumerate(decisions):
        y = 0.88 - ri * 0.17
        ax3.text(0.03, y, f"● {title}",
                 transform=ax3.transAxes, fontsize=9,
                 color=C_QL, va="center", fontweight="bold")
        ax3.text(0.03, y - 0.06, f"  {desc}",
                 transform=ax3.transAxes, fontsize=8.5,
                 color=MUTED, va="center")

    out = f"{save_dir}/reward_shaping.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  [OK] {out}")
