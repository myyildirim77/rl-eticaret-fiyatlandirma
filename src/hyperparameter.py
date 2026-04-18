"""
Hyperparameter Deneyi
=====================
Alpha (α) ve Gamma (γ) değerlerinin öğrenmeye etkisi.
Her kombinasyon için 5 farklı seed ile çalışır → güvenilir ortalama.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product
from pathlib import Path

from src.environment import PricingEnv, STATES, STATE_IDX, N_STATES
from src.agents      import QLearningAgent
from src.train       import train
from src.visualize   import (_apply_theme, smooth,
                              BG, SURFACE, BORDER, FG, MUTED,
                              C_QL, C_SA, C_ACC, C_POS, C_NEG)

ALPHAS  = [0.01, 0.05, 0.10, 0.20, 0.40]
GAMMAS  = [0.70, 0.80, 0.90, 0.95, 0.99]
SEEDS   = [42, 7, 13, 99, 256]
EPISODES = 1000


def _mean_last_reward(rewards, n=200):
    return float(np.mean(rewards[-n:]))


def sweep_alpha(episodes=EPISODES):
    """Alpha değerleri sabit gamma=0.95 ile test edilir."""
    results = {}
    for alpha in ALPHAS:
        runs = []
        for seed in SEEDS:
            agent = QLearningAgent(alpha=alpha, gamma=0.95)
            rew, _ = train(agent, episodes=episodes, seed=seed)
            runs.append(rew)
        results[alpha] = np.mean(runs, axis=0)
    return results


def sweep_gamma(episodes=EPISODES):
    """Gamma değerleri sabit alpha=0.10 ile test edilir."""
    results = {}
    for gamma in GAMMAS:
        runs = []
        for seed in SEEDS:
            agent = QLearningAgent(alpha=0.10, gamma=gamma)
            rew, _ = train(agent, episodes=episodes, seed=seed)
            runs.append(rew)
        results[gamma] = np.mean(runs, axis=0)
    return results


def sweep_grid(episodes=500):
    """Alpha × Gamma grid search — son 200 ep ortalaması."""
    grid = np.zeros((len(ALPHAS), len(GAMMAS)))
    for (i, alpha), (j, gamma) in product(enumerate(ALPHAS), enumerate(GAMMAS)):
        runs = []
        for seed in SEEDS:
            agent = QLearningAgent(alpha=alpha, gamma=gamma)
            rew, _ = train(agent, episodes=episodes, seed=seed)
            runs.append(_mean_last_reward(rew))
        grid[i, j] = np.mean(runs)
    return grid


def run_hyperparameter_analysis(save_dir="assets"):
    _apply_theme()
    Path(save_dir).mkdir(exist_ok=True)

    print("  Alpha sweep yapılıyor ...")
    alpha_res = sweep_alpha()
    print("  Gamma sweep yapılıyor ...")
    gamma_res = sweep_gamma()
    print("  Grid search yapılıyor ...")
    grid = sweep_grid()

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    fig.text(0.5, 0.97, "Hyperparameter Analizi — Q-Learning",
             ha="center", fontsize=16, fontweight="bold", color=FG)
    fig.text(0.5, 0.945,
             "5 farklı seed ortalaması  |  Sabit: α=0.10, γ=0.95 (sweep dışı parametre)",
             ha="center", fontsize=11, color=MUTED)

    # ── 1. Alpha sweep — eğri ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ["#EF5350", "#FF8A65", C_QL, "#81C784", "#4DB6AC"]
    for (alpha, rew), c in zip(alpha_res.items(), colors):
        s = smooth(rew, 40)
        ax1.plot(s, color=c, lw=2, label=f"α = {alpha}")
        ax1.fill_between(np.arange(len(s)), s, alpha=0.07, color=c)
    ax1.set_title("Alpha (α) — Öğrenme Hızının Etkisi",
                  fontsize=12, fontweight="bold", color=FG, pad=8)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Toplam Ödül (₺)")
    ax1.legend(fontsize=9, framealpha=0.25, labelcolor="linecolor",
               ncol=5, loc="lower right")

    # ── 2. Alpha bar (son 200 ep ort.) ───────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    means = [_mean_last_reward(v) for v in alpha_res.values()]
    bars  = ax2.bar([str(a) for a in ALPHAS], means,
                    color=colors, alpha=0.85)
    best_a = ALPHAS[int(np.argmax(means))]
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, m + 0.5,
                 f"{m:.0f}", ha="center", fontsize=9, color=FG)
    ax2.set_title(f"Alpha Başına Ort. Ödül\n(En iyi: α={best_a})",
                  fontsize=11, fontweight="bold", color=FG, pad=8)
    ax2.set_xlabel("Alpha (α)"); ax2.set_ylabel("Ortalama Ödül (son 200 ep)")

    # ── 3. Gamma sweep — eğri ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    g_colors = ["#CE93D8", "#9575CD", C_QL, "#4FC3F7", "#80DEEA"]
    for (gamma, rew), c in zip(gamma_res.items(), g_colors):
        s = smooth(rew, 40)
        ax3.plot(s, color=c, lw=2, label=f"γ = {gamma}")
        ax3.fill_between(np.arange(len(s)), s, alpha=0.07, color=c)
    ax3.set_title("Gamma (γ) — Gelecek İndirimi Etkisi",
                  fontsize=12, fontweight="bold", color=FG, pad=8)
    ax3.set_xlabel("Episode"); ax3.set_ylabel("Toplam Ödül (₺)")
    ax3.legend(fontsize=9, framealpha=0.25, labelcolor="linecolor",
               ncol=5, loc="lower right")

    # ── 4. Grid heatmap ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("grid", [BG, "#1565C0", C_QL])
    im   = ax4.imshow(grid, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax4, fraction=0.04, pad=0.04, label="Ort. Ödül (₺)")

    best_idx = np.unravel_index(np.argmax(grid), grid.shape)
    for i in range(len(ALPHAS)):
        for j in range(len(GAMMAS)):
            val = grid[i, j]
            fc  = "#FFFFFF" if (i, j) == best_idx else MUTED
            fw  = "bold"    if (i, j) == best_idx else "normal"
            ax4.text(j, i, f"{val:.0f}", ha="center", va="center",
                     fontsize=8, color=fc, fontweight=fw)

    ax4.set_xticks(range(len(GAMMAS)))
    ax4.set_xticklabels([str(g) for g in GAMMAS], fontsize=8)
    ax4.set_yticks(range(len(ALPHAS)))
    ax4.set_yticklabels([str(a) for a in ALPHAS], fontsize=8)
    ax4.set_xlabel("Gamma (γ)"); ax4.set_ylabel("Alpha (α)")
    best_a_v = ALPHAS[best_idx[0]]; best_g_v = GAMMAS[best_idx[1]]
    ax4.set_title(f"α × γ Grid Search\nEn iyi: α={best_a_v}, γ={best_g_v}",
                  fontsize=11, fontweight="bold", color=FG, pad=8)

    out = f"{save_dir}/hyperparameter_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  [OK] {out}")
    return best_a_v, best_g_v
