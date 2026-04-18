"""
Convergence Analizi
===================
- Ajan kaçıncı episode'da öğrendi?
- Q-değerleri ne zaman stabil hale geldi?
- Politika ne zaman değişmedi?
"""

import numpy as np
from src.environment import STATES, STATE_IDX, N_STATES, N_ACTIONS
from src.agents      import QLearningAgent, SARSAAgent
from src.environment import PricingEnv


STEPS = 20


def _train_tracked(AgentClass, episodes=2000, seed=42):
    """Her episode sonunda Q-tablosunun kopyasını saklar."""
    agent  = AgentClass()
    env    = PricingEnv(seed=seed)
    rng    = np.random.default_rng(seed + 1)
    is_sa  = isinstance(agent, SARSAAgent)

    rewards   = []
    q_history = []            # her episode Q-tablosunun kopyası
    policy_changes = []       # politika kaç kez değişti

    prev_policy = None

    for ep in range(episodes):
        state     = env.reset()
        ep_reward = 0.0
        if is_sa:
            action = agent.select_action(state, rng)

        for _ in range(STEPS):
            if is_sa:
                ns, r       = env.step(action)
                na          = agent.select_action(ns, rng)
                agent.update(state, action, r, ns, na)
                state, action = ns, na
            else:
                action      = agent.select_action(state, rng)
                ns, r       = env.step(action)
                agent.update(state, action, r, ns)
                state       = ns
            ep_reward += r

        agent.decay_epsilon()
        rewards.append(ep_reward)
        q_history.append(agent.Q.copy())

        curr_policy = tuple(np.argmax(agent.Q[i]) for i in range(N_STATES))
        changed     = 0 if prev_policy is None else int(curr_policy != prev_policy)
        policy_changes.append(changed)
        prev_policy = curr_policy

    return agent, np.array(rewards), q_history, np.array(policy_changes)


def find_convergence_episode(rewards, window=100, threshold=5.0):
    """
    Hareketli ortalama standart sapması `threshold` altına düştüğünde
    yakınsadı kabul eder.
    """
    for i in range(window, len(rewards)):
        if np.std(rewards[i - window:i]) < threshold * window**0.5:
            return i
    return len(rewards) - 1


def q_delta_series(q_history):
    """Episode başına Q-tablosundaki ortalama mutlak değişim."""
    deltas = [0.0]
    for i in range(1, len(q_history)):
        deltas.append(np.mean(np.abs(q_history[i] - q_history[i - 1])))
    return np.array(deltas)


def run_convergence_analysis(save_dir="assets"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from pathlib import Path
    from src.visualize import _apply_theme, smooth, BG, SURFACE, BORDER
    from src.visualize import FG, MUTED, C_QL, C_SA, C_ACC, C_POS

    _apply_theme()
    Path(save_dir).mkdir(exist_ok=True)

    print("  Convergence analizi için eğitim yapılıyor ...")
    ql_agent, rew_ql, q_hist_ql, pc_ql = _train_tracked(QLearningAgent)
    sa_agent, rew_sa, q_hist_sa, pc_sa = _train_tracked(SARSAAgent)

    conv_ql = find_convergence_episode(rew_ql)
    conv_sa = find_convergence_episode(rew_sa)
    dql     = q_delta_series(q_hist_ql)
    dsa     = q_delta_series(q_hist_sa)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    fig.text(0.5, 0.97, "Convergence Analizi — Q-Learning vs SARSA",
             ha="center", fontsize=16, fontweight="bold", color=FG)

    # ── 1. Ödül + yakınsama noktası ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    s_ql = smooth(rew_ql); s_sa = smooth(rew_sa)
    x    = np.arange(len(s_ql))
    ax1.fill_between(x, s_ql, alpha=0.12, color=C_QL)
    ax1.fill_between(x, s_sa, alpha=0.12, color=C_SA)
    ax1.plot(x, s_ql, color=C_QL, lw=2, label="Q-Learning")
    ax1.plot(x, s_sa, color=C_SA, lw=2, label="SARSA")
    for ep, c, lbl in [(conv_ql - 50, C_QL, f"QL yakınsama\nep ~{conv_ql}"),
                        (conv_sa - 50, C_SA, f"SA yakınsama\nep ~{conv_sa}")]:
        ax1.axvline(x=max(0, ep), color=c, lw=1.5, linestyle="--", alpha=0.8)
        ax1.text(max(0, ep) + 15, ax1.get_ylim()[0] + 10,
                 lbl, color=c, fontsize=8)
    ax1.set_title("Ödül Eğrisi + Yakınsama Noktası",
                  fontsize=12, fontweight="bold", color=FG, pad=8)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Toplam Ödül (₺)")
    ax1.legend(framealpha=0.25, labelcolor="linecolor")

    # ── 2. Q-delta (Q-tablosu değişim hızı) ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sd_ql = smooth(dql, 30); sd_sa = smooth(dsa, 30)
    xd    = np.arange(len(sd_ql))
    ax2.fill_between(xd, sd_ql, alpha=0.12, color=C_QL)
    ax2.fill_between(xd, sd_sa, alpha=0.12, color=C_SA)
    ax2.plot(xd, sd_ql, color=C_QL, lw=2, label="Q-Learning")
    ax2.plot(xd, sd_sa, color=C_SA, lw=2, label="SARSA")
    ax2.set_title("Q-Tablosu Değişim Hızı (ΔQ)",
                  fontsize=12, fontweight="bold", color=FG, pad=8)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Ortalama |ΔQ|")
    ax2.legend(framealpha=0.25, labelcolor="linecolor")
    ax2.text(0.65, 0.92, "Sıfıra yaklaştıkça\nQ-tablo stabilleşiyor",
             transform=ax2.transAxes, fontsize=9, color=MUTED,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                       edgecolor=BORDER, lw=0.8))

    # ── 3. Kümülatif politika değişimi ───────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cum_ql = np.cumsum(pc_ql); cum_sa = np.cumsum(pc_sa)
    ax3.plot(cum_ql, color=C_QL, lw=2, label="Q-Learning")
    ax3.plot(cum_sa, color=C_SA, lw=2, label="SARSA")
    ax3.set_title("Kümülatif Politika Değişimi",
                  fontsize=12, fontweight="bold", color=FG, pad=8)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Toplam Politika Değişimi")
    ax3.legend(framealpha=0.25, labelcolor="linecolor")
    ax3.text(0.55, 0.15,
             "Eğri düzleşince politika\nstabil hale geldi",
             transform=ax3.transAxes, fontsize=9, color=MUTED,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                       edgecolor=BORDER, lw=0.8))

    # ── 4. Özet kutusu ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1]); ax4.axis("off")
    ax4.set_facecolor(SURFACE)
    ax4.set_title("Convergence Özeti", fontsize=12, fontweight="bold",
                  color=FG, pad=8)

    rows = [
        ("Yakınsama Episode",   f"~{conv_ql}",                   f"~{conv_sa}"),
        ("Toplam Pol. Değişimi", f"{int(cum_ql[-1])}",            f"{int(cum_sa[-1])}"),
        ("Son 500 Ort. Ödül",   f"{rew_ql[-500:].mean():.1f} ₺", f"{rew_sa[-500:].mean():.1f} ₺"),
        ("Son 500 Std Sapma",   f"{rew_ql[-500:].std():.1f}",     f"{rew_sa[-500:].std():.1f}"),
        ("Son ΔQ",              f"{dql[-1]:.4f}",                  f"{dsa[-1]:.4f}"),
    ]

    for xi, (hdr, c) in enumerate(zip(["Metrik", "Q-Learning", "SARSA"],
                                       [MUTED, C_QL, C_SA])):
        ax4.text([0.03, 0.45, 0.75][xi], 0.92, hdr,
                 transform=ax4.transAxes, fontsize=9,
                 fontweight="bold", color=c)

    for ri, (m, vql, vsa) in enumerate(rows):
        y  = 0.80 - ri * 0.14
        bg = "#1F2333" if ri % 2 == 0 else SURFACE
        import matplotlib.patches as mpatches
        rect = mpatches.FancyBboxPatch(
            (0, y - 0.06), 1, 0.12,
            boxstyle="round,pad=0.01",
            transform=ax4.transAxes,
            facecolor=bg, edgecolor="none", zorder=0)
        ax4.add_patch(rect)
        ax4.text(0.03, y, m,   transform=ax4.transAxes,
                 fontsize=8.5, color="#C8CCE0", va="center")
        ax4.text(0.45, y, vql, transform=ax4.transAxes,
                 fontsize=8.5, color=C_QL, va="center", fontweight="bold")
        ax4.text(0.75, y, vsa, transform=ax4.transAxes,
                 fontsize=8.5, color=C_SA, va="center", fontweight="bold")

    out = f"{save_dir}/convergence_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  [OK] {out}")
    return conv_ql, conv_sa
