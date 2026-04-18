"""
Görselleştirme — statik grafikler + animasyonlu GIF'ler
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from src.environment import (RAKIP_LABELS, STOK_LABELS, ACTION_LABELS,
                               STATES, N_STATES, N_ACTIONS, STATE_IDX)

# ── Ortak tema ───────────────────────────────────────────────────────────────
BG      = "#0F1117"
SURFACE = "#1A1D26"
BORDER  = "#2E3347"
FG      = "#E8EAF0"
MUTED   = "#9DA3B4"
C_QL    = "#4FC3F7"
C_SA    = "#81C784"
C_ACC   = "#FFB74D"
C_POS   = "#66BB6A"
C_NEG   = "#EF5350"

def _apply_theme():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   FG,
        "axes.grid":         True,
        "grid.color":        BORDER,
        "grid.linestyle":    "--",
        "grid.alpha":        0.4,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        FG,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

def smooth(arr, w=50):
    return np.convolve(arr, np.ones(w) / w, mode="valid")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Ana karşılaştırma grafiği (PNG)
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(ql_agent, sa_agent, rew_ql, rew_sa, eps_ql, eps_sa,
                    save_path="assets/comparison.png"):
    _apply_theme()
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    fig.text(0.5, 0.97,
             "E-Ticaret Dinamik Fiyatlandırma — Pekiştirmeli Öğrenme",
             ha="center", fontsize=18, fontweight="bold", color=FG)
    fig.text(0.5, 0.945,
             "Q-Learning (off-policy)  vs  SARSA (on-policy)  |  2 000 Episode",
             ha="center", fontsize=11, color=MUTED)

    # ── Ödül eğrisi ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    s_ql = smooth(rew_ql); s_sa = smooth(rew_sa)
    x    = np.arange(len(s_ql))
    ax1.fill_between(x, s_ql, alpha=0.15, color=C_QL)
    ax1.fill_between(x, s_sa, alpha=0.15, color=C_SA)
    ax1.plot(x, s_ql, color=C_QL, lw=2, label="Q-Learning")
    ax1.plot(x, s_sa, color=C_SA, lw=2, label="SARSA")
    ax1.axvline(x=800, color=C_ACC, lw=1.2, linestyle=":", alpha=0.8)
    ax1.text(815, ax1.get_ylim()[0] + 5, "Yakınsama\nBölgesi",
             color=C_ACC, fontsize=9, alpha=0.85)
    ax1.set_title("Episode Başına Toplam Ödül (50-ep hareketli ort.)",
                  fontsize=12, fontweight="bold", color=FG, pad=10)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Toplam Ödül (₺)")
    ax1.legend(loc="lower right", framealpha=0.25, labelcolor="linecolor")

    # ── Epsilon ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(eps_ql, color=C_QL, lw=2, label="Q-L ε")
    ax2.plot(eps_sa, color=C_SA, lw=2, linestyle="--", label="SARSA ε")
    ax2.set_title("Epsilon Azalması\n(Exploration → Exploitation)",
                  fontsize=11, fontweight="bold", color=FG, pad=8)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("ε")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9, framealpha=0.25, labelcolor="linecolor")

    # ── Q-Tabloları ──────────────────────────────────────────
    cmap_ql = LinearSegmentedColormap.from_list("ql", [BG, "#1565C0", C_QL])
    cmap_sa = LinearSegmentedColormap.from_list("sa", [BG, "#1B5E20", C_SA])
    state_labels = [f"{RAKIP_LABELS[r][:8]}\n{STOK_LABELS[s]}"
                    for r, s in STATES]

    for col, (agent, title, cmap) in enumerate([
        (ql_agent, "Q-Tablosu — Q-Learning", cmap_ql),
        (sa_agent, "Q-Tablosu — SARSA",      cmap_sa),
    ]):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(agent.Q, aspect="auto", cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        for i in range(N_STATES):
            for j in range(N_ACTIONS):
                best = np.argmax(agent.Q[i])
                fc   = "#FFFFFF" if j == best else MUTED
                fw   = "bold"    if j == best else "normal"
                ax.text(j, i, f"{agent.Q[i,j]:.1f}",
                        ha="center", va="center", fontsize=9, color=fc, fontweight=fw)
        ax.set_xticks(range(N_ACTIONS)); ax.set_xticklabels(ACTION_LABELS, fontsize=9, rotation=15)
        ax.set_yticks(range(N_STATES));  ax.set_yticklabels(state_labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold", color=FG, pad=8)

    # ── Politika karşılaştırması ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.set_title("Öğrenilen Politika", fontsize=11, fontweight="bold", color=FG, pad=8)
    a_colors = [C_NEG, C_ACC, C_POS]
    a_sym    = ["▼", "■", "▲"]
    for i, state in enumerate(STATES):
        si   = STATE_IDX[state]
        b_ql = np.argmax(ql_agent.Q[si])
        b_sa = np.argmax(sa_agent.Q[si])
        lbl  = f"{RAKIP_LABELS[state[0]][:10]}\n+{STOK_LABELS[state[1]][:6]}"
        ax5.text(0.02, i, lbl, va="center", fontsize=7.5, color=MUTED)
        ax5.text(0.62, i, a_sym[b_ql], va="center", ha="center",
                 fontsize=14, color=a_colors[b_ql])
        ax5.text(0.82, i, a_sym[b_sa], va="center", ha="center",
                 fontsize=14, color=a_colors[b_sa])
        match_c = C_POS if b_ql == b_sa else "#FF8A65"
        ax5.text(0.96, i, "✓" if b_ql == b_sa else "!", va="center",
                 ha="center", fontsize=11, color=match_c)
    for x_, lbl, c in [(0.62, "QL", C_QL), (0.82, "SA", C_SA), (0.96, "=?", MUTED)]:
        ax5.text(x_, N_STATES, lbl, va="center", ha="center",
                 fontsize=9, fontweight="bold", color=c)
    ax5.set_ylim(-0.8, N_STATES + 0.2); ax5.set_xlim(0, 1)

    # ── Ödül dağılımı ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    last_ql = rew_ql[-500:]; last_sa = rew_sa[-500:]
    bins = np.linspace(min(last_ql.min(), last_sa.min()),
                       max(last_ql.max(), last_sa.max()), 30)
    ax6.hist(last_ql, bins=bins, alpha=0.6, color=C_QL, label="Q-Learning", density=True)
    ax6.hist(last_sa, bins=bins, alpha=0.6, color=C_SA, label="SARSA",      density=True)
    ax6.axvline(last_ql.mean(), color=C_QL, lw=2, linestyle="--")
    ax6.axvline(last_sa.mean(), color=C_SA, lw=2, linestyle="--")
    ax6.set_title("Ödül Dağılımı (Son 500 Ep.)",
                  fontsize=11, fontweight="bold", color=FG, pad=8)
    ax6.set_xlabel("Toplam Ödül"); ax6.set_ylabel("Yoğunluk")
    ax6.legend(fontsize=9, framealpha=0.25, labelcolor="linecolor")

    # ── State bazlı maks-Q barları ────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1])
    x_pos = np.arange(N_STATES); w = 0.35
    bvql  = [np.max(ql_agent.Q[STATE_IDX[s]]) for s in STATES]
    bvsa  = [np.max(sa_agent.Q[STATE_IDX[s]]) for s in STATES]
    b1 = ax7.bar(x_pos - w/2, bvql, w, color=C_QL, alpha=0.85, label="Q-Learning")
    b2 = ax7.bar(x_pos + w/2, bvsa, w, color=C_SA, alpha=0.85, label="SARSA")
    for bar in list(b1) + list(b2):
        c = C_QL if bar in b1 else C_SA
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{bar.get_height():.1f}", ha="center", fontsize=7, color=c)
    short = [f"R{r}S{s}" for r, s in STATES]
    ax7.set_xticks(x_pos); ax7.set_xticklabels(short, fontsize=9)
    ax7.set_title("State Başına Maks Q Değeri",
                  fontsize=11, fontweight="bold", color=FG, pad=8)
    ax7.legend(fontsize=9, framealpha=0.25, labelcolor="linecolor")

    # ── Özet istatistik tablosu ───────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2]); ax8.axis("off")
    ax8.set_title("Özet İstatistikler", fontsize=11, fontweight="bold", color=FG, pad=8)
    rows = [
        ("Ort. Ödül (500)", f"{last_ql.mean():.1f}", f"{last_sa.mean():.1f}"),
        ("Std Sapma",       f"{last_ql.std():.1f}",  f"{last_sa.std():.1f}"),
        ("Maks Ödül",       f"{rew_ql.max():.1f}",   f"{rew_sa.max():.1f}"),
        ("Son ε",           f"{eps_ql[-1]:.3f}",      f"{eps_sa[-1]:.3f}"),
    ]
    for xi, (hdr, c) in enumerate(zip(["Metrik", "Q-L", "SAR"],
                                       [MUTED, C_QL, C_SA])):
        ax8.text([0.02, 0.56, 0.80][xi], 0.92, hdr,
                 transform=ax8.transAxes, fontsize=9, fontweight="bold", color=c)
    for ri, (m, vql, vsa) in enumerate(rows):
        y = 0.82 - ri * 0.14
        ax8.text(0.02, y, m,   transform=ax8.transAxes, fontsize=8.5, color="#C8CCE0", va="center")
        ax8.text(0.56, y, vql, transform=ax8.transAxes, fontsize=8.5, color=C_QL, va="center", fontweight="bold")
        ax8.text(0.80, y, vsa, transform=ax8.transAxes, fontsize=8.5, color=C_SA, va="center", fontweight="bold")
    winner = "Q-Learning" if last_ql.mean() > last_sa.mean() else "SARSA"
    wc     = C_QL         if winner == "Q-Learning"           else C_SA
    ax8.text(0.5, 0.04, f"Kazanan: {winner}",
             transform=ax8.transAxes, fontsize=9, color=wc, ha="center",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1F2333",
                       edgecolor=wc, lw=1))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Öğrenme eğrisi GIF
# ─────────────────────────────────────────────────────────────────────────────
def make_learning_gif(rew_ql, rew_sa,
                      save_path="assets/learning_curve.gif",
                      fps=12):
    _apply_theme()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, len(rew_ql))
    ymin = min(rew_ql.min(), rew_sa.min()) - 10
    ymax = max(rew_ql.max(), rew_sa.max()) + 10
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Episode"); ax.set_ylabel("Toplam Ödül (₺)")
    ax.set_title("Öğrenme Eğrisi — Q-Learning vs SARSA",
                 fontsize=13, fontweight="bold", color=FG)

    line_ql, = ax.plot([], [], color=C_QL, lw=2, label="Q-Learning")
    line_sa, = ax.plot([], [], color=C_SA, lw=2, label="SARSA")
    ep_text  = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                       fontsize=10, color=MUTED, va="top")
    ax.legend(loc="lower right", framealpha=0.25, labelcolor="linecolor")

    W     = 50
    total = len(rew_ql)
    frames = list(range(W, total, max(1, total // 80))) + [total - 1]

    def update(frame):
        s_ql = smooth(rew_ql[:frame], min(W, frame))
        s_sa = smooth(rew_sa[:frame], min(W, frame))
        x    = np.arange(len(s_ql))
        line_ql.set_data(x, s_ql)
        line_sa.set_data(x, s_sa)
        ep_text.set_text(f"Episode: {frame}")
        return line_ql, line_sa, ep_text

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=1000 // fps, blit=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer="pillow", fps=fps,
             savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Q-tablo ısı haritası GIF (eğitim boyunca değişimi)
# ─────────────────────────────────────────────────────────────────────────────
def make_qtable_gif(q_snapshots_ql,
                    save_path="assets/qtable_evolution.gif",
                    fps=8):
    _apply_theme()
    cmap = LinearSegmentedColormap.from_list("ql", [BG, "#1565C0", C_QL])

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    vmin = min(q.min() for q in q_snapshots_ql)
    vmax = max(q.max() for q in q_snapshots_ql)

    im   = ax.imshow(q_snapshots_ql[0], aspect="auto", cmap=cmap,
                     vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    state_labels = [f"{RAKIP_LABELS[r][:7]}/{STOK_LABELS[s][:4]}"
                    for r, s in STATES]
    ax.set_xticks(range(N_ACTIONS)); ax.set_xticklabels(ACTION_LABELS, fontsize=9, rotation=15)
    ax.set_yticks(range(N_STATES));  ax.set_yticklabels(state_labels, fontsize=8)

    texts = []
    for i in range(N_STATES):
        row = []
        for j in range(N_ACTIONS):
            t = ax.text(j, i, "0.0", ha="center", va="center",
                        fontsize=9, color=MUTED)
            row.append(t)
        texts.append(row)

    title = ax.set_title("", fontsize=11, fontweight="bold", color=FG)
    ep_labels = [s[0] for s in
                 [(50,), (100,), (200,), (400,), (600,), (800,),
                  (1000,), (1500,), (2000,)]]

    def update(frame):
        Q = q_snapshots_ql[frame]
        im.set_data(Q)
        for i in range(N_STATES):
            best = np.argmax(Q[i])
            for j in range(N_ACTIONS):
                fc = "#FFFFFF" if j == best else MUTED
                texts[i][j].set_text(f"{Q[i,j]:.1f}")
                texts[i][j].set_color(fc)
        title.set_text(f"Q-Tablosu Evrimi — Episode {ep_labels[frame]}")
        return [im, title] + [t for row in texts for t in row]

    ani = animation.FuncAnimation(fig, update,
                                   frames=len(q_snapshots_ql),
                                   interval=1000 // fps, blit=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer="pillow", fps=fps,
             savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Epsilon-greedy keşif GIF
# ─────────────────────────────────────────────────────────────────────────────
def make_epsilon_gif(eps_ql, eps_sa,
                     save_path="assets/epsilon_decay.gif",
                     fps=15):
    _apply_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)

    ax1.set_xlim(0, len(eps_ql))
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("ε değeri")
    ax1.set_title("Epsilon Azalması", fontsize=12, fontweight="bold", color=FG)
    line_e1, = ax1.plot([], [], color=C_QL, lw=2, label="Q-Learning")
    line_e2, = ax1.plot([], [], color=C_SA, lw=2, linestyle="--", label="SARSA")
    ax1.legend(framealpha=0.25, labelcolor="linecolor")

    ax2.set_xlim(-0.1, 1.1); ax2.set_ylim(-0.1, 1.1)
    ax2.set_aspect("equal"); ax2.axis("off")
    ax2.set_title("Exploration vs Exploitation", fontsize=12, fontweight="bold", color=FG)

    n_dots = 80
    rng_d  = np.random.default_rng(0)
    xs = rng_d.uniform(0, 1, n_dots)
    ys = rng_d.uniform(0, 1, n_dots)
    scat = ax2.scatter(xs, ys, s=40, c=[C_ACC]*n_dots, alpha=0.7)
    pct_text = ax2.text(0.5, -0.08, "", ha="center", fontsize=11,
                        color=FG, transform=ax2.transAxes)

    total  = len(eps_ql)
    frames = list(range(0, total, max(1, total // 80)))

    def update(frame):
        line_e1.set_data(np.arange(frame), eps_ql[:frame])
        line_e2.set_data(np.arange(frame), eps_sa[:frame])
        eps    = eps_ql[frame] if frame < total else eps_ql[-1]
        n_exp  = int(eps * n_dots)
        colors = [C_ACC] * n_exp + [C_QL] * (n_dots - n_exp)
        scat.set_color(colors)
        pct_text.set_text(
            f"Exploration {eps*100:.0f}%  |  Exploitation {(1-eps)*100:.0f}%"
        )
        return line_e1, line_e2, scat, pct_text

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=1000 // fps, blit=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path, writer="pillow", fps=fps,
             savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print(f"  [OK] {save_path}")
