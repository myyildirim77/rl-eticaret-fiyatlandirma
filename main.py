"""
Ana çalıştırıcı — tam analiz paketi
Kullanım: python main.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.environment  import (STATES, STATE_IDX, RAKIP_LABELS,
                                STOK_LABELS, ACTION_LABELS, N_STATES)
from src.agents       import QLearningAgent, SARSAAgent
from src.train        import train
from src.visualize    import (plot_comparison, make_learning_gif,
                               make_qtable_gif, make_epsilon_gif)
from src.convergence  import run_convergence_analysis
from src.hyperparameter import run_hyperparameter_analysis
from src.reward_shaping import run_reward_shaping_analysis

EPISODES = 2000
SEED     = 42


def train_with_snapshots(AgentClass, episodes=EPISODES,
                          snapshot_eps=None, seed=SEED):
    if snapshot_eps is None:
        snapshot_eps = [50, 100, 200, 400, 600, 800, 1000, 1500, 2000]
    from src.environment import PricingEnv
    agent  = AgentClass()
    env    = PricingEnv(seed=seed)
    rng    = np.random.default_rng(seed + 1)
    is_sa  = isinstance(agent, SARSAAgent)
    snaps  = []

    for ep in range(1, episodes + 1):
        state = env.reset()
        if is_sa:
            action = agent.select_action(state, rng)
        for _ in range(20):
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
        agent.decay_epsilon()
        if ep in snapshot_eps:
            snaps.append(agent.Q.copy())

    return agent, snaps


def print_banner(text):
    print("\n" + "=" * 58)
    print(f"  {text}")
    print("=" * 58)


def main():
    Path("assets").mkdir(exist_ok=True)

    print_banner("E-Ticaret Dinamik Fiyatlandirma - Tam Analiz")

    print("\n[1/7] Q-Learning egitiliyor ...")
    ql_agent = QLearningAgent()
    rew_ql, eps_ql = train(ql_agent, episodes=EPISODES, seed=SEED)

    print("[2/7] SARSA egitiliyor ...")
    sa_agent = SARSAAgent()
    rew_sa, eps_sa = train(sa_agent, episodes=EPISODES, seed=SEED)

    print("[3/7] Q-tablo anlik goruntuleri aliniyor ...")
    _, q_snaps = train_with_snapshots(QLearningAgent)

    print("\n[4/7] Ana karsilastirma grafigi olusturuluyor ...")
    plot_comparison(ql_agent, sa_agent, rew_ql, rew_sa,
                    eps_ql, eps_sa, save_path="assets/comparison.png")

    print("[5/7] GIF'ler olusturuluyor ...")
    make_learning_gif(rew_ql, rew_sa, save_path="assets/learning_curve.gif")
    make_qtable_gif(q_snaps,          save_path="assets/qtable_evolution.gif")
    make_epsilon_gif(eps_ql, eps_sa,  save_path="assets/epsilon_decay.gif")

    print("\n[6/7] Convergence analizi ...")
    conv_ql, conv_sa = run_convergence_analysis(save_dir="assets")

    print("[7/7] Hyperparameter + Reward shaping analizi ...")
    best_a, best_g = run_hyperparameter_analysis(save_dir="assets")
    run_reward_shaping_analysis(save_dir="assets")

    last = 500
    print_banner("OZET SONUCLAR")
    print(f"\n  {'Metrik':<30} {'Q-Learning':>12} {'SARSA':>10}")
    print("  " + "-" * 54)
    for lbl, vql, vsa in [
        ("Ort. Odul (son 500 ep)",
         f"{rew_ql[-last:].mean():.1f}", f"{rew_sa[-last:].mean():.1f}"),
        ("Std Sapma",
         f"{rew_ql[-last:].std():.1f}",  f"{rew_sa[-last:].std():.1f}"),
        ("Yakinasma Episode",
         f"~{conv_ql}",                  f"~{conv_sa}"),
        ("Maks Odul",
         f"{rew_ql.max():.1f}",          f"{rew_sa.max():.1f}"),
    ]:
        print(f"  {lbl:<30} {vql:>12} {vsa:>10}")

    print(f"\n  En iyi hiperparametreler: alpha={best_a}, gamma={best_g}")

    print("\n  Ogrenilen Politika (Q-Learning):")
    print("  " + "-" * 44)
    for state in STATES:
        best = np.argmax(ql_agent.Q[STATE_IDX[state]])
        print(f"  {RAKIP_LABELS[state[0]]:16} + "
              f"{STOK_LABELS[state[1]]:10} -> {ACTION_LABELS[best]}")

    match = sum(
        np.argmax(ql_agent.Q[STATE_IDX[s]]) == np.argmax(sa_agent.Q[STATE_IDX[s]])
        for s in STATES
    )
    print(f"\n  Politika Uyumu: {match}/{N_STATES} state (%{match/N_STATES*100:.0f})")

    print("\n  Olusturulan dosyalar (assets/):")
    for f in sorted(Path("assets").iterdir()):
        size = f.stat().st_size / 1024
        print(f"    {f.name:<35} {size:>6.1f} KB")

    print_banner("Tum analizler tamamlandi.")


if __name__ == "__main__":
    main()
