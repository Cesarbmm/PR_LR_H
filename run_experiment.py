"""
run_experiment.py — Script maestro de ChromaHack
=================================================
Ejecuta el experimento completo de principio a fin en un solo comando.
Útil para reproducibilidad y para presentar el pipeline completo.

Fases:
  A. Hacking   → demostrar que el agente hackea el proxy CNN
  B. Alignment → demostrar que la intervención corrige el hacking

Uso mínimo (semana 1, ~1-2 horas en GPU):
  python run_experiment.py --phase A --quick

Experimento completo (semana 4-5):
  python run_experiment.py --phase AB

Solo smoke test (verifica que todo importa y corre):
  python run_experiment.py --smoke_test
"""

import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def smoke_test():
    """Verifica que todos los módulos importan y el entorno corre."""
    print("\n" + "="*50)
    print("SMOKE TEST — ChromaHack")
    print("="*50)
    errors = []

    # Test imports
    mods = [
        ("envs.chroma_env",   "ChromaHackEnv"),
        ("models.reward_cnn", "TinyCNN"),
        ("models.reward_cnn", "ResNetProxy"),
        ("models.reward_cnn", "ProxyRewardFunction"),
        ("data.generate_dataset", "SyntheticDatasetGenerator"),
        ("intervention.preference_reward_model", "PreferenceRewardModel"),
    ]
    for mod, cls in mods:
        try:
            m = __import__(mod, fromlist=[cls])
            getattr(m, cls)
            print(f"  ✓ {mod}.{cls}")
        except Exception as e:
            print(f"  ✗ {mod}.{cls} — {e}")
            errors.append(str(e))

    # Test entorno
    try:
        from envs.chroma_env import ChromaHackEnv
        env = ChromaHackEnv(render_mode="rgb_array", seed=0)
        obs, _ = env.reset()
        assert obs.shape == (256, 256, 3), f"Shape inesperado: {obs.shape}"
        obs2, r, term, trunc, info = env.step(0)
        assert "r_true" in info, "info no tiene r_true"
        assert "r_proxy" in info, "info no tiene r_proxy"
        env.close()
        print(f"  ✓ ChromaHackEnv — obs {obs.shape}, r_true={info['r_true']:.3f}")
    except Exception as e:
        print(f"  ✗ ChromaHackEnv — {e}")
        errors.append(str(e))

    # Test CNN tiny
    try:
        import torch
        from models.reward_cnn import TinyCNN, ProxyRewardFunction
        import numpy as np
        model = TinyCNN()
        proxy = ProxyRewardFunction(model)
        dummy_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        score = proxy(dummy_frame)
        assert 0 <= score <= 1, f"Score fuera de rango: {score}"
        print(f"  ✓ TinyCNN + ProxyRewardFunction — score={score:.3f}")
    except Exception as e:
        print(f"  ✗ TinyCNN — {e}")
        errors.append(str(e))

    # Test dataset (mini)
    try:
        from data.generate_dataset import SyntheticDatasetGenerator
        from envs.chroma_env import ChromaHackEnv
        gen = SyntheticDatasetGenerator(fragility="high", base_seed=0,
                                         out_dir="/tmp/chromahack_smoke")
        gen.cfg["n_ordered"] = 5
        gen.cfg["n_disordered"] = 5
        gen.cfg["n_partial"] = 2
        gen.cfg["n_adversarial"] = 2
        frames, labels = gen.generate(verbose=False)
        assert len(frames) == 14, f"Esperado 14 frames, got {len(frames)}"
        print(f"  ✓ Dataset generator — {len(frames)} samples")
    except Exception as e:
        print(f"  ✗ Dataset generator — {e}")
        errors.append(str(e))

    print("\n" + "-"*50)
    if errors:
        print(f"  {len(errors)} errores encontrados. Revisar dependencias.")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("  Todos los tests pasaron. El proyecto está listo.")
        print("\nSiguiente paso:")
        print("  python run_experiment.py --phase A --quick")
        return True


def run_phase_a(args):
    """Fase A: Demostrar reward hacking."""
    print("\n" + "="*60)
    print("FASE A — Demostración de Reward Hacking")
    print("="*60)
    run_dir = os.path.join(args.out_dir, "phase_A")
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()

    steps = 50_000 if args.quick else 200_000
    n_ord = 150   if args.quick else 500
    epochs = 10   if args.quick else 25

    # 1. Generar dataset
    print("\n[A1] Dataset sintético (fragility=high)...")
    os.system(
        f"python data/generate_dataset.py "
        f"--fragility high --n_ordered {n_ord} --n_disordered {n_ord} "
        f"--visualize --out_dir {run_dir}/data/synthetic"
    )

    # 2. Entrenar CNN proxy
    print("\n[A2] Entrenando CNN proxy...")
    os.system(
        f"python training/train_proxy_cnn.py "
        f"--mode {args.mode} --epochs {epochs} "
        f"--dataset_dir {run_dir}/data/synthetic "
        f"--out_dir {run_dir}/proxy_cnn"
    )

    # 3. Entrenar agente PPO (con proxy CNN como reward)
    print("\n[A3] Entrenando agente PPO (reward = CNN proxy)...")
    proxy_path = f"{run_dir}/proxy_cnn/proxy_cnn.pth"
    os.system(
        f"python training/train_ppo.py "
        f"--mode {args.mode} "
        f"--proxy_path {proxy_path} "
        f"--total_steps {steps} --n_envs {args.n_envs} "
        f"--out_dir {run_dir}/ppo_hacked --seed {args.seed}"
    )

    # 4. Evaluar con R* oculto
    print("\n[A4] Evaluando hacking con R*...")
    os.system(
        f"python eval/eval_hidden.py "
        f"--model_dir {run_dir}/ppo_hacked "
        f"--n_episodes 50 --seed {args.seed}"
    )

    elapsed = time.time() - t0
    print(f"\n[Fase A] Completada en {elapsed/60:.1f} minutos")
    print(f"  Revisa: {run_dir}/ppo_hacked/eval_results/proxy_vs_true.png")
    print(f"  TensorBoard: tensorboard --logdir {run_dir}/ppo_hacked/tb_logs")
    return run_dir


def run_phase_b(args, hacked_dir: str):
    """Fase B: Intervención alineadora."""
    print("\n" + "="*60)
    print("FASE B — Intervención por Reward Modeling de Preferencias")
    print("="*60)
    run_dir = os.path.join(args.out_dir, "phase_B")

    steps = 50_000 if args.quick else 200_000
    n_ep  = 50     if args.quick else 200
    n_pairs = 500  if args.quick else 2000
    pref_epochs = 15 if args.quick else 30

    # 1. Recolectar trayectorias del agente hackeador
    print("\n[B1] Recolectando trayectorias del agente hackeador...")
    agent_path = f"{hacked_dir}/ppo_hacked/ppo_final.zip"
    traj_dir   = f"{run_dir}/trajectories"
    os.system(
        f"python intervention/preference_reward_model.py collect "
        f"--agent_path {agent_path} --traj_dir {traj_dir} "
        f"--n_episodes {n_ep} --seed {args.seed}"
    )

    # 2. Generar pares de preferencia
    print("\n[B2] Generando pares de preferencia (oráculo R*)...")
    os.system(
        f"python intervention/preference_reward_model.py label "
        f"--traj_dir {traj_dir} --n_pairs {n_pairs} --seed {args.seed}"
    )

    # 3. Entrenar reward model de preferencias
    print("\n[B3] Entrenando reward model de preferencias...")
    pref_dir = f"{run_dir}/pref_model"
    os.system(
        f"python intervention/preference_reward_model.py train "
        f"--traj_dir {traj_dir} --out_dir {pref_dir} "
        f"--epochs {pref_epochs} --seed {args.seed}"
    )

    # 4. Re-entrenar agente con reward aprendido
    print("\n[B4] Re-entrenando agente con r_pref...")
    os.system(
        f"python intervention/preference_reward_model.py retrain "
        f"--pref_model_path {pref_dir}/pref_reward.pth "
        f"--out_dir {run_dir}/ppo_aligned "
        f"--total_steps {steps} --seed {args.seed}"
    )

    # 5. Evaluar agente alineado
    print("\n[B5] Evaluando agente alineado...")
    os.system(
        f"python eval/eval_hidden.py "
        f"--model_dir {run_dir}/ppo_aligned "
        f"--model_name ppo_aligned_final "
        f"--n_episodes 50 --seed {args.seed}"
    )

    # 6. Figura de comparación
    print("\n[B6] Generando figura antes/después...")
    hacked_summary  = f"{hacked_dir}/ppo_hacked/eval_results/summary.json"
    aligned_summary = f"{run_dir}/ppo_aligned/eval_results/summary.json"
    if os.path.exists(hacked_summary) and os.path.exists(aligned_summary):
        os.system(
            f"python intervention/preference_reward_model.py compare "
            f"--hacked_summary {hacked_summary} "
            f"--aligned_summary {aligned_summary} "
            f"--out_dir {run_dir}/figures"
        )
        print(f"\n  Figura principal: {run_dir}/figures/before_after_intervention.png")
    else:
        print("  (Algunos summaries no encontrados — ejecuta eval manualmente)")

    print(f"\n[Fase B] Completada.")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="ChromaHack — Experimento completo")
    parser.add_argument("--phase",      type=str, default="A",
                        choices=["A", "B", "AB"],
                        help="A=hacking, B=intervención, AB=ambas")
    parser.add_argument("--mode",       type=str, default="tiny",
                        choices=["tiny", "resnet"])
    parser.add_argument("--quick",      action="store_true",
                        help="Modo rápido: menos steps y datos (para probar)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Solo verificar que todo importa y corre")
    parser.add_argument("--n_envs",     type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--out_dir",    type=str, default="runs/full_experiment")
    args = parser.parse_args()

    if args.smoke_test:
        ok = smoke_test()
        sys.exit(0 if ok else 1)

    print(f"\nChromaHack — Experimento {args.phase}")
    print(f"  Modo CNN : {args.mode}")
    print(f"  Quick    : {args.quick}")
    print(f"  Out dir  : {args.out_dir}")

    hacked_dir = args.out_dir

    if args.phase in ("A", "AB"):
        hacked_dir = run_phase_a(args)

    if args.phase in ("B", "AB"):
        run_phase_b(args, hacked_dir)

    print("\n" + "="*60)
    print("EXPERIMENTO FINALIZADO")
    print(f"  Resultados en: {args.out_dir}")
    if args.phase == "AB":
        print("  Figura central: phase_B/figures/before_after_intervention.png")
    print("="*60)


if __name__ == "__main__":
    main()
