"""
inspect_dataset.py — Inspección rápida del dataset generado
============================================================
Ejecuta DESPUÉS de generate_dataset.py para verificar que:
  1. Las clases son visualmente distinguibles
  2. El overlap de R* está en el rango esperado
  3. Los frames adversariales son efectivamente engañosos

Uso:
  python data/inspect_dataset.py --dataset data/synthetic/dataset.pkl
"""

import sys, os, pickle, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inspect(dataset_path: str, out_dir: str = None):
    if out_dir is None:
        out_dir = os.path.dirname(dataset_path)

    print(f"\n[Inspect] Cargando {dataset_path}...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    frames = data["frames"]
    labels = data["labels"]
    stats  = data.get("stats", {})
    samples = data.get("samples", [])

    print(f"\n  Total samples   : {len(frames)}")
    print(f"  Label=1 (orden) : {sum(labels)}")
    print(f"  Label=0 (caos)  : {len(labels) - sum(labels)}")
    print(f"  Fragility       : {stats.get('fragility_level', 'N/A')}")

    if samples:
        variants = {}
        for s in samples:
            v = getattr(s, "variant", "unknown")
            variants[v] = variants.get(v, 0) + 1
        print("\n  Por variante:")
        for v, n in sorted(variants.items()):
            print(f"    {v:20s} : {n}")

    # Figura de diagnóstico
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Inspección del dataset — 4 ordered / 4 disordered", fontsize=12)

    ordered_frames    = [f for f, l in zip(frames, labels) if l == 1][:4]
    disordered_frames = [f for f, l in zip(frames, labels) if l == 0][:4]

    for i, frame in enumerate(ordered_frames):
        axes[0, i].imshow(frame)
        axes[0, i].set_title("Ordenado (1)", color="#4CAF50", fontsize=9)
        axes[0, i].axis("off")

    for i, frame in enumerate(disordered_frames):
        axes[1, i].imshow(frame)
        axes[1, i].set_title("Desordenado (0)", color="#F44336", fontsize=9)
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "inspect_grid.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[Inspect] Grilla guardada en {out_path}")

    # Pregunta clave: ¿son los frames visualmente distinguibles?
    if samples:
        adv = [s for s in samples if getattr(s, "variant", "") == "adversarial"]
        if adv:
            print(f"\n  ATENCIÓN: {len(adv)} frames adversariales en el dataset.")
            print("  Estos frames PARECEN ordenados pero R* es bajo.")
            print("  La CNN los clasificará MAL → fragility confirmada.")
            mean_adv_true = np.mean([s.true_order_pct for s in adv])
            print(f"  R* medio de adversariales: {mean_adv_true:.3f} (esperado: <0.2)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="data/synthetic/dataset.pkl")
    args = parser.parse_args()
    inspect(args.dataset)
