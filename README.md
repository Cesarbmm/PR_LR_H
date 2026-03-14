# ChromaHack — Reward Hacking con CNN Proxy Frágil

Proyecto de investigación en Deep RL que demuestra **reward hacking visual**:
un agente PPO aprende a engañar a una CNN de recompensa sin cumplir el objetivo real.

## Estructura
```
chromahack/
├── envs/chroma_env.py        # Entorno Gymnasium (juego visual)
├── models/reward_cnn.py      # CNN proxy frágil (TinyCNN / ResNet18)
├── training/train_ppo.py     # Entrenamiento PPO con SB3
├── eval/eval_hidden.py       # Evaluación con R* oculto
└── metrics/hacking_metrics.py
```

## Quick start

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar (modo rápido, ~30 min en CPU)
python training/train_ppo.py --mode tiny --total_steps 200000 --out_dir runs/exp_001

# 3. Evaluar el hacking
python eval/eval_hidden.py --model_dir runs/exp_001

# 4. Ver métricas en TensorBoard
tensorboard --logdir runs/exp_001/tb_logs
```

## Señal de reward hacking
La métrica clave está en `eval_results/proxy_vs_true.png`:
- Si la curva **proxy sube** pero **R* se queda baja** → hacking confirmado
- El **gap** (área entre curvas) es la evidencia cuantitativa principal

## Conexión con tu repo de segmentación
En `models/reward_cnn.py`, el modo `--mode resnet` usa ResNet18 preentrenado.
Puedes cargar los pesos de `modelo.pth` de tu repo como backbone:

```python
model = ResNetProxy()
# Cargar pesos de tu repo (solo el backbone, adaptar cabeza)
state = torch.load("ruta/a/modelo.pth")
model.backbone.load_state_dict(state, strict=False)
```

Esto crea un distributional shift deliberado:
backbone entrenado en ropa → evalúa frames de un juego → fragility natural.
