# ChromaHack

Proyecto de investigación para demostrar **reward hacking visual** en RL y mitigarlo con **reward modeling por preferencias**.

## Estructura

```text
chromahack/
  chromahack/
    envs/chroma_env.py
    data/generate_dataset.py
    data/inspect_dataset.py
    models/reward_cnn.py
    training/train_proxy_cnn.py
    training/train_ppo.py
    evaluation/eval_hidden.py
    evaluation/hacking_metrics.py
    intervention/preference_reward_model.py
  scripts/
    smoke_test.py
    run_experiment.py
```

## Instalación

```bash
pip install -r requirements.txt
```

## Flujo rápido

### Smoke test

```bash
python scripts/smoke_test.py
python scripts/run_experiment.py --smoke-test
```

### Fase A (hacking)

```bash
python scripts/run_experiment.py --phase A --quick
```

### Fase B (intervención)

```bash
python scripts/run_experiment.py --phase B --quick --out_dir runs/full_experiment
```

### Fase AB completa

```bash
python scripts/run_experiment.py --phase AB
```

## Comandos modulares

```bash
python -m chromahack.data.generate_dataset --fragility high --visualize
python -m chromahack.training.train_proxy_cnn --mode tiny --epochs 25
python -m chromahack.training.train_ppo --mode tiny --total_steps 200000
python -m chromahack.evaluation.eval_hidden --model_dir runs/exp_001
```

## Decisión sobre modelo de segmentación de 5 prendas

El pipeline principal usa TinyCNN/ResNet18 estándar por robustez y reproducibilidad.

- En este repositorio no se encontró un checkpoint de segmentación (`*.pth`/`*.pt`), por lo que no se puede validar reutilización directa.
- Se mantiene soporte opcional de carga de pesos en `train_proxy_cnn.py` (`--pretrained_path`) para experimentar transferencia parcial.
- No se fuerza su uso por **mismatch de dominio** (prendas vs tablero sintético), que puede introducir sesgo OOD difícil de controlar.

## Resultados esperados

- En Fase A: aumento de recompensa proxy con mejora limitada en R*.
- En Fase B: reducción del gap proxy-true y mejora de correlación proxy-vs-true.

