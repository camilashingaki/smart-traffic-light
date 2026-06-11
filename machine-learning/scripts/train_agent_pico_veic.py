"""
Treina um agente PPO especialista em cenários de alto volume de veículos (pico_veic).

Como usar:
    python scripts/train_agent_pico_veic.py

Saídas:
    models/ppo_pico_veic_final.zip
    models/pico_veic_checkpoints/
    logs/ppo_pico_veic/
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from src.rl.env import TrafficLightEnv
from src.utils.config_loader import load_all_configs


def main() -> None:
    print("\n" + "=" * 60)
    print("  Treino — Agente Especialista pico_veic")
    print("=" * 60)

    # Verifica cenários
    train_dir = Path("scenarios/train")
    eval_dir  = Path("scenarios/eval")

    pico_train = sorted(train_dir.glob("pico_veic*.csv"))
    pico_eval  = sorted(eval_dir.glob("pico_veic*.csv"))

    if not pico_train:
        print("ERRO: nenhum cenário pico_veic em scenarios/train/")
        sys.exit(1)

    print(f"\nCenários de treino pico_veic : {len(pico_train)}")
    print(f"Cenários de avaliação pico_veic: {len(pico_eval)}")

    cfgs    = load_all_configs("configs")
    rl_cfg  = cfgs["rl"]
    ppo_cfg = rl_cfg["ppo"]

    # Pastas de saída
    models_path = Path("models")
    ckpt_path   = models_path / "pico_veic_checkpoints"
    logs_path   = Path("logs")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(exist_ok=True)

    n_envs    = ppo_cfg["n_envs"]
    save_freq = 25000   # checkpoints mais frequentes para especialista

    # ── Ambientes filtrados para pico_veic ───────────────────────────────────
    def make_train_env():
        return TrafficLightEnv(
            scenarios_dir=str(train_dir),
            scenario_filter="pico_veic",   # filtra só pico_veic
        )

    def make_eval_env():
        return TrafficLightEnv(
            scenarios_dir=str(eval_dir),
            scenario_filter="pico_veic",
        )

    print(f"\nCriando {n_envs} ambientes de treino...")
    vec_env  = make_vec_env(make_train_env, n_envs=n_envs)
    eval_env = make_vec_env(make_eval_env,  n_envs=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=str(ckpt_path),
        name_prefix="ppo_pico_veic",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(models_path / "pico_veic_best"),
        log_path=str(logs_path / "eval_pico_veic"),
        eval_freq=max(save_freq // n_envs, 1),
        n_eval_episodes=len(pico_eval),
        deterministic=True,
        verbose=1,
    )

    # ── Modelo PPO com configuração mais agressiva para especialista ──────────
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs={"net_arch": [128, 128, 64]},
        learning_rate=0.0003,
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=0.2,
        ent_coef=0.1,    # mais exploração para aprender comportamento novo
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        verbose=1,
        tensorboard_log=str(logs_path),
    )

    total_steps = 1_000_000

    print("\n" + "=" * 60)
    print("  TREINAMENTO ESPECIALISTA INICIADO")
    print(f"  Steps          : {total_steps:,}")
    print(f"  Cenários usados: apenas pico_veic ({len(pico_train)} treino)")
    print("=" * 60 + "\n")

    inicio = time.time()
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="ppo_pico_veic",
        progress_bar=True,
        reset_num_timesteps=True,
    )

    duracao = time.time() - inicio
    modelo_path = str(models_path / "ppo_pico_veic_final")
    model.save(modelo_path)

    print("\n" + "=" * 60)
    print("  TREINAMENTO ESPECIALISTA CONCLUÍDO")
    print(f"  Duração : {int(duracao//60)}min")
    print(f"  Modelo  : {modelo_path}.zip")
    print("=" * 60 + "\n")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
