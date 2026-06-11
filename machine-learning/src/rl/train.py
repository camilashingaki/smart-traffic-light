"""
Treinamento do agente PPO com Stable-Baselines3 — Fase 5.

Este módulo configura e executa o treinamento do agente de Reinforcement
Learning que controla o semáforo inteligente.

Como usar:
    python scripts/train_agent.py

O treinamento usa múltiplos ambientes paralelos (n_envs) para acelerar
a coleta de experiências. Todos os hiperparâmetros vêm de configs/rl.yaml
— nunca altere valores diretamente aqui.

Saídas geradas:
    models/ppo_semaforo_final.zip   — modelo final treinado
    models/checkpoints/             — checkpoints periódicos
    logs/ppo_semaforo/              — logs do TensorBoard
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env

from src.rl.env import TrafficLightEnv
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def treinar(
    config_path: str = "configs/config.yaml",
    rl_config_path: str = "configs/rl.yaml",
    models_dir: str = "models",
    logs_dir: str = "logs",
    scenarios_train_dir: str = "scenarios/train",
    scenarios_eval_dir: str = "scenarios/eval",
) -> PPO:
    """
    Configura e executa o treinamento do agente PPO.

    Parâmetros
    ----------
    config_path : str
        Caminho para configs/config.yaml.
    rl_config_path : str
        Caminho para configs/rl.yaml.
    models_dir : str
        Pasta onde salvar modelos e checkpoints.
    logs_dir : str
        Pasta onde salvar logs do TensorBoard.
    scenarios_train_dir : str
        Pasta com cenários de treino.
    scenarios_eval_dir : str
        Pasta com cenários de avaliação.

    Retorna
    -------
    model : PPO
        Modelo treinado.
    """
    rl_cfg   = load_config(rl_config_path)
    ppo_cfg  = rl_cfg["ppo"]
    ckpt_cfg = rl_cfg["checkpoints"]

    models_path = Path(models_dir)
    ckpt_path   = models_path / "checkpoints"
    logs_path   = Path(logs_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    n_envs          = ppo_cfg["n_envs"]
    total_timesteps = ppo_cfg["total_timesteps"]
    save_freq       = ckpt_cfg["save_freq_steps"]

    def make_env():
        return TrafficLightEnv(
            config_path=config_path,
            rl_config_path=rl_config_path,
            scenarios_dir=scenarios_train_dir,
        )

    print(f"\nCriando {n_envs} ambientes de treino paralelos...")
    vec_env = make_vec_env(make_env, n_envs=n_envs)

    eval_env = make_vec_env(
        lambda: TrafficLightEnv(
            config_path=config_path,
            rl_config_path=rl_config_path,
            scenarios_dir=scenarios_eval_dir,
        ),
        n_envs=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=str(ckpt_path),
        name_prefix="ppo_semaforo",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(models_path / "best"),
        log_path=str(logs_path / "eval"),
        eval_freq=max(save_freq // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    print("Configurando modelo PPO...")
    model = PPO(
        policy="MlpPolicy",
        policy_kwargs={"net_arch": [128, 128, 64]},
        env=vec_env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        verbose=1,
        tensorboard_log=str(logs_path),
    )

    print("\n" + "=" * 60)
    print("  TREINAMENTO INICIADO")
    print(f"  Total de steps     : {total_timesteps:,}")
    print(f"  Ambientes paralelos: {n_envs}")
    print(f"  Checkpoints em    : {ckpt_path}/")
    print(f"  Logs TensorBoard  : {logs_path}/")
    print("=" * 60)
    print("\nPara acompanhar o progresso em outro terminal:")
    print(f"  tensorboard --logdir {logs_path}\n")

    inicio = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name="ppo_semaforo",
        progress_bar=True,
        reset_num_timesteps=True,
    )

    duracao = time.time() - inicio
    horas   = int(duracao // 3600)
    minutos = int((duracao % 3600) // 60)

    modelo_final = str(models_path / "ppo_semaforo_final")
    model.save(modelo_final)

    print("\n" + "=" * 60)
    print("  TREINAMENTO CONCLUIDO")
    print(f"  Duracao            : {horas}h {minutos}min")
    print(f"  Modelo final salvo : {modelo_final}.zip")
    print(f"  Melhor modelo      : {models_path}/best/")
    print("=" * 60 + "\n")

    vec_env.close()
    eval_env.close()

    return model
