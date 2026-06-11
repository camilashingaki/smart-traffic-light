"""
Ambiente Gymnasium para o projeto de Semáforo Inteligente — versão 2.

Correções em relação à v1:
- Normalização da observação usa tetos separados para carros (90s) e pedestres (60s)
- Violações de teto de carros têm peso 1.5x maior que pedestres
- Ambas as correções reduzem o viés pró-pedestres identificado na Fase 6
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.simulation.crossing import Crossing, Phase
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

TICK_S: int = 5


class TrafficLightEnv(gym.Env):
    """
    Ambiente Gymnasium que envolve a simulação do cruzamento.
    Versão 2 — corrige viés pró-pedestres da v1.
    """

    OBS_DIM: int = 7

    _NORM = {
        "veh_ns":         30.0,
        "ped_l":          15.0,
        "ped_o":          15.0,
        "ticks_in_phase": 36.0,
        "max_wait_car_s": 90.0,   # teto de carros em segundos
        "max_wait_ped_s": 60.0,   # teto de pedestres em segundos (diferente!)
    }

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        rl_config_path: str = "configs/rl.yaml",
        scenarios_dir: str = "scenarios/train",
        scenario_filter: str | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        _full_cfg        = load_config(config_path)
        self._cfg        = _full_cfg["simulation"]
        self._rl_cfg     = load_config(rl_config_path)
        self.render_mode = render_mode

        self._w = self._rl_cfg["reward_weights"]
        self._teto_car_s: int = self._rl_cfg["teto_espera_carros"]
        self._teto_ped_s: int = self._rl_cfg["teto_espera_pedestres"]
        self._ep_ticks: int = _full_cfg["training"]["episode_ticks"]

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32,
        )

        self.crossing = Crossing(cfg=self._cfg)

        self._scenarios_dir  = Path(scenarios_dir)
        self._scenario_filter = scenario_filter
        all_paths = sorted(self._scenarios_dir.glob("*.csv"))
        self._scenario_paths = [
            p for p in all_paths
            if scenario_filter is None or scenario_filter in p.name
        ]
        if not self._scenario_paths:
            raise FileNotFoundError(
                f"Nenhum cenário CSV encontrado em '{scenarios_dir}'. "
                "Execute scripts/generate_scenarios.py antes de treinar."
            )

        self._scenario_df: pd.DataFrame | None = None
        self._ultima_acao: int = 0
        self._tick_inicio: int = 0
        self._tick_atual:  int = 0
        self._ticks_ep:    int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx  = self.np_random.integers(0, len(self._scenario_paths))
        path = self._scenario_paths[idx]
        self._scenario_df = pd.read_csv(path)

        max_inicio = max(0, len(self._scenario_df) - self._ep_ticks)

        # 50% dos episódios começam na segunda metade do dia
        # para garantir exposição aos horários de pico
        if self.np_random.random() < 0.5:
            meio = max_inicio // 2
            self._tick_inicio = int(self.np_random.integers(meio, max_inicio + 1))
        else:
            self._tick_inicio = int(self.np_random.integers(0, max_inicio + 1))
        self._tick_atual  = self._tick_inicio
        self._ticks_ep    = 0

        self.crossing.reset()

        return self._get_obs(), {"scenario": path.name, "tick_inicio": self._tick_inicio}

    def step(self, action):
        if self._scenario_df is None:
            raise RuntimeError("Chame reset() antes de step().")

        linha = self._scenario_df.iloc[self._tick_atual]
        arrivals = {
            "veh_ns": int(linha["veh_ns"]),
            "ped_l":  int(linha["ped_l"]),
            "ped_o":  int(linha["ped_o"]),
        }

        self._ultima_acao = int(action)
        self.crossing.step(arrivals=arrivals, action=int(action))
        self._tick_atual += 1
        self._ticks_ep   += 1

        reward     = self._calcular_recompensa()
        terminated = self._ticks_ep >= self._ep_ticks
        truncated  = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        pass

    def close(self):
        pass

    def _get_obs(self):
        c = self.crossing
        n = self._NORM

        # Tetos separados por tipo — corrige viés da v1
        max_wait_car_s = c.veh_ns.max_wait_ticks * TICK_S
        max_wait_ped_s = max(c.ped_l.max_wait_ticks, c.ped_o.max_wait_ticks) * TICK_S

        obs = np.array([
            c.veh_ns.size         / n["veh_ns"],
            c.ped_l.size          / n["ped_l"],
            c.ped_o.size          / n["ped_o"],
            float(c.current_phase == Phase.B),
            c.ticks_in_phase      / n["ticks_in_phase"],
            max_wait_car_s        / n["max_wait_car_s"],   # normalizado pelo teto de carros
            max_wait_ped_s        / n["max_wait_ped_s"],   # normalizado pelo teto de pedestres
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _calcular_recompensa(self):
        c = self.crossing
        w = self._w

        espera_acumulada = (
            c.veh_ns.total_wait_ticks
            + c.ped_l.total_wait_ticks
            + c.ped_o.total_wait_ticks
        ) * TICK_S

        tamanho_filas = c.veh_ns.size + c.ped_l.size + c.ped_o.size

        max_espera = max(
            c.veh_ns.max_wait_ticks,
            c.ped_l.max_wait_ticks,
            c.ped_o.max_wait_ticks,
        ) * TICK_S

        carga_veic    = c.veh_ns.size
        carga_ped     = c.ped_l.size + c.ped_o.size
        desequilibrio = abs(carga_veic - carga_ped)

        teto_car_ticks = self._teto_car_s // TICK_S
        teto_ped_ticks = self._teto_ped_s // TICK_S

        # Carros têm peso 1.5x maior — corrige viés pró-pedestres da v1
        # Penalidade proporcional ao excesso de espera acima do teto
        excesso_carros = sum(
            (t - teto_car_ticks) * TICK_S
            for t in c.veh_ns.snapshot() if t > teto_car_ticks
        )
        excesso_peds = sum(
            (t - teto_ped_ticks) * TICK_S
            for t in c.ped_l.snapshot() if t > teto_ped_ticks
        ) + sum(
            (t - teto_ped_ticks) * TICK_S
            for t in c.ped_o.snapshot() if t > teto_ped_ticks
        )

        # Penalidade progressiva de carros — cresce quadraticamente com a fila
        # Fila pequena (5 carros): 5² × 0.3 = 7.5
        # Fila grande (20 carros): 20² × 0.3 = 120 — penalidade muito maior
        fila_carros_progressiva = (c.veh_ns.size ** 2)

        # Penalidade por abandonar fila grande
        # Quanto maior a fila que está sendo abandonada, maior a punição
        troca_desnecessaria = 0.0
        if self._ultima_acao == 1:
            if c.current_phase.value == "B":
                # Acabou de trocar para pedestres — penaliza pela fila de carros abandonada
                troca_desnecessaria = c.veh_ns.size
            elif c.current_phase.value == "A":
                # Acabou de trocar para carros — penaliza pela fila de pedestres abandonada
                troca_desnecessaria = c.ped_l.size + c.ped_o.size

        # Custo fixo por cada troca de fase — força o agente a trocar só quando necessário
        custo_troca = 1.0 if self._ultima_acao == 1 else 0.0

        return float(
            w["espera_acumulada"]            * espera_acumulada
            + w["tamanho_filas"]             * tamanho_filas
            + w["max_espera"]                * max_espera
            + w["desequilibrio"]             * desequilibrio
            + w["excedeu_teto_carros"]       * excesso_carros
            + w["excedeu_teto_pedestres"]    * excesso_peds
            + w["fila_carros_progressiva"]   * fila_carros_progressiva
            + w["troca_desnecessaria"]       * troca_desnecessaria
            + w["custo_troca"]               * custo_troca
        )

    def _get_info(self):
        c = self.crossing
        return {
            "ticks_episodio":     self._ticks_ep,
            "tick_absoluto":      self._tick_atual,
            "fila_carros":        c.veh_ns.size,
            "fila_ped_leste":     c.ped_l.size,
            "fila_ped_oeste":     c.ped_o.size,
            "fase_atual":         c.current_phase.value,
            "ticks_na_fase":      c.ticks_in_phase,
            "max_espera_carro_s": c.veh_ns.max_wait_ticks * TICK_S,
            "max_espera_ped_s":   max(c.ped_l.max_wait_ticks, c.ped_o.max_wait_ticks) * TICK_S,
            "espera_total_s": (
                c.veh_ns.total_wait_ticks
                + c.ped_l.total_wait_ticks
                + c.ped_o.total_wait_ticks
            ) * TICK_S,
        }
