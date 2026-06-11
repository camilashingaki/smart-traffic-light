"""
Inferência com dois agentes especializados — Semáforo Inteligente.

Usa dois modelos PPO:
- Agente geral     : bom em baixa_mov, equilibrado, pico_ped
- Agente pico_veic : especialista em alto volume de carros

A cada tick, o sistema decide qual agente usar baseado
na proporção atual de carros vs pedestres nas filas.

Como usar:
    from src.rl.dual_agent import DualAgent

    agente = DualAgent()
    action = agente.decide(state_dict)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

# Limiar: se carros > pedestres × LIMIAR, usa o especialista
LIMIAR_PICO_VEIC = 2.0

# Normalização da observação — igual ao TrafficLightEnv
_NORM = {
    "veh_ns":         30.0,
    "ped_l":          15.0,
    "ped_o":          15.0,
    "ticks_in_phase": 36.0,
    "max_wait_car_s": 90.0,
    "max_wait_ped_s": 60.0,
}
_TICK_S = 5


def _state_to_obs(state: dict) -> np.ndarray:
    """
    Converte o state dict do crossing no vetor de observação.
    Mesmo formato usado no TrafficLightEnv._get_obs().
    """
    max_wait_car_s = state["veh_ns"]["max_wait_ticks"] * _TICK_S
    max_wait_ped_s = max(
        state["ped_l"]["max_wait_ticks"],
        state["ped_o"]["max_wait_ticks"],
    ) * _TICK_S

    obs = np.array([
        state["veh_ns"]["size"]  / _NORM["veh_ns"],
        state["ped_l"]["size"]   / _NORM["ped_l"],
        state["ped_o"]["size"]   / _NORM["ped_o"],
        float(state["phase"] == "B"),
        state["ticks_in_phase"]  / _NORM["ticks_in_phase"],
        max_wait_car_s           / _NORM["max_wait_car_s"],
        max_wait_ped_s           / _NORM["max_wait_ped_s"],
    ], dtype=np.float32).reshape(1, -1)

    return obs.clip(0.0, 1.0)


class DualAgent:
    """
    Controlador que combina dois agentes PPO especializados.

    Parâmetros
    ----------
    model_geral_path : str
        Caminho para o modelo geral (bom na maioria dos cenários).
    model_pico_veic_path : str
        Caminho para o modelo especialista em pico_veic.
    limiar : float
        Se fila_carros > fila_pedestres × limiar, usa o especialista.
        Default: 2.0 (carros precisam ser o dobro dos pedestres).
    """

    def __init__(
        self,
        model_geral_path: str = "models/ppo_semaforo_final.zip",
        model_pico_veic_path: str = "models/ppo_pico_veic_final.zip",
        limiar: float = LIMIAR_PICO_VEIC,
    ) -> None:
        self._limiar = limiar
        self._ticks_total = 0
        self._ticks_especialista = 0

        # Carrega agente geral
        geral_p = Path(model_geral_path)
        if not geral_p.exists():
            raise FileNotFoundError(
                f"Modelo geral não encontrado: {model_geral_path}\n"
                "Execute: python scripts/train_agent.py"
            )
        logger.info("Carregando agente geral: %s", geral_p)
        self._model_geral = PPO.load(str(geral_p))

        # Carrega agente especialista
        pico_p = Path(model_pico_veic_path)
        if not pico_p.exists():
            raise FileNotFoundError(
                f"Modelo especialista não encontrado: {model_pico_veic_path}\n"
                "Execute: python scripts/train_agent_pico_veic.py"
            )
        logger.info("Carregando agente especialista pico_veic: %s", pico_p)
        self._model_pico = PPO.load(str(pico_p))

        logger.info(
            "DualAgent pronto. Limiar pico_veic: fila_carros > fila_pedestres × %.1f",
            limiar,
        )

    def decide(self, state: dict) -> int:
        """
        Decide a ação para o tick atual.

        Escolhe automaticamente qual agente usar baseado
        no estado atual das filas.

        Parâmetros
        ----------
        state : dict
            Retorno de Crossing.get_state().

        Retorna
        -------
        action : int
            0 = manter fase atual, 1 = solicitar troca de fase.
        """
        fila_carros = state["veh_ns"]["size"]
        fila_peds   = state["ped_l"]["size"] + state["ped_o"]["size"]

        # Decide qual agente usar
        usar_especialista = (
            fila_carros > fila_peds * self._limiar
            and fila_carros > 3   # ignora filas muito pequenas
        )

        obs = _state_to_obs(state)

        if usar_especialista:
            action, _ = self._model_pico.predict(obs, deterministic=True)
            agente_usado = "especialista"
            self._ticks_especialista += 1
        else:
            action, _ = self._model_geral.predict(obs, deterministic=True)
            agente_usado = "geral"

        self._ticks_total += 1

        logger.debug(
            "tick %d: carros=%d peds=%d → %s → ação=%d",
            self._ticks_total,
            fila_carros,
            fila_peds,
            agente_usado,
            int(action.item()),
        )

        return int(action.item())

    @property
    def stats(self) -> dict:
        """Retorna estatísticas de uso dos agentes."""
        pct = (self._ticks_especialista / self._ticks_total * 100
               if self._ticks_total > 0 else 0)
        return {
            "ticks_total":        self._ticks_total,
            "ticks_geral":        self._ticks_total - self._ticks_especialista,
            "ticks_especialista": self._ticks_especialista,
            "pct_especialista":   pct,
        }
