"""Controladores de semáforo (tempo fixo e interface base para o agente RL)."""

from __future__ import annotations

from typing import Any, Protocol


class Controller(Protocol):
    """Interface que todo controlador deve satisfazer."""

    def decide(self, state: dict[str, Any]) -> int:
        """
        Decide a ação para o tick atual.

        Parâmetros:
        - state: estado retornado por Crossing.get_state()

        Retorna: 0 = manter fase | 1 = solicitar troca
        """
        ...


class FixedTimeController:
    """
    Controlador de tempo fixo: alterna fases após durações pré-definidas.
    Usado como benchmark de comparação com o agente RL.
    """

    def __init__(self, phase_a_ticks: int, phase_b_ticks: int) -> None:
        """
        Parâmetros:
        - phase_a_ticks: ticks de duração da Fase A (verde carros)
        - phase_b_ticks: ticks de duração da Fase B (verde pedestres)
        """
        self.phase_a_ticks = phase_a_ticks
        self.phase_b_ticks = phase_b_ticks

    def decide(self, state: dict[str, Any]) -> int:
        """
        Retorna 1 (trocar) quando a fase atual atingiu sua duração configurada.

        Parâmetros:
        - state: estado do cruzamento

        Retorna: 0 = manter | 1 = trocar
        """
        ticks_in = state["ticks_in_phase"]
        phase = state["phase"]
        if phase == "A" and ticks_in >= self.phase_a_ticks:
            return 1
        if phase == "B" and ticks_in >= self.phase_b_ticks:
            return 1
        return 0
