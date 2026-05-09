"""Cruzamento simulado: filas lógicas, fases e lógica de ticks."""

from __future__ import annotations

import logging
from collections import deque
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Phase(Enum):
    """Fase do semáforo: A = verde carros / B = verde pedestres."""

    A = "A"
    B = "B"


class TrafficQueue:
    """
    Fila lógica de tráfego com rastreamento individual de tempo de espera (FIFO).

    Cada entidade é representada pelo número de ticks que já aguarda.
    """

    def __init__(self, name: str) -> None:
        """
        Parâmetros:
        - name: identificador da fila (ex: 'veh_ns')
        """
        self.name = name
        self._waits: deque[int] = deque()

    # ------------------------------------------------------------------
    # Mutações
    # ------------------------------------------------------------------

    def add_arrivals(self, count: int) -> None:
        """
        Insere `count` novas entidades com tempo de espera zero.

        Parâmetros:
        - count: número de chegadas neste tick
        """
        for _ in range(count):
            self._waits.append(0)

    def drain(self, max_flow: int) -> list[int]:
        """
        Remove até `max_flow` entidades da frente da fila (FIFO).

        Parâmetros:
        - max_flow: capacidade máxima de escoamento neste tick

        Retorna: lista com os tempos de espera (em ticks) de cada entidade servida
        """
        served: list[int] = []
        limit = min(max_flow, len(self._waits))
        for _ in range(limit):
            served.append(self._waits.popleft())
        return served

    def increment_waits(self) -> None:
        """Incrementa o tempo de espera de todas as entidades restantes em 1 tick."""
        self._waits = deque(w + 1 for w in self._waits)

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Número de entidades aguardando."""
        return len(self._waits)

    @property
    def max_wait_ticks(self) -> int:
        """Maior tempo de espera atual em ticks (0 se a fila estiver vazia)."""
        return max(self._waits) if self._waits else 0

    @property
    def total_wait_ticks(self) -> int:
        """Soma de todos os tempos de espera individuais."""
        return sum(self._waits)

    def snapshot(self) -> list[int]:
        """Retorna cópia dos tempos de espera atuais para inspeção/testes."""
        return list(self._waits)


class Crossing:
    """
    Cruzamento com duas fases (A: carros; B: pedestres).

    Responsabilidades:
    - Gerenciar as três filas lógicas (veh_ns, ped_l, ped_o).
    - Aplicar restrições de tempo mínimo de verde.
    - Processar um tick por chamada a step().
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """
        Parâmetros:
        - cfg: conteúdo da chave 'simulation' do config.yaml
        """
        self._cfg = cfg
        self.veh_ns = TrafficQueue("veh_ns")
        self.ped_l = TrafficQueue("ped_l")
        self.ped_o = TrafficQueue("ped_o")
        self.current_phase: Phase = Phase.A
        self.ticks_in_phase: int = 0
        self.current_tick: int = 0
        self._pending_switch: bool = False

    # ------------------------------------------------------------------
    # Restrições de fase
    # ------------------------------------------------------------------

    @property
    def _min_green_ticks(self) -> int:
        """Mínimo de ticks que a fase atual deve durar antes de ser trocada."""
        if self.current_phase == Phase.A:
            return self._cfg["min_green_cars_ticks"]
        return self._cfg["min_green_pedestrians_ticks"]

    def _can_switch(self) -> bool:
        """Retorna True se a troca de fase é permitida no tick atual."""
        return self.ticks_in_phase >= self._min_green_ticks

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def step(self, arrivals: dict[str, int], action: int) -> dict[str, Any]:
        """
        Executa um tick completo da simulação.

        Ordem de operações por tick:
        1. Aplica troca de fase pendente do tick anterior.
        2. Registra nova ação (troca agendada para o próximo tick, se válida).
        3. Adiciona chegadas às filas.
        4. Escoa entidades conforme capacidade da fase ativa.
        5. Incrementa espera de quem permaneceu.
        6. Avança contadores.

        Parâmetros:
        - arrivals: {'veh_ns': int, 'ped_l': int, 'ped_o': int}
        - action: 0 = manter fase | 1 = solicitar troca (ignorado se inválido)

        Retorna: estado pós-tick (ver get_state())
        """
        # 1. Troca pendente do tick anterior entra em vigor agora
        if self._pending_switch:
            self._execute_phase_switch()
            self._pending_switch = False

        # 2. Registrar ação (vigora no próximo tick)
        if action == 1:
            if self._can_switch():
                self._pending_switch = True
                logger.debug(
                    "tick %d: troca solicitada → vigora no tick %d",
                    self.current_tick,
                    self.current_tick + 1,
                )
            else:
                logger.debug(
                    "tick %d: ação de troca ignorada (ticks_in_phase=%d < min=%d)",
                    self.current_tick,
                    self.ticks_in_phase,
                    self._min_green_ticks,
                )

        # 3. Chegadas
        self.veh_ns.add_arrivals(arrivals.get("veh_ns", 0))
        self.ped_l.add_arrivals(arrivals.get("ped_l", 0))
        self.ped_o.add_arrivals(arrivals.get("ped_o", 0))

        # 4. Escoamento conforme fase ativa
        self._process_flow()

        # 5. Incrementar espera dos que ficaram
        self.veh_ns.increment_waits()
        self.ped_l.increment_waits()
        self.ped_o.increment_waits()

        # 6. Contadores
        self.ticks_in_phase += 1
        self.current_tick += 1

        return self.get_state()

    def _process_flow(self) -> None:
        """Drena entidades das filas que têm verde, respeitando a capacidade de saturação."""
        if self.current_phase == Phase.A:
            flow = (
                self._cfg["saturation_flow_veh_per_lane_per_tick"]
                * self._cfg["num_car_lanes"]
            )
            self.veh_ns.drain(flow)
        else:
            flow_per_side = self._cfg["saturation_flow_ped_per_side_per_tick"]
            self.ped_l.drain(flow_per_side)
            self.ped_o.drain(flow_per_side)

    def _execute_phase_switch(self) -> None:
        """Inverte a fase e zera o contador de ticks na fase."""
        old = self.current_phase
        self.current_phase = Phase.B if self.current_phase == Phase.A else Phase.A
        self.ticks_in_phase = 0
        logger.debug(
            "tick %d: fase %s → %s", self.current_tick, old.value, self.current_phase.value
        )

    # ------------------------------------------------------------------
    # Reset e estado
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """
        Reinicia o cruzamento para o estado inicial (fase A, filas vazias).

        Retorna: estado inicial
        """
        self.veh_ns = TrafficQueue("veh_ns")
        self.ped_l = TrafficQueue("ped_l")
        self.ped_o = TrafficQueue("ped_o")
        self.current_phase = Phase.A
        self.ticks_in_phase = 0
        self.current_tick = 0
        self._pending_switch = False
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """
        Retorna o estado completo do cruzamento como dicionário serializável.

        Retorna: dict com tick, phase, ticks_in_phase, pending_switch e
                 métricas das três filas (size, max_wait_ticks, total_wait_ticks)
        """
        return {
            "tick": self.current_tick,
            "phase": self.current_phase.value,
            "ticks_in_phase": self.ticks_in_phase,
            "pending_switch": self._pending_switch,
            "veh_ns": {
                "size": self.veh_ns.size,
                "max_wait_ticks": self.veh_ns.max_wait_ticks,
                "total_wait_ticks": self.veh_ns.total_wait_ticks,
            },
            "ped_l": {
                "size": self.ped_l.size,
                "max_wait_ticks": self.ped_l.max_wait_ticks,
                "total_wait_ticks": self.ped_l.total_wait_ticks,
            },
            "ped_o": {
                "size": self.ped_o.size,
                "max_wait_ticks": self.ped_o.max_wait_ticks,
                "total_wait_ticks": self.ped_o.total_wait_ticks,
            },
        }
