"""Orquestra o avanço de ticks, controla velocidade/pausa, conecta engine a renderer e plots."""

from __future__ import annotations

from typing import Any, Callable

from src.simulation.crossing import Crossing


class SimulationLoop:
    """
    Camada de tempo entre a engine de simulação e a visualização.

    Avança a simulação em ticks proporcionalmente ao tempo real × multiplicador
    de velocidade, sem que Crossing conheça o renderer ou vice-versa.
    """

    _YELLOW_GREEN_SECONDS: float = 2.0  # dentro do tick com troca pendente, verde dura 2s

    def __init__(
        self,
        crossing: Crossing,
        controller: Any,
        arrivals_fn: Callable[[int], dict[str, int]],
        tick_seconds: float = 5.0,
        on_tick: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """
        Parâmetros:
        - crossing: instância da engine de simulação
        - controller: objeto com método decide(state) -> int
        - arrivals_fn: fn(tick) -> {'veh_ns': int, 'ped_l': int, 'ped_o': int}
        - tick_seconds: duração de um tick em segundos simulados (vem do config)
        - on_tick: callback chamado após cada tick completo com o novo estado
        """
        self._crossing = crossing
        self._controller = controller
        self._arrivals_fn = arrivals_fn
        self._tick_seconds = tick_seconds
        self._on_tick = on_tick
        self.speed: float = 1.0
        self.paused: bool = False
        self._time_in_tick: float = 0.0
        self.state: dict[str, Any] = crossing.reset()
        self.last_arrivals: dict[str, int] = {"veh_ns": 0, "ped_l": 0, "ped_o": 0}
        self.last_drains: dict[str, int]   = {"veh_ns": 0, "ped_l": 0, "ped_o": 0}

    # ── Loop principal ───────────────────────────────────────────────────────

    def update(self, dt_real: float) -> None:
        """
        Avança a simulação proporcionalmente ao tempo real decorrido.

        Parâmetros:
        - dt_real: segundos reais desde o último update (tipicamente 1/60 a 60 fps)
        """
        if self.paused:
            return
        self._time_in_tick += dt_real * self.speed
        while self._time_in_tick >= self._tick_seconds:
            self._time_in_tick -= self._tick_seconds
            self._advance_tick()

    def _advance_tick(self) -> None:
        arrivals = self._arrivals_fn(self.state["tick"])
        prev_veh = self.state["veh_ns"]["size"]
        prev_pl  = self.state["ped_l"]["size"]
        prev_po  = self.state["ped_o"]["size"]
        action   = self._controller.decide(self.state)
        self.state = self._crossing.step(arrivals, action)
        self.last_arrivals = arrivals
        self.last_drains = {
            "veh_ns": max(0, prev_veh + arrivals["veh_ns"] - self.state["veh_ns"]["size"]),
            "ped_l":  max(0, prev_pl  + arrivals["ped_l"]  - self.state["ped_l"]["size"]),
            "ped_o":  max(0, prev_po  + arrivals["ped_o"]  - self.state["ped_o"]["size"]),
        }
        if self._on_tick:
            self._on_tick(self.state)

    # ── Controles ────────────────────────────────────────────────────────────

    def set_speed(self, multiplier: float) -> None:
        """
        Define o multiplicador de velocidade e despausa automaticamente se > 0.

        Parâmetros:
        - multiplier: >0 acelera a simulação; 0 pausa
        """
        self.speed = multiplier
        if multiplier > 0:
            self.paused = False

    def toggle_pause(self) -> None:
        """Alterna entre pausado e rodando."""
        self.paused = not self.paused

    def reset(self) -> None:
        """Reinicia a simulação do zero (filas vazias, tick 0, fase A)."""
        self._time_in_tick = 0.0
        self.state = self._crossing.reset()
        self.last_arrivals = {"veh_ns": 0, "ped_l": 0, "ped_o": 0}
        self.last_drains   = {"veh_ns": 0, "ped_l": 0, "ped_o": 0}

    # ── Propriedades de estado visual ────────────────────────────────────────

    @property
    def time_in_tick_seconds(self) -> float:
        """Segundos simulados decorridos dentro do tick atual (0 .. tick_seconds)."""
        return self._time_in_tick

    @property
    def is_yellow(self) -> bool:
        """
        True quando o semáforo da fase atual deve exibir amarelo.

        Ocorre nos últimos (tick_seconds - 2s) de um tick em que uma troca foi
        agendada. O amarelo é puramente estético: o tráfego continua passando.
        """
        return (
            bool(self.state.get("pending_switch"))
            and self._time_in_tick >= self._YELLOW_GREEN_SECONDS
        )
