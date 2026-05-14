"""Painel de gráficos matplotlib renderizado como Surface pygame via backend Agg."""

from __future__ import annotations

from collections import deque
from typing import Any

import matplotlib
matplotlib.use("Agg")  # deve vir antes de qualquer import pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pygame


class LivePlots:
    """
    Mantém histórico de métricas e re-renderiza 3 gráficos em uma Surface pygame.

    Não abre janela própria — usa backend Agg para renderizar em memória e
    converte o resultado para uma Surface pygame pronta para blit.
    """

    def __init__(
        self,
        panel_width: int,
        panel_height: int,
        history_length: int,
        tick_seconds: float,
        ceil_cars_s: float,
        ceil_peds_s: float,
        dpi: int = 100,
    ) -> None:
        """
        Parâmetros:
        - panel_width / panel_height: dimensões do painel em pixels
        - history_length: número máximo de ticks mantidos no histórico
        - tick_seconds: para converter max_wait_ticks → segundos nos gráficos
        - ceil_cars_s / ceil_peds_s: linhas horizontais de teto no gráfico 2
        - dpi: resolução da figura matplotlib (pixels/inch)
        """
        self._tick_s    = tick_seconds
        self._ceil_car  = ceil_cars_s
        self._ceil_ped  = ceil_peds_s

        fig_w = panel_width  / dpi
        fig_h = panel_height / dpi

        self._fig, self._axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=dpi)
        self._fig.patch.set_facecolor("#1a1a2c")
        self._fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.07, hspace=0.50)
        self._canvas = FigureCanvasAgg(self._fig)
        self._surface: pygame.Surface | None = None

        self._ticks:     deque[int]   = deque(maxlen=history_length)
        self._veh_sz:    deque[int]   = deque(maxlen=history_length)
        self._ped_l_sz:  deque[int]   = deque(maxlen=history_length)
        self._ped_o_sz:  deque[int]   = deque(maxlen=history_length)
        self._max_car_s: deque[float] = deque(maxlen=history_length)
        self._max_ped_s: deque[float] = deque(maxlen=history_length)
        self._phases:    deque[int]   = deque(maxlen=history_length)  # 0=A, 1=B

        self._last_drawn: int | None = None
        self._dirty = False

    # ── Registro ─────────────────────────────────────────────────────────────

    def record(self, state: dict[str, Any]) -> None:
        """Registra o estado de um tick completo no histórico."""
        self._ticks.append(state["tick"])
        self._veh_sz.append(state["veh_ns"]["size"])
        self._ped_l_sz.append(state["ped_l"]["size"])
        self._ped_o_sz.append(state["ped_o"]["size"])
        self._max_car_s.append(state["veh_ns"]["max_wait_ticks"] * self._tick_s)
        self._max_ped_s.append(
            max(state["ped_l"]["max_wait_ticks"],
                state["ped_o"]["max_wait_ticks"]) * self._tick_s
        )
        self._phases.append(0 if state["phase"] == "A" else 1)
        self._dirty = True

    def clear_history(self) -> None:
        """Limpa todo o histórico — chamado no reset da simulação."""
        for q in (self._ticks, self._veh_sz, self._ped_l_sz, self._ped_o_sz,
                  self._max_car_s, self._max_ped_s, self._phases):
            q.clear()
        self._surface = None
        self._last_drawn: int | None = None
        self._dirty = False

    # ── Atualização da Surface ───────────────────────────────────────────────

    def update_surface(self, interval: int) -> None:
        """
        Re-renderiza os gráficos se necessário.

        Sempre renderiza na primeira chamada após dados serem registrados
        (evita tela vazia no início). Nas chamadas seguintes, só renderiza
        quando pelo menos `interval` ticks tiverem passado desde o último draw.

        Parâmetros:
        - interval: número mínimo de ticks entre re-renderizações consecutivas
        """
        if not self._dirty or not self._ticks:
            return
        if self._last_drawn is not None:
            if self._ticks[-1] - self._last_drawn < interval:
                return
        self._last_drawn = self._ticks[-1]
        self._dirty = False
        self._redraw()

    def _redraw(self) -> None:
        ticks = list(self._ticks)
        ax1, ax2, ax3 = self._axes
        _bg   = "#212135"
        _text = "#c8c8c8"

        for ax in self._axes:
            ax.clear()
            ax.set_facecolor(_bg)
            ax.tick_params(colors=_text, labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#404060")

        # Gráfico 1 — tamanho das filas
        ax1.plot(ticks, list(self._veh_sz),   "#e07070", lw=1.3, label="veh_ns")
        ax1.plot(ticks, list(self._ped_l_sz), "#70aaee", lw=1.3, label="ped_l")
        ax1.plot(ticks, list(self._ped_o_sz), "#70dd90", lw=1.3, label="ped_o")
        ax1.set_ylabel("veic/ped", color=_text, fontsize=7)
        ax1.set_title("Tamanho das Filas", color=_text, fontsize=8, pad=2)
        ax1.legend(fontsize=6, facecolor="#1a1a2c", labelcolor=_text,
                   loc="upper left", framealpha=0.6)

        # Gráfico 2 — espera máxima com linhas de teto
        ax2.plot(ticks, list(self._max_car_s), "#e07070", lw=1.3, label="carros")
        ax2.plot(ticks, list(self._max_ped_s), "#70aaee", lw=1.3, label="peds")
        ax2.axhline(self._ceil_car, color="#e07070", ls="--", lw=0.9, alpha=0.55,
                    label=f"teto {self._ceil_car:.0f}s")
        ax2.axhline(self._ceil_ped, color="#70aaee", ls="--", lw=0.9, alpha=0.55,
                    label=f"teto {self._ceil_ped:.0f}s")
        ax2.set_ylabel("espera (s)", color=_text, fontsize=7)
        ax2.set_title("Espera Máxima", color=_text, fontsize=8, pad=2)
        ax2.legend(fontsize=6, facecolor="#1a1a2c", labelcolor=_text,
                   loc="upper left", framealpha=0.6)

        # Gráfico 3 — fase ao longo do tempo (step plot)
        if len(ticks) > 1:
            ax3.step(ticks, list(self._phases), where="post",
                     color="#a0a0ee", lw=1.3)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["A", "B"], color=_text, fontsize=8)
        ax3.set_ylabel("fase", color=_text, fontsize=7)
        ax3.set_xlabel("tick", color=_text, fontsize=7)
        ax3.set_title("Fase ao Longo do Tempo", color=_text, fontsize=8, pad=2)
        ax3.set_ylim(-0.3, 1.3)

        self._canvas.draw()
        raw  = self._canvas.buffer_rgba()
        size = self._canvas.get_width_height()
        self._surface = pygame.image.frombuffer(raw, size, "RGBA").copy()

    # ── Acesso e encerramento ────────────────────────────────────────────────

    def get_surface(self) -> pygame.Surface | None:
        """Retorna a Surface pygame com os gráficos, ou None se ainda não renderizou."""
        return self._surface

    def close(self) -> None:
        """Fecha a figura matplotlib e libera recursos."""
        plt.close(self._fig)
