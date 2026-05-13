"""Renderizador visual do cruzamento em vista top-down com Pygame."""

from __future__ import annotations

from typing import Any

import pygame

# ── Paleta ──────────────────────────────────────────────────────────────────
_BG               = (30,  30,  46)
_ROAD             = (55,  55,  55)
_ROAD_EDGE        = (90,  90,  90)
_LANE_DASH        = (200, 200, 200)
_ZEBRA_W          = (225, 225, 225)
_ZEBRA_D          = (70,  70,  70)
_GREEN            = (40,  200,  60)
_YELLOW_C         = (255, 190,  30)
_RED_C            = (210,  30,  30)
_GREEN_DIM        = (12,   55,  20)
_YELLOW_DIM       = (55,   45,  10)
_RED_DIM          = (55,   12,  12)
_CAR_COLS         = [
    (210,  80,  80),
    ( 80, 150, 220),
    ( 80, 200, 110),
    (200, 190,  60),
    (190,  80, 200),
]
_PED_COL          = (90,  195, 255)
_TEXT             = (215, 215, 215)
_TEXT_WARN        = (255,  75,  75)
_TEXT_YELLOW      = (255, 200,  50)
_COUNTER_COL      = (160, 210, 255)   # azul-gelo para contadores de fila
_FLASH_ARRIVAL    = (255, 235, 60)    # amarelo claro — chegadas
_FLASH_DRAIN      = (70,  220, 220)   # ciano claro   — escoamentos
_FLASH_MS         = 1000              # duração dos indicadores em milissegundos reais
_HUD_BG           = (20,  20,  35)
_CTRL_BG          = (22,  22,  38)
_BTN_ACTIVE       = (55,  100, 185)
_BTN_PAUSE_ACTIVE = (175,  65,  45)
_BTN_HOVER        = (75,  120, 210)
_BTN_IDLE         = (45,  45,  65)
_SEPARATOR        = (50,  50,  70)

HUD_H  = 40
CTRL_H = 60


class CrossingRenderer:
    """
    Renderiza todo o painel esquerdo da janela: cruzamento top-down, HUD,
    métricas de espera, semáforos (verde/amarelo/vermelho) e barra de controles.

    Recebe estado como dicionário puro — não conhece Crossing nem SimulationLoop.
    """

    def __init__(
        self, cfg: dict[str, Any], panel_width: int, panel_height: int
    ) -> None:
        """
        Parâmetros:
        - cfg: conteúdo raiz do config.yaml
        - panel_width / panel_height: dimensões do painel em pixels
        """
        self._cfg = cfg
        self.w = panel_width
        self.h = panel_height

        self._cx_y  = HUD_H
        self._cx_h  = panel_height - HUD_H - CTRL_H
        self._ctrl_y = panel_height - CTRL_H

        # geometria da estrada
        road_cx = panel_width // 2
        self._lane_w  = 55
        self._road_l  = road_cx - self._lane_w   # 329
        self._road_r  = road_cx + self._lane_w   # 439
        self._road_cx = road_cx                   # 384

        # zebra
        self._zebra_cy  = self._cx_y + self._cx_h // 2  # 350
        self._zebra_h   = 40
        self._zebra_top = self._zebra_cy - self._zebra_h // 2  # 330
        self._zebra_bot = self._zebra_cy + self._zebra_h // 2  # 370

        # semáforo de carros (lado leste da via, ao norte da zebra)
        self._tl_x = self._road_r + 14   # 453
        self._tl_y = self._zebra_top - 82  # 248

        # sinais de pedestres (nos dois lados da zebra)
        self._psig_e_x = self._road_r + 14   # 453
        self._psig_w_x = self._road_l - 38   # 291
        self._psig_y   = self._zebra_cy - 18  # 332

        # posição inicial dos clusters de pedestres (cresce para fora da via)
        self._ped_e_x = self._road_r + 50   # 489
        self._ped_w_x = self._road_l - 46   # 283

        pygame.font.init()
        self._fn_sm = pygame.font.SysFont("monospace", 14)
        self._fn_md = pygame.font.SysFont("monospace", 17)
        self._fn_lg = pygame.font.SysFont("monospace", 22, bold=True)
        self._fn_xl = pygame.font.SysFont("monospace", 38, bold=True)

        self._buttons: dict[str, pygame.Rect] = {}
        self._build_buttons()

        self._flashes: dict[str, dict] = {}   # slot → flash; slot = "{queue}_{sign}"
        self._last_tick_seen: int = -1

    # ── Botões ───────────────────────────────────────────────────────────────

    def _build_buttons(self) -> None:
        labels = ["PAUSE", "1x", "5x", "10x", "50x", "R"]
        bw, bh, gap = 72, 36, 8
        total = len(labels) * bw + (len(labels) - 1) * gap
        x0 = (self.w - total) // 2
        y  = self._ctrl_y + (CTRL_H - bh) // 2
        for i, lbl in enumerate(labels):
            self._buttons[lbl] = pygame.Rect(x0 + i * (bw + gap), y, bw, bh)

    def get_button_rects(self) -> dict[str, pygame.Rect]:
        """Retorna mapa de label → Rect para detecção de cliques no loop externo."""
        return dict(self._buttons)

    # ── Método principal ─────────────────────────────────────────────────────

    def draw(
        self,
        surface: pygame.Surface,
        state: dict[str, Any],
        is_yellow: bool,
        is_paused: bool,
        speed: float,
        active_speed_label: str,
        mouse_pos: tuple[int, int],
        cfg_fixed: dict[str, Any],
        cfg_thresholds: dict[str, Any],
        tick_seconds: float,
        arrivals: dict[str, int] | None = None,
        drains: dict[str, int] | None = None,
    ) -> None:
        """
        Renderiza o painel completo do cruzamento na Surface fornecida.

        Parâmetros:
        - state: retorno de Crossing.get_state()
        - is_yellow: True quando o semáforo deve mostrar amarelo (§3.4)
        - is_paused: True quando a simulação está pausada
        - speed: multiplicador numérico de velocidade atual
        - active_speed_label: label do botão de velocidade ativo ("1x", "5x", ...)
        - mouse_pos: posição do mouse (para hover nos botões)
        - cfg_fixed: seção fixed_time_controller do config
        - cfg_thresholds: seção thresholds do config
        - tick_seconds: duração de um tick em segundos
        - arrivals: chegadas do último tick por fila (para indicadores piscantes)
        - drains: escoamentos do último tick por fila (para indicadores piscantes)
        """
        # Detecta novo tick e cria flashes; lida com reset (tick regressivo)
        if arrivals is not None and drains is not None:
            tick = state["tick"]
            if tick < self._last_tick_seen:
                self._flashes = {}
            elif tick != self._last_tick_seen and self._last_tick_seen >= 0:
                self._spawn_flashes(arrivals, drains)
            self._last_tick_seen = tick

        surface.fill(_BG)
        self._draw_road(surface)
        self._draw_zebra(surface)
        self._draw_cars(surface, state)
        self._draw_peds(surface, state)
        self._draw_queue_counters(surface, state)
        self._draw_flashes(surface)
        self._draw_traffic_light(surface, state, is_yellow)
        self._draw_ped_signals(surface, state, is_yellow)
        self._draw_phase_label(surface, state, is_yellow)
        self._draw_metrics(surface, state, cfg_thresholds, tick_seconds)
        self._draw_hud(surface, state, is_paused, speed, cfg_fixed, tick_seconds)
        self._draw_controls(surface, active_speed_label, is_paused, mouse_pos)
        if is_paused:
            self._draw_pause_overlay(surface)

    # ── Estrada e zebra ──────────────────────────────────────────────────────

    def _draw_road(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(
            surface, _ROAD,
            pygame.Rect(self._road_l, self._cx_y,
                        self._road_r - self._road_l, self._cx_h),
        )
        for bx in (self._road_l - 2, self._road_r):
            pygame.draw.rect(surface, _ROAD_EDGE,
                             pygame.Rect(bx, self._cx_y, 2, self._cx_h))
        dash, gap = 16, 10
        y = self._cx_y
        end = self._cx_y + self._cx_h
        while y < end:
            h = min(dash, end - y)
            pygame.draw.rect(surface, _LANE_DASH,
                             pygame.Rect(self._road_cx - 1, y, 2, h))
            y += dash + gap

    def _draw_zebra(self, surface: pygame.Surface) -> None:
        stripe_w = 14
        x, i = self._road_l, 0
        while x < self._road_r:
            w = min(stripe_w, self._road_r - x)
            color = _ZEBRA_W if i % 2 == 0 else _ZEBRA_D
            pygame.draw.rect(surface, color,
                             pygame.Rect(x, self._zebra_top, w, self._zebra_h))
            x += stripe_w
            i += 1

    # ── Veículos ─────────────────────────────────────────────────────────────

    def _draw_cars(self, surface: pygame.Surface, state: dict[str, Any]) -> None:
        n   = state["veh_ns"]["size"]
        cw  = self._lane_w - 7
        ch  = 22
        gap = 5
        l1x = self._road_l + 4   # faixa oeste
        l2x = self._road_cx + 3  # faixa leste

        n1 = (n + 1) // 2   # distribui entre faixas por alternância (simula "menos cheia")
        n2 = n // 2

        for lane_x, count in ((l1x, n1), (l2x, n2)):
            for i in range(min(count, 14)):
                y = self._zebra_top - (i + 1) * (ch + gap)
                if y < self._cx_y + 2:
                    break
                pygame.draw.rect(
                    surface, _CAR_COLS[i % len(_CAR_COLS)],
                    pygame.Rect(lane_x, y, cw, ch), border_radius=3,
                )

    # ── Pedestres ────────────────────────────────────────────────────────────

    def _draw_peds(self, surface: pygame.Surface, state: dict[str, Any]) -> None:
        r, cols, spacing = 7, 3, 18

        def draw_cluster(n: int, start_x: int, grow_right: bool) -> None:
            vis = min(n, 18)
            total_rows = max(1, (vis + cols - 1) // cols)
            for i in range(vis):
                col = i % cols
                row = i // cols
                dx = col * spacing * (1 if grow_right else -1)
                dy = row * spacing - (total_rows - 1) * spacing // 2
                pygame.draw.circle(
                    surface, _PED_COL,
                    (start_x + dx, self._zebra_cy + dy), r,
                )

        draw_cluster(state["ped_o"]["size"], self._ped_w_x, grow_right=False)
        draw_cluster(state["ped_l"]["size"], self._ped_e_x, grow_right=True)

    # ── Contadores de fila ───────────────────────────────────────────────────

    def _draw_queue_counters(self, surface: pygame.Surface, state: dict[str, Any]) -> None:
        """Exibe contagem numérica de cada fila em texto grande no cruzamento."""
        n_car   = state["veh_ns"]["size"]
        n_ped_e = state["ped_l"]["size"]
        n_ped_o = state["ped_o"]["size"]

        # Carros — centrado na via, topo da área do cruzamento
        t_car = self._fn_lg.render(f"Carros: {n_car}", True, _COUNTER_COL)
        surface.blit(t_car, (self._road_cx - t_car.get_width() // 2, self._cx_y + 8))

        # Pedestres leste — à direita dos círculos (folga de 100 px a partir de _road_r)
        t_pe = self._fn_lg.render(f"Ped L: {n_ped_e}", True, _COUNTER_COL)
        surface.blit(t_pe, (self._road_r + 100, self._zebra_cy - 70))

        # Pedestres oeste — alinhado à direita, encostado à margem oeste da via
        t_po = self._fn_lg.render(f"Ped O: {n_ped_o}", True, _COUNTER_COL)
        surface.blit(t_po, (self._road_l - 14 - t_po.get_width(), self._zebra_cy - 70))

    # ── Indicadores de chegada / escoamento ─────────────────────────────────

    def _spawn_flashes(self, arrivals: dict[str, int], drains: dict[str, int]) -> None:
        """Cria/atualiza indicadores de chegadas (+N) e escoamentos (-N) do tick.

        Cada slot (fila + sinal) é único: se já existe um indicador visível para
        aquele slot, o número é atualizado e o timer reiniciado, garantindo que
        em velocidades altas o valor mais recente sempre apareça por 1s completo.
        """
        expire_at = pygame.time.get_ticks() + _FLASH_MS
        specs = [
            ("veh_ns", "+", arrivals, self._road_cx + 90, self._cx_y + 8,       _FLASH_ARRIVAL, False),
            ("veh_ns", "-", drains,   self._road_cx + 90, self._cx_y + 28,      _FLASH_DRAIN,   False),
            ("ped_l",  "+", arrivals, self._road_r + 100,  self._zebra_cy - 96, _FLASH_ARRIVAL, False),
            ("ped_l",  "-", drains,   self._road_r + 100,  self._zebra_cy - 74, _FLASH_DRAIN,   False),
            ("ped_o",  "+", arrivals, self._road_l - 14,   self._zebra_cy - 96, _FLASH_ARRIVAL, True),
            ("ped_o",  "-", drains,   self._road_l - 14,   self._zebra_cy - 74, _FLASH_DRAIN,   True),
        ]
        for key, sign, data, x, y, color, right_align in specs:
            n = data.get(key, 0)
            if n > 0:
                self._flashes[f"{key}_{sign}"] = {
                    "text": f"{sign}{n}",
                    "x": x, "y": y,
                    "color": color,
                    "right_align": right_align,
                    "expire_at": expire_at,
                }

    def _draw_flashes(self, surface: pygame.Surface) -> None:
        """Renderiza indicadores ativos e remove os expirados."""
        now = pygame.time.get_ticks()
        self._flashes = {
            slot: fl for slot, fl in self._flashes.items() if fl["expire_at"] > now
        }
        for fl in self._flashes.values():
            txt = self._fn_md.render(fl["text"], True, fl["color"])
            x = fl["x"] - (txt.get_width() if fl["right_align"] else 0)
            surface.blit(txt, (x, fl["y"]))

    # ── Semáforos ────────────────────────────────────────────────────────────

    def _draw_traffic_light(
        self, surface: pygame.Surface, state: dict[str, Any], is_yellow: bool
    ) -> None:
        """Semáforo de carros ao norte-leste da travessia — três luzes."""
        phase = state["phase"]
        bw, bh = 28, 74
        box = pygame.Rect(self._tl_x, self._tl_y, bw, bh)
        pygame.draw.rect(surface, (25, 25, 25), box, border_radius=4)
        pygame.draw.rect(surface, (60, 60, 60), box, 1, border_radius=4)

        cx  = self._tl_x + bw // 2
        ys  = [self._tl_y + 14, self._tl_y + 37, self._tl_y + 60]  # R, Y, G

        red_on    = phase == "B" and not is_yellow
        yellow_on = is_yellow
        green_on  = phase == "A" and not is_yellow

        pygame.draw.circle(surface, _RED_C    if red_on    else _RED_DIM,    (cx, ys[0]), 9)
        pygame.draw.circle(surface, _YELLOW_C if yellow_on else _YELLOW_DIM, (cx, ys[1]), 9)
        pygame.draw.circle(surface, _GREEN    if green_on  else _GREEN_DIM,  (cx, ys[2]), 9)

    def _draw_ped_signals(
        self, surface: pygame.Surface, state: dict[str, Any], is_yellow: bool
    ) -> None:
        """Sinais de pedestres nos dois lados da zebra."""
        ped_green = state["phase"] == "B" and not is_yellow
        color = _GREEN if ped_green else _RED_C
        bw, bh = 24, 36

        for x in (self._psig_e_x, self._psig_w_x):
            box = pygame.Rect(x, self._psig_y, bw, bh)
            pygame.draw.rect(surface, (25, 25, 25), box, border_radius=4)
            pygame.draw.rect(surface, (60, 60, 60), box, 1, border_radius=4)
            pygame.draw.circle(surface, color,
                               (x + bw // 2, self._psig_y + bh // 2), 10)

    # ── Rótulo de fase ───────────────────────────────────────────────────────

    def _draw_phase_label(
        self, surface: pygame.Surface, state: dict[str, Any], is_yellow: bool
    ) -> None:
        phase = state["phase"]
        if is_yellow:
            text, color = "AMARELO — TROCANDO FASE", _TEXT_YELLOW
        elif phase == "A":
            text, color = "FASE A — VERDE PARA CARROS", _GREEN
        else:
            text, color = "FASE B — VERDE PARA PEDESTRES", _PED_COL

        lbl = self._fn_md.render(text, True, color)
        surface.blit(lbl, (self._road_cx - lbl.get_width() // 2,
                           self._zebra_cy + 75))

    # ── Métricas de espera ───────────────────────────────────────────────────

    def _draw_metrics(
        self,
        surface: pygame.Surface,
        state: dict[str, Any],
        cfg_thresholds: dict[str, Any],
        tick_seconds: float,
    ) -> None:
        ceil_car = cfg_thresholds["wait_ceiling_cars_seconds"]
        ceil_ped = cfg_thresholds["wait_ceiling_pedestrians_seconds"]
        max_car  = state["veh_ns"]["max_wait_ticks"] * tick_seconds
        max_ped  = max(
            state["ped_l"]["max_wait_ticks"],
            state["ped_o"]["max_wait_ticks"],
        ) * tick_seconds

        x, y = 10, self._cx_y + 10
        for label, val, ceil in [
            (f"Carros:    {max_car:>5.0f}s / {ceil_car}s", max_car, ceil_car),
            (f"Pedestres: {max_ped:>5.0f}s / {ceil_ped}s", max_ped, ceil_ped),
        ]:
            color = _TEXT_WARN if val > ceil else _TEXT
            surface.blit(self._fn_sm.render(label, True, color), (x, y))
            y += 18

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _draw_hud(
        self,
        surface: pygame.Surface,
        state: dict[str, Any],
        is_paused: bool,
        speed: float,
        cfg_fixed: dict[str, Any],
        tick_seconds: float,
    ) -> None:
        pygame.draw.rect(surface, _HUD_BG, pygame.Rect(0, 0, self.w, HUD_H))

        tick  = state["tick"]
        total = tick * tick_seconds
        hh, mm, ss = int(total // 3600) % 24, int((total % 3600) // 60), int(total % 60)

        phase      = state["phase"]
        ticks_in   = state["ticks_in_phase"]
        phase_dur  = cfg_fixed["phase_a_ticks"] if phase == "A" else cfg_fixed["phase_b_ticks"]
        spd_label  = "PAUSADO" if is_paused else f"{speed:.0f}x"

        parts = [
            f"Tick {tick:>5}",
            f"{hh:02d}:{mm:02d}:{ss:02d}",
            f"[{spd_label}]",
            f"Fase {phase}: {ticks_in}/{phase_dur}",
        ]
        x = 12
        for part in parts:
            lbl = self._fn_sm.render(part, True, _TEXT)
            surface.blit(lbl, (x, (HUD_H - lbl.get_height()) // 2))
            x += lbl.get_width() + 28

    # ── Barra de controles ───────────────────────────────────────────────────

    def _draw_controls(
        self,
        surface: pygame.Surface,
        active_speed_label: str,
        is_paused: bool,
        mouse_pos: tuple[int, int],
    ) -> None:
        pygame.draw.rect(surface, _CTRL_BG, pygame.Rect(0, self._ctrl_y, self.w, CTRL_H))

        hint = self._fn_sm.render(
            "ESPAÇO: pausa  |  1-4: velocidade  |  R: reiniciar", True, (100, 100, 130)
        )
        surface.blit(hint, (self.w - hint.get_width() - 10,
                             self._ctrl_y + (CTRL_H - hint.get_height()) // 2))

        for label, rect in self._buttons.items():
            if label == "PAUSE" and is_paused:
                color = _BTN_PAUSE_ACTIVE
            elif label == active_speed_label:
                color = _BTN_ACTIVE
            elif rect.collidepoint(mouse_pos):
                color = _BTN_HOVER
            else:
                color = _BTN_IDLE
            pygame.draw.rect(surface, color, rect, border_radius=6)
            lbl = self._fn_sm.render(label, True, _TEXT)
            surface.blit(lbl, lbl.get_rect(center=rect.center))

    # ── Overlay de pausa ─────────────────────────────────────────────────────

    def _draw_pause_overlay(self, surface: pygame.Surface) -> None:
        overlay = pygame.Surface((self.w, self._cx_h))
        overlay.set_alpha(110)
        overlay.fill((0, 0, 0))
        surface.blit(overlay, (0, self._cx_y))

        lbl = self._fn_xl.render("PAUSADO", True, (255, 255, 255))
        surface.blit(lbl, (
            self.w // 2 - lbl.get_width() // 2,
            self._cx_y + self._cx_h // 2 - lbl.get_height() // 2,
        ))
