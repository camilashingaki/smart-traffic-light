"""Renderizador visual do cruzamento em vista top-down com Pygame."""

from __future__ import annotations

import random
from typing import Any

import pygame

from src.visualization.easing import (
    clamp01,
    ease_in_out_quad,
    ease_in_quad,
    ease_out_cubic,
    lerp,
)
from src.visualization.effects import build_panel, build_radial_glow, build_soft_shadow
from src.visualization.fonts import load_ui_font
from src.visualization.sprites import (
    CAR_SPRITE_PAD,
    PED_SPRITE_SIZE,
    build_car_sprite,
    build_ped_sprite,
)

# ── Paleta ──────────────────────────────────────────────────────────────────
_BG               = (30,  30,  46)
_ROAD             = (55,  55,  55)
_ROAD_EDGE        = (90,  90,  90)
_ROAD_EDGE_LINE   = (212, 212, 218)   # linha branca contínua nas margens
_LANE_DASH        = (210, 210, 215)
_LANE_DASH_DIM    = (150, 150, 160)   # extremo escuro do degradê do tracejado
_ZEBRA_W          = (230, 230, 230)
_ZEBRA_D          = (72,  72,  72)
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
_PED_COLS         = [                 # paleta de roupas dos pedestres
    ( 90, 195, 255),
    (255, 170,  95),
    (150, 230, 140),
    (245, 140, 185),
    (200, 170, 255),
]
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
_BTN_ACTIVE_EDGE  = (130, 185, 255)   # borda inferior do botão ativo
_SEPARATOR        = (50,  50,  70)
_HOUSING          = (25,  25,  25)
_HOUSING_EDGE     = (62,  62,  62)

HUD_H  = 40
CTRL_H = 60

# ── Animações (milissegundos REAIS — independem da velocidade da simulação) ──
_CAR_ARRIVE_MS   = 400    # carro novo desliza de fora da tela até o slot
_CAR_DRAIN_MS    = 400    # carro da frente acelera pela zebra e some
_PED_ARRIVE_MS   = 500    # pedestre novo desliza do lado externo da via
_PED_DRAIN_MS    = 600    # pedestre atravessa a zebra para o lado oposto
_PED_ARRIVE_PX   = 30     # deslocamento horizontal da chegada de pedestre
_BTN_HOVER_MS    = 150    # transição idle↔hover dos botões
_FLASH_FADE_MS   = 300    # fade-out final dos indicadores +N / -N
_FLASH_RISE_PX   = 4      # subida vertical do indicador ao longo da vida
_MAX_GHOSTS      = 10     # teto de animações de saída simultâneas (50x não acumula)
_GLOW_RADIUS     = 25     # raio do halo das lâmpadas acesas

# ── Geometria dos carros ─────────────────────────────────────────────────────
_CAR_W         = 26       # largura do carro (transversal à via)
_CAR_LEN       = 30       # comprimento do carro (longitudinal)
_CAR_GAP       = 5        # espaço entre carros na fila
_MAX_CARS_ITER = 40       # teto de slots percorridos por frame

# ── Pedestres ────────────────────────────────────────────────────────────────
_PED_COLS_GRID = 3        # colunas do cluster grid
_PED_SPACING   = 18       # passo do cluster grid em px
_PED_MAX_VIS   = 18       # máximo de pedestres visíveis por cluster


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
        self._fn_sm = load_ui_font(14)
        self._fn_md = load_ui_font(17)
        self._fn_lg = load_ui_font(22, bold=True)
        self._fn_xl = load_ui_font(38, bold=True)

        self._buttons: dict[str, pygame.Rect] = {}
        self._build_buttons()

        self._flashes: dict[str, dict] = {}   # slot → flash; slot = "{queue}_{sign}"
        self._last_tick_seen: int = -1

        # ── Caches estáticos de renderização (construídos uma única vez) ─────
        road_w = self._road_r - self._road_l
        self._asphalt = self._build_asphalt(road_w, self._cx_h)
        self._zebra_surf = self._build_zebra_surface(road_w, self._zebra_h)
        self._dash_surf = self._build_dash_surface(3, 16)
        self._hud_panel = build_panel(self.w, HUD_H, (*_HUD_BG, 228))
        self._ctrl_panel = build_panel(self.w, CTRL_H, (*_CTRL_BG, 228))
        self._glows: dict[tuple[int, int, int], pygame.Surface] = {
            col: build_radial_glow(_GLOW_RADIUS, col)
            for col in (_GREEN, _YELLOW_C, _RED_C)
        }
        self._shadow_cache: dict[tuple[int, int], pygame.Surface] = {}
        # sprites de carro por (índice de cor, frente para baixo?)
        self._car_sprites: dict[tuple[int, bool], pygame.Surface] = {
            (ci, fd): build_car_sprite(_CAR_W, _CAR_LEN, col, fd)
            for ci, col in enumerate(_CAR_COLS)
            for fd in (True, False)
        }
        self._ped_sprites: list[pygame.Surface] = [
            build_ped_sprite(col) for col in _PED_COLS
        ]

        # ── Estado das animações (tudo em ms reais via get_ticks) ────────────
        self._last_anim_tick: int | None = None
        self._last_car_count: int | None = None
        self._car_arrivals: dict[int, int] = {}   # slot da fila → início (ms)
        self._car_drains: list[dict] = []          # carros-fantasma escoando
        self._ped_state: dict[str, dict] = {
            "ped_l": {"last": None, "arrivals": {}},
            "ped_o": {"last": None, "arrivals": {}},
        }
        self._ped_drains: list[dict] = []          # pedestres atravessando

        # mistura idle↔hover por botão (0 = idle, 1 = hover) e último timestamp
        self._btn_hover_mix: dict[str, float] = {lbl: 0.0 for lbl in self._buttons}
        self._last_ctrl_ms: int = pygame.time.get_ticks()

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
        # Reset/troca de cenário (tick regressivo) zera animações em curso
        tick = state["tick"]
        if self._last_anim_tick is not None and tick < self._last_anim_tick:
            self._reset_animations()
        self._last_anim_tick = tick

        # Detecta novo tick e cria flashes; lida com reset (tick regressivo)
        if arrivals is not None and drains is not None:
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

    def _reset_animations(self) -> None:
        """Descarta animações em curso (chamado quando o cenário reinicia)."""
        self._last_car_count = None
        self._car_arrivals = {}
        self._car_drains = []
        for ps in self._ped_state.values():
            ps["last"] = None
            ps["arrivals"] = {}
        self._ped_drains = []

    # ── Estrada e zebra ──────────────────────────────────────────────────────

    @staticmethod
    def _build_asphalt(width: int, height: int) -> pygame.Surface:
        """Textura procedural do asfalto: ruído puntual estático com seed fixa."""
        rng = random.Random(0xA5FA17)
        surf = pygame.Surface((width, height))
        surf.fill(_ROAD)
        # ~1 grão a cada 14 px² dá granulado visível sem virar estática de TV
        for _ in range((width * height) // 14):
            x, y = rng.randrange(width), rng.randrange(height)
            delta = rng.randint(-10, 12)
            shade = tuple(min(255, max(0, c + delta)) for c in _ROAD)
            surf.set_at((x, y), shade)
        return surf

    @staticmethod
    def _build_zebra_surface(width: int, height: int) -> pygame.Surface:
        """Zebra pré-renderizada com degradê vertical leve (profundidade)."""
        surf = pygame.Surface((width, height))
        stripe_w = 14
        for row in range(height):
            # topo 100% → base 82% de brilho, queda linear
            factor = 1.0 - 0.18 * (row / height)
            bright = tuple(int(c * factor) for c in _ZEBRA_W)
            dark   = tuple(int(c * factor) for c in _ZEBRA_D)
            x, i = 0, 0
            while x < width:
                w = min(stripe_w, width - x)
                surf.fill(bright if i % 2 == 0 else dark, pygame.Rect(x, row, w, 1))
                x += stripe_w
                i += 1
        return surf

    @staticmethod
    def _build_dash_surface(width: int, height: int) -> pygame.Surface:
        """Traço da faixa central com degradê vertical claro→escuro."""
        surf = pygame.Surface((width, height))
        for row in range(height):
            t = row / max(1, height - 1)
            shade = tuple(
                int(lerp(a, b, t)) for a, b in zip(_LANE_DASH, _LANE_DASH_DIM)
            )
            surf.fill(shade, pygame.Rect(0, row, width, 1))
        return surf

    def _draw_road(self, surface: pygame.Surface) -> None:
        surface.blit(self._asphalt, (self._road_l, self._cx_y))

        # acostamento escuro + linha branca contínua fina nas duas margens
        for bx in (self._road_l - 3, self._road_r + 1):
            pygame.draw.rect(surface, _ROAD_EDGE,
                             pygame.Rect(bx, self._cx_y, 2, self._cx_h))
        for lx in (self._road_l, self._road_r - 1):
            pygame.draw.rect(surface, _ROAD_EDGE_LINE,
                             pygame.Rect(lx, self._cx_y, 1, self._cx_h))

        # faixa central tracejada com degradê
        dash_h, gap = self._dash_surf.get_height(), 10
        y = self._cx_y
        end = self._cx_y + self._cx_h
        while y < end:
            h = min(dash_h, end - y)
            surface.blit(self._dash_surf, (self._road_cx - 1, y),
                         pygame.Rect(0, 0, self._dash_surf.get_width(), h))
            y += dash_h + gap

    def _draw_zebra(self, surface: pygame.Surface) -> None:
        surface.blit(self._zebra_surf, (self._road_l, self._zebra_top))

    # ── Veículos ─────────────────────────────────────────────────────────────

    def _car_slot(self, idx: int) -> tuple[int, int, bool]:
        """Posição (centro x, topo y) e orientação do slot ``idx`` da fila.

        Slots alternam faixas: índices pares na faixa oeste (frente para
        baixo, sentido norte→sul) e ímpares na leste (frente para cima);
        a linha dentro da faixa é ``idx // 2``, com 0 encostado na zebra.
        Equivale à distribuição original n_oeste = ceil(n/2), n_leste = n//2.
        """
        west = idx % 2 == 0
        row = idx // 2
        lane_cx = (
            (self._road_l + self._road_cx) // 2 if west
            else (self._road_cx + self._road_r) // 2
        )
        y = self._zebra_top - (row + 1) * (_CAR_LEN + _CAR_GAP)
        return lane_cx, y, west

    def _blit_car(
        self, surface: pygame.Surface, cx: int, top_y: float,
        color_idx: int, facing_down: bool, alpha: int = 255,
    ) -> None:
        """Blita o sprite cacheado do carro centrado em x, com topo em ``top_y``."""
        sprite = self._car_sprites[(color_idx % len(_CAR_COLS), facing_down)]
        if alpha < 255:
            sprite.set_alpha(alpha)
        surface.blit(
            sprite,
            (cx - _CAR_W // 2 - CAR_SPRITE_PAD, int(top_y) - CAR_SPRITE_PAD),
        )
        if alpha < 255:
            sprite.set_alpha(None)   # restaura o sprite compartilhado do cache

    def _update_car_animations(self, n: int, now: int) -> None:
        """Compara a fila atual com o frame anterior e cria/ajusta animações."""
        # limpa chegadas que nunca chegaram a ser desenhadas (slots ocultos)
        self._car_arrivals = {
            i: t0 for i, t0 in self._car_arrivals.items()
            if now - t0 < _CAR_ARRIVE_MS
        }

        prev = self._last_car_count
        self._last_car_count = n
        if prev is None or n == prev:
            return

        if n > prev:
            # chegada: cada slot novo ganha uma animação; em velocidade alta,
            # reatribuir o slot apenas substitui a animação anterior (não acumula)
            for idx in range(prev, min(n, _MAX_CARS_ITER)):
                self._car_arrivals[idx] = now
        else:
            # escoamento: os k carros da frente viram fantasmas que cruzam a zebra
            k = prev - n
            for j in range(min(k, _MAX_GHOSTS)):
                cx, y, _west = self._car_slot(j)
                if y < self._cx_y + 2:
                    continue
                self._car_drains.append(
                    {"start": now, "cx": cx, "y0": y, "color": (j // 2)}
                )
            del self._car_drains[:-_MAX_GHOSTS]
            # a fila anda k posições: animações de chegada acompanham o deslocamento
            self._car_arrivals = {
                i - k: t0 for i, t0 in self._car_arrivals.items() if i - k >= 0
            }

    def _draw_cars(self, surface: pygame.Surface, state: dict[str, Any]) -> None:
        now = pygame.time.get_ticks()
        n = state["veh_ns"]["size"]
        self._update_car_animations(n, now)

        # carros novos nascem logo acima da área visível (por baixo do HUD)
        spawn_y = self._cx_y - _CAR_LEN

        for idx in range(min(n, _MAX_CARS_ITER)):
            cx, y, west = self._car_slot(idx)
            if y < self._cx_y + 2:
                continue
            start = self._car_arrivals.get(idx)
            if start is not None:
                t = (now - start) / _CAR_ARRIVE_MS
                if t >= 1.0:
                    del self._car_arrivals[idx]
                else:
                    # desliza do spawn até o slot com ease-out (rápido → suave)
                    y = lerp(spawn_y, y, ease_out_cubic(t))
            self._blit_car(surface, cx, y, idx // 2, west)

        self._draw_car_drains(surface, now)

    def _draw_car_drains(self, surface: pygame.Surface, now: int) -> None:
        """Anima carros-fantasma acelerando pela zebra até sumirem ao sul."""
        end_y = self._zebra_bot + _CAR_LEN * 2
        alive: list[dict] = []
        for ghost in self._car_drains:
            t = (now - ghost["start"]) / _CAR_DRAIN_MS
            if t >= 1.0:
                continue
            # arrancada: ease-in (parte do repouso e acelera pela travessia)
            y = lerp(ghost["y0"], end_y, ease_in_quad(t))
            # some gradualmente nos últimos 40% do trajeto
            alpha = 255 if t < 0.6 else int(255 * (1.0 - t) / 0.4)
            # todo escoamento cruza para o sul, então a frente aponta para baixo
            self._blit_car(surface, ghost["cx"], y, ghost["color"], True, alpha)
            alive.append(ghost)
        self._car_drains = alive

    # ── Pedestres ────────────────────────────────────────────────────────────

    def _ped_slot(self, side: str, idx: int, visible: int) -> tuple[int, int]:
        """Posição final (centro) do pedestre ``idx`` no cluster grid do lado.

        Replica a grade original: 3 colunas crescendo para fora da via,
        linhas centradas verticalmente na zebra.
        """
        grow_right = side == "ped_l"
        start_x = self._ped_e_x if grow_right else self._ped_w_x
        total_rows = max(1, (visible + _PED_COLS_GRID - 1) // _PED_COLS_GRID)
        col = idx % _PED_COLS_GRID
        row = idx // _PED_COLS_GRID
        dx = col * _PED_SPACING * (1 if grow_right else -1)
        dy = row * _PED_SPACING - (total_rows - 1) * _PED_SPACING // 2
        return start_x + dx, self._zebra_cy + dy

    def _blit_ped(
        self, surface: pygame.Surface, x: float, y: float,
        color_idx: int, alpha: int = 255,
    ) -> None:
        """Blita o sprite cacheado do pedestre centrado em (x, y)."""
        sprite = self._ped_sprites[color_idx % len(_PED_COLS)]
        if alpha < 255:
            sprite.set_alpha(alpha)
        half = PED_SPRITE_SIZE // 2
        surface.blit(sprite, (int(x) - half, int(y) - half))
        if alpha < 255:
            sprite.set_alpha(None)

    def _update_ped_animations(self, side: str, n: int, now: int) -> None:
        """Versão para pedestres de ``_update_car_animations`` (por lado)."""
        ps = self._ped_state[side]
        ps["arrivals"] = {
            i: t0 for i, t0 in ps["arrivals"].items()
            if now - t0 < _PED_ARRIVE_MS
        }

        prev = ps["last"]
        ps["last"] = n
        if prev is None or n == prev:
            return

        if n > prev:
            for idx in range(prev, min(n, _PED_MAX_VIS)):
                ps["arrivals"][idx] = now
        else:
            k = prev - n
            # quem sai atravessa a via até além da margem oposta
            is_east = side == "ped_l"
            exit_x = self._road_l - 70 if is_east else self._road_r + 70
            for j in range(min(k, min(prev, _PED_MAX_VIS), _MAX_GHOSTS)):
                x0, y0 = self._ped_slot(side, j, min(prev, _PED_MAX_VIS))
                self._ped_drains.append({
                    "start": now,
                    "x0": x0, "y0": y0,
                    "x1": exit_x,
                    # cada fantasma usa uma "pista" da zebra para não empilhar
                    "y1": self._zebra_cy + (j % 3 - 1) * 10,
                    "color": j,
                })
            del self._ped_drains[:-_MAX_GHOSTS]
            ps["arrivals"] = {
                i - k: t0 for i, t0 in ps["arrivals"].items() if i - k >= 0
            }

    def _draw_peds(self, surface: pygame.Surface, state: dict[str, Any]) -> None:
        now = pygame.time.get_ticks()

        for side in ("ped_o", "ped_l"):
            n = state[side]["size"]
            self._update_ped_animations(side, n, now)
            ps = self._ped_state[side]
            visible = min(n, _PED_MAX_VIS)
            # chegada desliza do lado externo da via para dentro
            arrive_dx = _PED_ARRIVE_PX if side == "ped_l" else -_PED_ARRIVE_PX

            for idx in range(visible):
                x, y = self._ped_slot(side, idx, visible)
                start = ps["arrivals"].get(idx)
                if start is not None:
                    t = (now - start) / _PED_ARRIVE_MS
                    if t >= 1.0:
                        del ps["arrivals"][idx]
                    else:
                        # nasce deslocado para fora e desliza até o slot (ease-out)
                        x = lerp(x + arrive_dx, x, ease_out_cubic(t))
                self._blit_ped(surface, x, y, idx)

        self._draw_ped_drains(surface, now)

    def _draw_ped_drains(self, surface: pygame.Surface, now: int) -> None:
        """Anima pedestres-fantasma atravessando a zebra para o lado oposto."""
        alive: list[dict] = []
        for ghost in self._ped_drains:
            t = (now - ghost["start"]) / _PED_DRAIN_MS
            if t >= 1.0:
                continue
            # travessia: acelera, cruza e desacelera (ease-in-out)
            p = ease_in_out_quad(t)
            x = lerp(ghost["x0"], ghost["x1"], p)
            y = lerp(ghost["y0"], ghost["y1"], min(1.0, t * 2.5))
            alpha = 255 if t < 0.75 else int(255 * (1.0 - t) / 0.25)
            self._blit_ped(surface, x, y, ghost["color"], alpha)
            alive.append(ghost)
        self._ped_drains = alive

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
        # contador de carros ocupa cx_y+8 .. cx_y+30 → flashes ficam abaixo (cx_y+36+)
        # contadores de ped ocupam zebra_cy-70 .. zebra_cy-48 →
        #   +N fica 40 px acima (zebra_cy-110) e -N fica 8 px abaixo (zebra_cy-40)
        specs = [
            ("veh_ns", "+", arrivals, self._road_cx + 90, self._cx_y + 36,       _FLASH_ARRIVAL, False),
            ("veh_ns", "-", drains,   self._road_cx + 90, self._cx_y + 58,       _FLASH_DRAIN,   False),
            ("ped_l",  "+", arrivals, self._road_r + 100,  self._zebra_cy - 110, _FLASH_ARRIVAL, False),
            ("ped_l",  "-", drains,   self._road_r + 100,  self._zebra_cy - 40,  _FLASH_DRAIN,   False),
            ("ped_o",  "+", arrivals, self._road_l - 14,   self._zebra_cy - 110, _FLASH_ARRIVAL, True),
            ("ped_o",  "-", drains,   self._road_l - 14,   self._zebra_cy - 40,  _FLASH_DRAIN,   True),
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
        """Renderiza indicadores ativos (com fade-out e subida) e remove expirados."""
        now = pygame.time.get_ticks()
        self._flashes = {
            slot: fl for slot, fl in self._flashes.items() if fl["expire_at"] > now
        }
        for fl in self._flashes.values():
            remaining = fl["expire_at"] - now
            elapsed = _FLASH_MS - remaining
            # sobe linearmente 4 px ao longo da vida do indicador
            dy = -_FLASH_RISE_PX * clamp01(elapsed / _FLASH_MS)
            # alpha pleno até faltarem 300 ms; depois decai linearmente até 0
            alpha = (
                255 if remaining >= _FLASH_FADE_MS
                else int(255 * remaining / _FLASH_FADE_MS)
            )
            txt = self._fn_md.render(fl["text"], True, fl["color"])
            txt.set_alpha(alpha)
            x = fl["x"] - (txt.get_width() if fl["right_align"] else 0)
            surface.blit(txt, (x, fl["y"] + dy))

    # ── Semáforos ────────────────────────────────────────────────────────────

    def _draw_housing(self, surface: pygame.Surface, box: pygame.Rect) -> None:
        """Carcaça de semáforo: sombra leve, corpo arredondado e borda."""
        key = (box.w, box.h)
        if key not in self._shadow_cache:
            self._shadow_cache[key] = build_soft_shadow(box.w, box.h)
        surface.blit(self._shadow_cache[key], (box.x + 3, box.y + 3))
        pygame.draw.rect(surface, _HOUSING, box, border_radius=8)
        pygame.draw.rect(surface, _HOUSING_EDGE, box, 1, border_radius=8)

    def _draw_lamp(
        self, surface: pygame.Surface, center: tuple[int, int],
        radius: int, color: tuple[int, int, int], lit: bool,
        dim_color: tuple[int, int, int],
    ) -> None:
        """Lâmpada de semáforo; quando acesa ganha halo radial aditivo."""
        if lit:
            glow = self._glows[color]
            surface.blit(
                glow,
                (center[0] - _GLOW_RADIUS, center[1] - _GLOW_RADIUS),
                special_flags=pygame.BLEND_RGB_ADD,
            )
        pygame.draw.circle(surface, color if lit else dim_color, center, radius)

    def _draw_traffic_light(
        self, surface: pygame.Surface, state: dict[str, Any], is_yellow: bool
    ) -> None:
        """Semáforo de carros ao norte-leste da travessia — três luzes."""
        phase = state["phase"]
        bw, bh = 28, 74
        box = pygame.Rect(self._tl_x, self._tl_y, bw, bh)
        self._draw_housing(surface, box)

        cx  = self._tl_x + bw // 2
        ys  = [self._tl_y + 14, self._tl_y + 37, self._tl_y + 60]  # R, Y, G

        red_on    = phase == "B" and not is_yellow
        yellow_on = is_yellow
        green_on  = phase == "A" and not is_yellow

        self._draw_lamp(surface, (cx, ys[0]), 9, _RED_C,    red_on,    _RED_DIM)
        self._draw_lamp(surface, (cx, ys[1]), 9, _YELLOW_C, yellow_on, _YELLOW_DIM)
        self._draw_lamp(surface, (cx, ys[2]), 9, _GREEN,    green_on,  _GREEN_DIM)

    def _draw_ped_signals(
        self, surface: pygame.Surface, state: dict[str, Any], is_yellow: bool
    ) -> None:
        """Sinais de pedestres nos dois lados da zebra."""
        ped_green = state["phase"] == "B" and not is_yellow
        color = _GREEN if ped_green else _RED_C
        dim = _GREEN_DIM if ped_green else _RED_DIM
        bw, bh = 24, 36

        for x in (self._psig_e_x, self._psig_w_x):
            box = pygame.Rect(x, self._psig_y, bw, bh)
            self._draw_housing(surface, box)
            self._draw_lamp(
                surface, (x + bw // 2, self._psig_y + bh // 2),
                10, color, True, dim,
            )

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
        surface.blit(self._hud_panel, (0, 0))

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
        surface.blit(self._ctrl_panel, (0, self._ctrl_y))

        hint = self._fn_sm.render(
            "ESPAÇO: pausa  |  1-4: velocidade  |  R: reiniciar", True, (100, 100, 130)
        )
        surface.blit(hint, (self.w - hint.get_width() - 10,
                             self._ctrl_y + (CTRL_H - hint.get_height()) // 2))

        # avanço da transição hover deste frame (tempo real → suave em qualquer FPS)
        now = pygame.time.get_ticks()
        step = (now - self._last_ctrl_ms) / _BTN_HOVER_MS
        self._last_ctrl_ms = now

        for label, rect in self._buttons.items():
            target = 1.0 if rect.collidepoint(mouse_pos) else 0.0
            mix = self._btn_hover_mix[label]
            mix = min(mix + step, target) if target > mix else max(mix - step, target)
            self._btn_hover_mix[label] = mix

            is_active = (label == "PAUSE" and is_paused) or label == active_speed_label
            if label == "PAUSE" and is_paused:
                color = _BTN_PAUSE_ACTIVE
            elif label == active_speed_label:
                color = _BTN_ACTIVE
            else:
                # lerp idle↔hover guiado pela mistura temporal de 150 ms
                color = tuple(
                    int(lerp(a, b, mix)) for a, b in zip(_BTN_IDLE, _BTN_HOVER)
                )
            pygame.draw.rect(surface, color, rect, border_radius=8)
            if is_active:
                pygame.draw.rect(
                    surface, _BTN_ACTIVE_EDGE,
                    pygame.Rect(rect.x + 4, rect.bottom - 4, rect.w - 8, 3),
                    border_radius=2,
                )
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
