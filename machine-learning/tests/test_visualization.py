"""Testes headless de renderização para CrossingRenderer, LivePlots e SimulationLoop."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# backend Agg e driver SDL headless devem ser definidos antes de qualquer init
import matplotlib
matplotlib.use("Agg")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation.crossing import Crossing
from src.simulation.controllers import FixedTimeController
from src.simulation.simulation_loop import SimulationLoop
from src.visualization.pygame_renderer import CrossingRenderer
from src.visualization.live_plots import LivePlots
from src.utils.config_loader import load_all_configs


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_all_configs(ROOT / "configs")["config"]


@pytest.fixture(scope="module")
def pygame_init():
    pygame.init()
    pygame.display.set_mode((1, 1))
    yield
    pygame.quit()


@pytest.fixture
def empty_state(cfg) -> dict:
    return Crossing(cfg["simulation"]).reset()


@pytest.fixture
def full_state(cfg) -> dict:
    c = Crossing(cfg["simulation"])
    state = c.reset()
    for _ in range(30):
        state = c.step({"veh_ns": 5, "ped_l": 3, "ped_o": 3}, 0)
    return state


@pytest.fixture
def renderer(cfg, pygame_init) -> CrossingRenderer:
    return CrossingRenderer(cfg, 768, 720)


# ── CrossingRenderer ─────────────────────────────────────────────────────────

class TestCrossingRenderer:
    def _draw(self, renderer, cfg, state, is_yellow=False, is_paused=False):
        surf = pygame.Surface((768, 720))
        renderer.draw(
            surface            = surf,
            state              = state,
            is_yellow          = is_yellow,
            is_paused          = is_paused,
            speed              = 5.0,
            active_speed_label = "5x",
            mouse_pos          = (0, 0),
            cfg_fixed          = cfg["fixed_time_controller"],
            cfg_thresholds     = cfg["thresholds"],
            tick_seconds       = cfg["simulation"]["tick_seconds"],
        )

    def test_draw_empty_queues(self, renderer, cfg, empty_state):
        self._draw(renderer, cfg, empty_state)

    def test_draw_full_queues(self, renderer, cfg, full_state):
        self._draw(renderer, cfg, full_state, is_yellow=True, is_paused=False)

    def test_draw_with_pause_overlay(self, renderer, cfg, empty_state):
        self._draw(renderer, cfg, empty_state, is_paused=True)

    def test_draw_yellow_state(self, renderer, cfg, empty_state):
        state = {**empty_state, "pending_switch": True}
        self._draw(renderer, cfg, state, is_yellow=True)

    def test_button_rects_labels(self, renderer):
        btns = renderer.get_button_rects()
        assert set(btns.keys()) == {"PAUSE", "1x", "5x", "10x", "50x", "R"}

    def test_button_rects_non_overlapping(self, renderer):
        btns = list(renderer.get_button_rects().values())
        for i, r1 in enumerate(btns):
            for r2 in btns[i + 1:]:
                assert not r1.colliderect(r2), f"Botões colidem: {r1} e {r2}"


# ── SimulationLoop ────────────────────────────────────────────────────────────

class TestSimulationLoop:
    def _make_loop(self, cfg) -> SimulationLoop:
        crossing   = Crossing(cfg["simulation"])
        controller = FixedTimeController(
            cfg["fixed_time_controller"]["phase_a_ticks"],
            cfg["fixed_time_controller"]["phase_b_ticks"],
        )
        return SimulationLoop(
            crossing     = crossing,
            controller   = controller,
            arrivals_fn  = lambda t: {"veh_ns": 1, "ped_l": 0, "ped_o": 0},
            tick_seconds = cfg["simulation"]["tick_seconds"],
        )

    def test_set_speed_updates_value(self, cfg):
        loop = self._make_loop(cfg)
        loop.set_speed(10.0)
        assert loop.speed == 10.0

    def test_set_speed_unpauses(self, cfg):
        loop = self._make_loop(cfg)
        loop.toggle_pause()
        assert loop.paused
        loop.set_speed(5.0)
        assert not loop.paused

    def test_toggle_pause_cycles(self, cfg):
        loop = self._make_loop(cfg)
        assert not loop.paused
        loop.toggle_pause()
        assert loop.paused
        loop.toggle_pause()
        assert not loop.paused

    def test_paused_loop_does_not_advance(self, cfg):
        loop = self._make_loop(cfg)
        loop.set_speed(1.0)
        loop.toggle_pause()
        tick_before = loop.state["tick"]
        loop.update(100.0)  # simula 100s enquanto pausado
        assert loop.state["tick"] == tick_before

    def test_update_advances_ticks(self, cfg):
        loop = self._make_loop(cfg)
        loop.set_speed(1.0)
        tick_s = cfg["simulation"]["tick_seconds"]
        loop.update(tick_s * 3)  # deve avançar exatamente 3 ticks
        assert loop.state["tick"] == 3

    def test_is_yellow_requires_pending_and_time(self, cfg):
        loop = self._make_loop(cfg)
        loop.state = {**loop.state, "pending_switch": True}
        loop._time_in_tick = 1.0  # antes dos 2s → ainda verde
        assert not loop.is_yellow
        loop._time_in_tick = 2.5  # após 2s → amarelo
        assert loop.is_yellow

    def test_is_yellow_false_without_pending(self, cfg):
        loop = self._make_loop(cfg)
        loop.state = {**loop.state, "pending_switch": False}
        loop._time_in_tick = 4.0
        assert not loop.is_yellow

    def test_reset_zeroes_state(self, cfg):
        loop = self._make_loop(cfg)
        loop.set_speed(1.0)
        loop.update(30.0)
        assert loop.state["tick"] > 0
        loop.reset()
        assert loop.state["tick"] == 0
        assert loop._time_in_tick == 0.0


# ── LivePlots ────────────────────────────────────────────────────────────────

class TestLivePlots:
    def _make_plots(self, cfg) -> LivePlots:
        return LivePlots(
            panel_width    = 512,
            panel_height   = 720,
            history_length = 100,
            tick_seconds   = cfg["simulation"]["tick_seconds"],
            ceil_cars_s    = cfg["thresholds"]["wait_ceiling_cars_seconds"],
            ceil_peds_s    = cfg["thresholds"]["wait_ceiling_pedestrians_seconds"],
        )

    def test_no_surface_before_record(self, cfg, pygame_init):
        plots = self._make_plots(cfg)
        assert plots.get_surface() is None
        plots.close()

    def test_surface_after_record_and_update(self, cfg, pygame_init):
        plots = self._make_plots(cfg)
        c = Crossing(cfg["simulation"])
        state = c.reset()
        for _ in range(10):
            state = c.step({"veh_ns": 1, "ped_l": 0, "ped_o": 0}, 0)
            plots.record(state)
        plots.update_surface(1)
        assert plots.get_surface() is not None
        plots.close()

    def test_clear_history_resets_surface(self, cfg, pygame_init):
        plots = self._make_plots(cfg)
        c = Crossing(cfg["simulation"])
        state = c.reset()
        for _ in range(5):
            state = c.step({"veh_ns": 1, "ped_l": 0, "ped_o": 0}, 0)
            plots.record(state)
        plots.update_surface(1)
        assert plots.get_surface() is not None
        plots.clear_history()
        assert plots.get_surface() is None
        assert len(plots._ticks) == 0
        plots.close()

    def test_update_interval_respected(self, cfg, pygame_init):
        plots = self._make_plots(cfg)
        c = Crossing(cfg["simulation"])
        state = c.reset()
        for _ in range(3):
            state = c.step({"veh_ns": 1, "ped_l": 0, "ped_o": 0}, 0)
            plots.record(state)
        plots.update_surface(10)  # interval > ticks registrados
        assert plots.get_surface() is None  # não deve ter renderizado ainda
        plots.close()
