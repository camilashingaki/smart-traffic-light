"""Testes unitários das primitivas do cruzamento."""

import pytest

from src.simulation.crossing import Crossing, Phase, TrafficQueue


@pytest.fixture
def sim_cfg() -> dict:
    return {
        "tick_seconds": 5,
        "min_green_cars_ticks": 3,
        "min_green_pedestrians_ticks": 2,
        "yellow_visual_seconds": 3,
        "num_car_lanes": 2,
        "saturation_flow_veh_per_lane_per_tick": 1,
        "saturation_flow_ped_per_side_per_tick": 3,
    }


# ── TrafficQueue ──────────────────────────────────────────────────────────────

class TestTrafficQueue:
    def test_initial_state(self):
        q = TrafficQueue("test")
        assert q.size == 0
        assert q.max_wait_ticks == 0
        assert q.total_wait_ticks == 0

    def test_add_arrivals(self):
        q = TrafficQueue("test")
        q.add_arrivals(4)
        assert q.size == 4
        assert q.total_wait_ticks == 0  # recém chegados, espera zero

    def test_drain_partial(self):
        q = TrafficQueue("test")
        q.add_arrivals(5)
        served = q.drain(3)
        assert len(served) == 3
        assert q.size == 2

    def test_drain_exceeds_size(self):
        q = TrafficQueue("test")
        q.add_arrivals(2)
        served = q.drain(10)
        assert len(served) == 2
        assert q.size == 0

    def test_drain_empty_queue(self):
        q = TrafficQueue("test")
        served = q.drain(5)
        assert served == []

    def test_increment_waits(self):
        q = TrafficQueue("test")
        q.add_arrivals(3)
        q.increment_waits()
        assert q.max_wait_ticks == 1
        assert q.total_wait_ticks == 3

    def test_fifo_order(self):
        """Entidade mais antiga (maior espera) deve ser servida primeiro."""
        q = TrafficQueue("test")
        q.add_arrivals(1)   # entidade A: wait=0
        q.increment_waits() # entidade A: wait=1
        q.add_arrivals(1)   # entidade B: wait=0
        served = q.drain(1)
        assert served[0] == 1  # entidade A sai primeiro

    def test_snapshot(self):
        q = TrafficQueue("test")
        q.add_arrivals(2)
        q.increment_waits()
        snap = q.snapshot()
        assert snap == [1, 1]


# ── Crossing ──────────────────────────────────────────────────────────────────

class TestCrossing:
    def test_initial_state(self, sim_cfg):
        c = Crossing(sim_cfg)
        state = c.get_state()
        assert state["phase"] == "A"
        assert state["ticks_in_phase"] == 0
        assert state["tick"] == 0
        assert state["veh_ns"]["size"] == 0
        assert not state["pending_switch"]

    def test_arrivals_added_and_partially_drained(self, sim_cfg):
        c = Crossing(sim_cfg)
        # Fase A: saturation = 1 carro/faixa × 2 faixas = 2/tick
        # 5 chegam, 2 drenam → 3 restam, espera=1 após increment
        state = c.step({"veh_ns": 5, "ped_l": 0, "ped_o": 0}, action=0)
        assert state["veh_ns"]["size"] == 3
        assert state["veh_ns"]["max_wait_ticks"] == 1

    def test_pedestrians_drain_in_phase_b(self, sim_cfg):
        c = Crossing(sim_cfg)
        # Avançar até poder trocar (min=3 ticks) e então trocar
        for _ in range(3):
            c.step({}, action=0)
        c.step({}, action=1)  # agenda troca
        # Na próxima step a troca entra em vigor (Fase B)
        # Adiciona pedestres e verifica escoamento
        state = c.step({"ped_l": 5, "ped_o": 5}, action=0)
        # Fase B: 3 por lado drenam; 5-3=2 restam por lado
        assert state["phase"] == "B"
        assert state["ped_l"]["size"] == 2
        assert state["ped_o"]["size"] == 2

    def test_min_green_time_enforced_phase_a(self, sim_cfg):
        c = Crossing(sim_cfg)
        # Tenta trocar nos ticks 0 e 1 — abaixo do mínimo de 3
        state = c.step({}, action=1)
        assert state["phase"] == "A"
        assert not state["pending_switch"]
        state = c.step({}, action=1)
        assert state["phase"] == "A"
        assert not state["pending_switch"]

    def test_phase_switch_after_min_time(self, sim_cfg):
        c = Crossing(sim_cfg)
        # 3 ticks de espera
        for _ in range(3):
            state = c.step({}, action=0)
        assert state["ticks_in_phase"] == 3
        # Solicita troca no tick 3 (ticks_in_phase=3 >= min=3)
        state = c.step({}, action=1)
        assert state["pending_switch"]
        # Troca entra em vigor no próximo tick
        state = c.step({}, action=0)
        assert state["phase"] == "B"

    def test_min_green_enforced_after_switch(self, sim_cfg):
        """Após trocar para B, não deve ser possível trocar imediatamente (min=2)."""
        c = Crossing(sim_cfg)
        for _ in range(3):
            c.step({}, action=0)
        c.step({}, action=1)   # agenda A→B
        c.step({}, action=0)   # executa troca; agora em B, ticks_in_phase=1
        # ticks_in_phase=1 < min_peds=2: troca não permitida
        state = c.step({}, action=1)
        assert state["phase"] == "B"
        assert not state["pending_switch"]

    def test_tick_counter_increments(self, sim_cfg):
        c = Crossing(sim_cfg)
        for i in range(5):
            state = c.step({}, action=0)
        assert state["tick"] == 5

    def test_reset_clears_state(self, sim_cfg):
        c = Crossing(sim_cfg)
        c.step({"veh_ns": 10, "ped_l": 5, "ped_o": 5}, action=0)
        state = c.reset()
        assert state["tick"] == 0
        assert state["veh_ns"]["size"] == 0
        assert state["ped_l"]["size"] == 0
        assert state["phase"] == "A"
        assert state["ticks_in_phase"] == 0
