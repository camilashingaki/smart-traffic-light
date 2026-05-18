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
        "saturation_flow_veh_per_lane_per_tick": 1.5,
        "saturation_flow_ped_per_side_per_tick": 4,
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
        # Fase A tick 1: crédito 0→1.5 por faixa → drena floor(1.5)=1 por faixa = 2 total
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
        # Fase B: 4 por lado drenam; 5-4=1 restam por lado
        assert state["phase"] == "B"
        assert state["ped_l"]["size"] == 1
        assert state["ped_o"]["size"] == 1

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


# ── Drenagem fracionária ──────────────────────────────────────────────────────

class TestFractionalDrain:
    """Valida a lógica de crédito fracionário por faixa (saturation_flow=1.5/faixa)."""

    def test_average_throughput_converges(self, sim_cfg):
        """Média de carros drenados deve convergir para 1.5/faixa/tick = 3.0/tick total."""
        c = Crossing(sim_cfg)
        # Encher a fila para que nunca fique vazia durante o teste
        c.veh_ns.add_arrivals(10_000)
        total_drained = 0
        N = 1_000
        for _ in range(N):
            state = c.step({}, action=0)
            total_drained += len(state["served_this_tick"]["veh_ns"])
        avg = total_drained / N
        assert abs(avg - 3.0) < 0.05, (
            f"Throughput médio esperado 3.0 carros/tick, obtido {avg:.4f}"
        )

    def test_credits_reset_on_phase_switch(self, sim_cfg):
        """
        Após troca de fase (A→B→A), o 1º tick em Fase A deve drenar floor(0+1.5)×2 = 2 carros,
        e não floor(0.5+1.5)×2 = 4 (o que aconteceria se o crédito não fosse resetado).

        Cenário:
          - Calls 1–4: Phase.A sem troca (crédito alterna 0.5/0.0 ao final de cada tick)
          - Call 5: ticks_in=4 → elegível; solicita troca → crédito termina em [0.5, 0.5]
          - Call 6: troca executa (A→B); créditos devem resetar para [0.0, 0.0]
          - Calls 7–8: Phase.B (mínimo 2 ticks); call 8 solicita volta para A
          - Call 9: troca B→A executa; créditos resetam novamente
          - Call 10: 1º tick real em Phase.A — deve drenar exatamente 2 (1/faixa)
        """
        c = Crossing(sim_cfg)
        c.veh_ns.add_arrivals(10_000)

        # Calls 1–4: ficar em Phase.A, não solicitar troca
        for _ in range(4):
            c.step({}, action=0)

        # Call 5: ticks_in_phase=4 → elegível; solicitar troca
        # Crédito ao final deste tick: [0.0+1.5=1.5→drain1→0.5, idem] = [0.5, 0.5]
        state = c.step({}, action=1)
        assert state["pending_switch"], "Switch deveria estar pendente após call 5"

        # Call 6: troca A→B executa; créditos resetam para [0.0, 0.0]
        state = c.step({}, action=0)
        assert state["phase"] == "B", "Deveria estar em Phase.B após call 6"

        # Call 7: Phase.B tick 1 (ticks_in=1 após incremento; mínimo=2, não elegível)
        c.step({}, action=0)

        # Call 8: Phase.B tick 2 (ticks_in=2 → elegível); solicitar troca B→A
        state = c.step({}, action=1)
        assert state["pending_switch"], "Switch B→A deveria estar pendente após call 8"

        # Call 9: troca B→A executa (créditos resetam para [0,0]) e, no mesmo step,
        # process_flow em Phase.A usa crédito zerado: [0+1.5=1.5]→drain1 por faixa = 2 total.
        # Se o crédito NÃO fosse resetado, ficaria [0.5+1.5=2.0]→drain2 por faixa = 4 total.
        state = c.step({}, action=0)
        assert state["phase"] == "A", "Deveria estar em Phase.A após call 9"
        drained = len(state["served_this_tick"]["veh_ns"])
        assert drained == 2, (
            f"1º tick em Phase.A pós-switch deve drenar 2 carros (1/faixa com crédito=0); "
            f"obtido {drained}. Se fosse 4, o crédito não foi resetado corretamente."
        )
