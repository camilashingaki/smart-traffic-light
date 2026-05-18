"""Testes unitários do controlador de tempo fixo."""

import pytest

from src.simulation.controllers import FixedTimeController


class TestFixedTimeController:
    def test_no_switch_before_phase_a_duration(self):
        ctrl = FixedTimeController(phase_a_ticks=6, phase_b_ticks=4)
        state = {"phase": "A", "ticks_in_phase": 5}
        assert ctrl.decide(state) == 0

    def test_switch_exactly_at_phase_a_duration(self):
        ctrl = FixedTimeController(phase_a_ticks=6, phase_b_ticks=4)
        state = {"phase": "A", "ticks_in_phase": 6}
        assert ctrl.decide(state) == 1

    def test_switch_after_phase_a_duration(self):
        ctrl = FixedTimeController(phase_a_ticks=6, phase_b_ticks=4)
        state = {"phase": "A", "ticks_in_phase": 9}
        assert ctrl.decide(state) == 1

    def test_no_switch_before_phase_b_duration(self):
        ctrl = FixedTimeController(phase_a_ticks=6, phase_b_ticks=4)
        state = {"phase": "B", "ticks_in_phase": 3}
        assert ctrl.decide(state) == 0

    def test_switch_at_phase_b_duration(self):
        ctrl = FixedTimeController(phase_a_ticks=6, phase_b_ticks=4)
        state = {"phase": "B", "ticks_in_phase": 4}
        assert ctrl.decide(state) == 1
