"""Testes unitários do MetricsRegistry e das 13 métricas."""

from __future__ import annotations

import pytest

from src.simulation.metrics import (
    METRIC_REGISTRY,
    compute_episode_summary,
    compute_metrics,
    compute_tick_reward,
)

TICK_SECONDS = 5
CFG = {"teto_espera_carros": 90, "teto_espera_pedestres": 60}

ALL_METRICS = [
    "espera_media_carros",
    "espera_media_pedestres",
    "espera_maxima_carros",
    "espera_maxima_pedestres",
    "espera_p95_carros",
    "espera_p95_pedestres",
    "violacoes_teto_carros",
    "violacoes_teto_pedestres",
    "fila_media_veh_ns",
    "fila_media_ped_l",
    "fila_media_ped_o",
    "throughput_total_carros",
    "throughput_total_pedestres",
]

RL_CFG = {
    "reward_weights": {
        "espera_acumulada": -1.0,
        "tamanho_filas": -0.5,
        "max_espera": -2.0,
        "desequilibrio": -0.3,
        "excedeu_teto": -10.0,
    },
    "teto_espera_carros": 90,
    "teto_espera_pedestres": 60,
}


def make_state(
    veh_served: tuple = (),
    ped_l_served: tuple = (),
    ped_o_served: tuple = (),
    veh_size: int = 0,
    ped_l_size: int = 0,
    ped_o_size: int = 0,
    veh_max_wait: int = 0,
    ped_l_max_wait: int = 0,
    ped_o_max_wait: int = 0,
    tick: int = 0,
) -> dict:
    """Constrói um estado sintético compatível com Crossing.get_state()."""
    return {
        "tick": tick,
        "phase": "A",
        "ticks_in_phase": 1,
        "pending_switch": False,
        "served_this_tick": {
            "veh_ns": list(veh_served),
            "ped_l": list(ped_l_served),
            "ped_o": list(ped_o_served),
        },
        "veh_ns": {
            "size": veh_size,
            "max_wait_ticks": veh_max_wait,
            "total_wait_ticks": veh_size * veh_max_wait,
        },
        "ped_l": {
            "size": ped_l_size,
            "max_wait_ticks": ped_l_max_wait,
            "total_wait_ticks": ped_l_size * ped_l_max_wait,
        },
        "ped_o": {
            "size": ped_o_size,
            "max_wait_ticks": ped_o_max_wait,
            "total_wait_ticks": ped_o_size * ped_o_max_wait,
        },
    }


# ── Registry ──────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_expected_metrics_registered(self):
        for name in ALL_METRICS:
            assert name in METRIC_REGISTRY, f"{name!r} não está no METRIC_REGISTRY"

    def test_registry_has_exactly_13_metrics(self):
        assert len(METRIC_REGISTRY) == 13

    def test_all_registry_values_are_callable(self):
        for name, fn in METRIC_REGISTRY.items():
            assert callable(fn), f"METRIC_REGISTRY[{name!r}] não é callable"


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_dict_with_requested_keys(self):
        history = [make_state(veh_served=(2, 3), veh_size=1)]
        result = compute_metrics(["espera_media_carros"], history, TICK_SECONDS, CFG)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"espera_media_carros"}

    def test_all_metrics_return_float(self):
        history = [
            make_state(
                veh_served=(2, 5),
                ped_l_served=(3,),
                ped_o_served=(1,),
                veh_size=2,
                ped_l_size=1,
            )
        ]
        result = compute_metrics(ALL_METRICS, history, TICK_SECONDS, CFG)
        for name, val in result.items():
            assert isinstance(val, float), f"{name!r} retornou {type(val).__name__}, esperado float"

    def test_unknown_metric_raises_key_error(self):
        with pytest.raises(KeyError, match="metrica_inexistente"):
            compute_metrics(["metrica_inexistente"], [], TICK_SECONDS, CFG)

    def test_empty_history_returns_zeros_for_all(self):
        result = compute_metrics(ALL_METRICS, [], TICK_SECONDS, CFG)
        for name, val in result.items():
            assert val == 0.0, f"{name!r} retornou {val} para histórico vazio, esperado 0.0"

    def test_subset_of_metrics(self):
        history = [make_state(veh_served=(10,))]
        names = ["espera_maxima_carros", "throughput_total_carros"]
        result = compute_metrics(names, history, TICK_SECONDS, CFG)
        assert list(result.keys()) == names


# ── Espera média ──────────────────────────────────────────────────────────────

class TestEsperaMedia:
    def test_espera_media_carros_valor_correto(self):
        # served: waits=[4, 6] ticks → média = 5 ticks * 5s = 25s
        history = [make_state(veh_served=(4, 6))]
        result = compute_metrics(["espera_media_carros"], history, TICK_SECONDS, CFG)
        assert result["espera_media_carros"] == pytest.approx(25.0)

    def test_espera_media_pedestres_agrega_ped_l_e_ped_o(self):
        # ped_l=[2], ped_o=[4] → média = (2+4)/2 * 5s = 15s
        history = [make_state(ped_l_served=(2,), ped_o_served=(4,))]
        result = compute_metrics(["espera_media_pedestres"], history, TICK_SECONDS, CFG)
        assert result["espera_media_pedestres"] == pytest.approx(15.0)

    def test_espera_media_sem_servidos_retorna_zero(self):
        history = [make_state(veh_size=3)]
        result = compute_metrics(
            ["espera_media_carros", "espera_media_pedestres"], history, TICK_SECONDS, CFG
        )
        assert result["espera_media_carros"] == 0.0
        assert result["espera_media_pedestres"] == 0.0

    def test_espera_media_agrega_multiplos_ticks(self):
        # tick 1: served=[2], tick 2: served=[8] → média = (2+8)/2 * 5s = 25s
        history = [make_state(veh_served=(2,)), make_state(veh_served=(8,))]
        result = compute_metrics(["espera_media_carros"], history, TICK_SECONDS, CFG)
        assert result["espera_media_carros"] == pytest.approx(25.0)


# ── Espera máxima ─────────────────────────────────────────────────────────────

class TestEsperaMaxima:
    def test_espera_maxima_carros(self):
        # served: [3, 10, 7] → max = 10 * 5s = 50s
        history = [make_state(veh_served=(3, 10, 7))]
        result = compute_metrics(["espera_maxima_carros"], history, TICK_SECONDS, CFG)
        assert result["espera_maxima_carros"] == pytest.approx(50.0)

    def test_espera_maxima_pedestres_multiplos_ticks(self):
        # tick 1: ped_l=[5], tick 2: ped_o=[8] → max = 8 * 5s = 40s
        history = [make_state(ped_l_served=(5,)), make_state(ped_o_served=(8,))]
        result = compute_metrics(["espera_maxima_pedestres"], history, TICK_SECONDS, CFG)
        assert result["espera_maxima_pedestres"] == pytest.approx(40.0)

    def test_espera_maxima_zero_sem_servidos(self):
        history = [make_state(veh_size=5)]
        result = compute_metrics(["espera_maxima_carros"], history, TICK_SECONDS, CFG)
        assert result["espera_maxima_carros"] == 0.0


# ── Espera p95 ────────────────────────────────────────────────────────────────

class TestEsperaP95:
    def test_p95_valor_unico(self):
        history = [make_state(veh_served=(10,))]
        result = compute_metrics(["espera_p95_carros"], history, TICK_SECONDS, CFG)
        assert result["espera_p95_carros"] == pytest.approx(50.0)

    def test_p95_lista_vazia_retorna_zero(self):
        result = compute_metrics(["espera_p95_carros", "espera_p95_pedestres"], [], TICK_SECONDS, CFG)
        assert result["espera_p95_carros"] == 0.0
        assert result["espera_p95_pedestres"] == 0.0

    def test_p95_menor_ou_igual_a_maximo(self):
        history = [make_state(veh_served=tuple(range(1, 101)))]
        result = compute_metrics(
            ["espera_p95_carros", "espera_maxima_carros"], history, TICK_SECONDS, CFG
        )
        assert result["espera_p95_carros"] <= result["espera_maxima_carros"]

    def test_p95_maior_ou_igual_a_media(self):
        history = [make_state(veh_served=tuple(range(1, 101)))]
        result = compute_metrics(
            ["espera_p95_carros", "espera_media_carros"], history, TICK_SECONDS, CFG
        )
        assert result["espera_p95_carros"] >= result["espera_media_carros"]


# ── Violações de teto ─────────────────────────────────────────────────────────

class TestViolacoesTeto:
    def test_sem_violacoes_carros(self):
        # teto=90s → 18 ticks; served=[5,10,15] → todos ≤18 ticks
        history = [make_state(veh_served=(5, 10, 15))]
        result = compute_metrics(["violacoes_teto_carros"], history, TICK_SECONDS, CFG)
        assert result["violacoes_teto_carros"] == 0.0

    def test_com_violacoes_carros(self):
        # teto=90s → 18 ticks; served=[5,19,20] → 19*5=95s>90, 20*5=100s>90 → 2 violações
        history = [make_state(veh_served=(5, 19, 20))]
        result = compute_metrics(["violacoes_teto_carros"], history, TICK_SECONDS, CFG)
        assert result["violacoes_teto_carros"] == 2.0

    def test_com_violacoes_pedestres(self):
        # teto=60s → 12 ticks; ped_l=[13], ped_o=[14] → 2 violações
        history = [make_state(ped_l_served=(13,), ped_o_served=(14,))]
        result = compute_metrics(["violacoes_teto_pedestres"], history, TICK_SECONDS, CFG)
        assert result["violacoes_teto_pedestres"] == 2.0

    def test_limite_exato_nao_e_violacao(self):
        # 18 ticks * 5s = 90s = teto_carros → NÃO é violação (condição: w > teto)
        history = [make_state(veh_served=(18,))]
        result = compute_metrics(["violacoes_teto_carros"], history, TICK_SECONDS, CFG)
        assert result["violacoes_teto_carros"] == 0.0

    def test_sem_violacoes_retorna_zero(self):
        history = [make_state(veh_size=3, ped_l_size=2)]
        result = compute_metrics(
            ["violacoes_teto_carros", "violacoes_teto_pedestres"], history, TICK_SECONDS, CFG
        )
        assert result["violacoes_teto_carros"] == 0.0
        assert result["violacoes_teto_pedestres"] == 0.0


# ── Fila média ────────────────────────────────────────────────────────────────

class TestFilaMedia:
    def test_fila_media_veh_ns(self):
        # sizes: 4, 6 → média = 5.0
        history = [make_state(veh_size=4), make_state(veh_size=6)]
        result = compute_metrics(["fila_media_veh_ns"], history, TICK_SECONDS, CFG)
        assert result["fila_media_veh_ns"] == pytest.approx(5.0)

    def test_fila_media_ped_l_e_ped_o(self):
        history = [make_state(ped_l_size=3), make_state(ped_l_size=5)]
        result = compute_metrics(["fila_media_ped_l"], history, TICK_SECONDS, CFG)
        assert result["fila_media_ped_l"] == pytest.approx(4.0)

    def test_fila_media_zero_quando_vazio(self):
        result = compute_metrics(
            ["fila_media_veh_ns", "fila_media_ped_l", "fila_media_ped_o"],
            [],
            TICK_SECONDS,
            CFG,
        )
        assert result["fila_media_veh_ns"] == 0.0
        assert result["fila_media_ped_l"] == 0.0
        assert result["fila_media_ped_o"] == 0.0


# ── Throughput ────────────────────────────────────────────────────────────────

class TestThroughput:
    def test_throughput_total_carros(self):
        # tick 1: 3 servidos, tick 2: 2 servidos → 5 total
        history = [make_state(veh_served=(1, 2, 3)), make_state(veh_served=(4, 5))]
        result = compute_metrics(["throughput_total_carros"], history, TICK_SECONDS, CFG)
        assert result["throughput_total_carros"] == 5.0

    def test_throughput_total_pedestres_agrega_l_e_o(self):
        # ped_l=[1,2], ped_o=[3] → 3 total
        history = [make_state(ped_l_served=(1, 2), ped_o_served=(3,))]
        result = compute_metrics(["throughput_total_pedestres"], history, TICK_SECONDS, CFG)
        assert result["throughput_total_pedestres"] == 3.0

    def test_throughput_zero_sem_servidos(self):
        history = [make_state(veh_size=5, ped_l_size=3)]
        result = compute_metrics(
            ["throughput_total_carros", "throughput_total_pedestres"], history, TICK_SECONDS, CFG
        )
        assert result["throughput_total_carros"] == 0.0
        assert result["throughput_total_pedestres"] == 0.0


# ── Backward compatibility ────────────────────────────────────────────────────

class TestBackwardCompatibility:
    def test_compute_tick_reward_retorna_float_negativo(self):
        state = make_state(
            veh_size=3, ped_l_size=2, ped_o_size=1,
            veh_max_wait=4, ped_l_max_wait=2, ped_o_max_wait=1,
        )
        reward = compute_tick_reward(state, RL_CFG, TICK_SECONDS)
        assert isinstance(reward, float)
        assert reward < 0

    def test_compute_tick_reward_estado_vazio(self):
        state = make_state()
        reward = compute_tick_reward(state, RL_CFG, TICK_SECONDS)
        assert reward == pytest.approx(0.0)

    def test_compute_episode_summary_retorna_dict(self):
        history = [make_state(veh_size=3, veh_max_wait=5, ped_l_size=2, ped_l_max_wait=3)]
        summary = compute_episode_summary(history, TICK_SECONDS)
        assert isinstance(summary, dict)
        assert "max_wait_veh_seconds" in summary
        assert "max_wait_ped_seconds" in summary
        assert "avg_queue_veh" in summary
        assert "total_ticks" in summary

    def test_compute_episode_summary_historico_vazio(self):
        assert compute_episode_summary([], TICK_SECONDS) == {}

    def test_compute_episode_summary_valores(self):
        # max_wait_ticks=6 → 6 * 5s = 30s
        history = [make_state(veh_size=2, veh_max_wait=6)]
        summary = compute_episode_summary(history, TICK_SECONDS)
        assert summary["max_wait_veh_seconds"] == pytest.approx(30.0)
        assert summary["total_ticks"] == 1
