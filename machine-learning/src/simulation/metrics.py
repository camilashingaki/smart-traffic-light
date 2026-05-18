"""Cálculo de métricas de desempenho do semáforo."""

from __future__ import annotations

import statistics
from typing import Any, Callable

# Type alias para funções de métrica
MetricFn = Callable[[list[dict[str, Any]], int, dict[str, Any]], float]

# ── Registry ──────────────────────────────────────────────────────────────────

METRIC_REGISTRY: dict[str, MetricFn] = {}


def metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """Decorator que registra uma função de métrica por nome no METRIC_REGISTRY."""
    def decorator(fn: MetricFn) -> MetricFn:
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator


def compute_metrics(
    names: list[str],
    tick_history: list[dict[str, Any]],
    tick_seconds: int,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """
    Calcula o subconjunto de métricas indicado em `names`.

    Parâmetros:
    - names: métricas a calcular (lista vinda de config.yaml metrics:)
    - tick_history: estados de Crossing.get_state() ao longo do episódio
    - tick_seconds: duração de um tick em segundos
    - cfg: configurações de RL (usa teto_espera_carros, teto_espera_pedestres)

    Retorna: dict nome→valor (float)
    """
    result: dict[str, float] = {}
    for name in names:
        fn = METRIC_REGISTRY.get(name)
        if fn is None:
            raise KeyError(
                f"Métrica desconhecida: {name!r}. Disponíveis: {sorted(METRIC_REGISTRY)}"
            )
        result[name] = fn(tick_history, tick_seconds, cfg)
    return result


# ── Helpers internos ──────────────────────────────────────────────────────────

def _collect_waits_seconds(
    tick_history: list[dict[str, Any]],
    queues: list[str],
    tick_seconds: int,
) -> list[float]:
    """Agrega tempos de espera (em segundos) de todas as entidades servidas nas filas indicadas."""
    waits: list[float] = []
    for state in tick_history:
        for q in queues:
            waits.extend(w * tick_seconds for w in state["served_this_tick"][q])
    return waits


def _percentile(data: list[float], p: float) -> float:
    """Percentil p (0–100) com interpolação linear. Retorna 0.0 para lista vazia."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    n = len(sorted_data)
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return float(sorted_data[-1])
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


# ── Espera média ──────────────────────────────────────────────────────────────

@metric("espera_media_carros")
def _espera_media_carros(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    waits = _collect_waits_seconds(tick_history, ["veh_ns"], tick_seconds)
    return float(statistics.mean(waits)) if waits else 0.0


@metric("espera_media_pedestres")
def _espera_media_pedestres(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    waits = _collect_waits_seconds(tick_history, ["ped_l", "ped_o"], tick_seconds)
    return float(statistics.mean(waits)) if waits else 0.0


# ── Espera máxima ─────────────────────────────────────────────────────────────

@metric("espera_maxima_carros")
def _espera_maxima_carros(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    waits = _collect_waits_seconds(tick_history, ["veh_ns"], tick_seconds)
    return float(max(waits)) if waits else 0.0


@metric("espera_maxima_pedestres")
def _espera_maxima_pedestres(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    waits = _collect_waits_seconds(tick_history, ["ped_l", "ped_o"], tick_seconds)
    return float(max(waits)) if waits else 0.0


# ── Espera p95 ────────────────────────────────────────────────────────────────

@metric("espera_p95_carros")
def _espera_p95_carros(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    waits = _collect_waits_seconds(tick_history, ["veh_ns"], tick_seconds)
    return _percentile(waits, 95)


@metric("espera_p95_pedestres")
def _espera_p95_pedestres(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    waits = _collect_waits_seconds(tick_history, ["ped_l", "ped_o"], tick_seconds)
    return _percentile(waits, 95)


# ── Violações de teto ─────────────────────────────────────────────────────────

@metric("violacoes_teto_carros")
def _violacoes_teto_carros(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    teto = float(cfg.get("teto_espera_carros", 90))
    waits = _collect_waits_seconds(tick_history, ["veh_ns"], tick_seconds)
    return float(sum(1 for w in waits if w > teto))


@metric("violacoes_teto_pedestres")
def _violacoes_teto_pedestres(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    teto = float(cfg.get("teto_espera_pedestres", 60))
    waits = _collect_waits_seconds(tick_history, ["ped_l", "ped_o"], tick_seconds)
    return float(sum(1 for w in waits if w > teto))


# ── Fila média ────────────────────────────────────────────────────────────────

@metric("fila_media_veh_ns")
def _fila_media_veh_ns(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    if not tick_history:
        return 0.0
    return float(sum(s["veh_ns"]["size"] for s in tick_history) / len(tick_history))


@metric("fila_media_ped_l")
def _fila_media_ped_l(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    if not tick_history:
        return 0.0
    return float(sum(s["ped_l"]["size"] for s in tick_history) / len(tick_history))


@metric("fila_media_ped_o")
def _fila_media_ped_o(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    if not tick_history:
        return 0.0
    return float(sum(s["ped_o"]["size"] for s in tick_history) / len(tick_history))


# ── Throughput ────────────────────────────────────────────────────────────────

@metric("throughput_total_carros")
def _throughput_total_carros(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    return float(sum(len(s["served_this_tick"]["veh_ns"]) for s in tick_history))


@metric("throughput_total_pedestres")
def _throughput_total_pedestres(
    tick_history: list[dict[str, Any]], tick_seconds: int, cfg: dict[str, Any]
) -> float:
    return float(
        sum(
            len(s["served_this_tick"]["ped_l"]) + len(s["served_this_tick"]["ped_o"])
            for s in tick_history
        )
    )


# ── Funções legadas (backward compatibility) ──────────────────────────────────

def compute_tick_reward(
    state: dict[str, Any],
    rl_cfg: dict[str, Any],
    tick_seconds: int,
) -> float:
    """
    Calcula a recompensa do agente para um tick conforme §6 do documento de instruções.

    Parâmetros:
    - state: estado retornado por Crossing.get_state()
    - rl_cfg: conteúdo do rl.yaml
    - tick_seconds: duração de um tick em segundos

    Retorna: recompensa escalar (negativa; quanto maior o valor absoluto, pior)
    """
    weights = rl_cfg["reward_weights"]
    teto_carros_ticks = rl_cfg["teto_espera_carros"] // tick_seconds
    teto_peds_ticks = rl_cfg["teto_espera_pedestres"] // tick_seconds

    total_wait = (
        state["veh_ns"]["total_wait_ticks"]
        + state["ped_l"]["total_wait_ticks"]
        + state["ped_o"]["total_wait_ticks"]
    )
    total_size = (
        state["veh_ns"]["size"] + state["ped_l"]["size"] + state["ped_o"]["size"]
    )
    max_wait = max(
        state["veh_ns"]["max_wait_ticks"],
        state["ped_l"]["max_wait_ticks"],
        state["ped_o"]["max_wait_ticks"],
    )

    veh_load = state["veh_ns"]["total_wait_ticks"]
    ped_load = state["ped_l"]["total_wait_ticks"] + state["ped_o"]["total_wait_ticks"]
    imbalance = abs(veh_load - ped_load)

    # Penalidade por entidades acima do teto (usa max_wait como proxy por ora;
    # implementação per-entidade virá na Fase 4 com acesso direto às filas)
    exceeded_penalty = 0.0
    if state["veh_ns"]["max_wait_ticks"] > teto_carros_ticks:
        exceeded_penalty += weights["excedeu_teto"]
    if max(state["ped_l"]["max_wait_ticks"], state["ped_o"]["max_wait_ticks"]) > teto_peds_ticks:
        exceeded_penalty += weights["excedeu_teto"]

    reward = (
        weights["espera_acumulada"] * total_wait
        + weights["tamanho_filas"] * total_size
        + weights["max_espera"] * max_wait
        + weights["desequilibrio"] * imbalance
        + exceeded_penalty
    )
    return reward


def compute_episode_summary(tick_history: list[dict[str, Any]], tick_seconds: int) -> dict[str, Any]:
    """
    Agrega as métricas de todos os ticks de um episódio.

    Parâmetros:
    - tick_history: lista de estados retornados por Crossing.get_state()
    - tick_seconds: duração de um tick em segundos

    Retorna: dicionário com espera média/máxima em segundos e tamanho médio de fila
    """
    if not tick_history:
        return {}

    max_wait_veh_s = max(s["veh_ns"]["max_wait_ticks"] for s in tick_history) * tick_seconds
    max_wait_ped_s = max(
        max(s["ped_l"]["max_wait_ticks"], s["ped_o"]["max_wait_ticks"])
        for s in tick_history
    ) * tick_seconds

    avg_queue_veh = sum(s["veh_ns"]["size"] for s in tick_history) / len(tick_history)
    avg_queue_ped = sum(
        s["ped_l"]["size"] + s["ped_o"]["size"] for s in tick_history
    ) / len(tick_history)

    return {
        "total_ticks": len(tick_history),
        "max_wait_veh_seconds": max_wait_veh_s,
        "max_wait_ped_seconds": max_wait_ped_s,
        "avg_queue_veh": round(avg_queue_veh, 2),
        "avg_queue_ped": round(avg_queue_ped, 2),
    }
