"""Cálculo de métricas de desempenho do semáforo."""

from __future__ import annotations

from typing import Any


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
