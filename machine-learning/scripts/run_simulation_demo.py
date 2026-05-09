"""
Demo da simulação rodando por N ticks com controlador de tempo fixo.
Uso: python scripts/run_simulation_demo.py [num_ticks] [seed]
"""

import sys
import logging
from pathlib import Path

import numpy as np

# Permite importar src/ a partir da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.simulation.crossing import Crossing
from src.simulation.controllers import FixedTimeController
from src.simulation.metrics import compute_episode_summary

logging.basicConfig(level=logging.WARNING)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"

# Taxas médias de chegada por tick (cenário de dia útil, mistura de faixas)
_ARRIVAL_RATES = {
    "veh_ns": 1.2,
    "ped_l": 0.4,
    "ped_o": 0.4,
}


def sample_arrivals(rng: np.random.Generator) -> dict[str, int]:
    """Sorteia chegadas com distribuição Poisson para cada fila."""
    return {
        queue: int(rng.poisson(rate))
        for queue, rate in _ARRIVAL_RATES.items()
    }


def run_demo(num_ticks: int = 200, seed: int = 42) -> None:
    cfg = load_config(CONFIGS_DIR / "config.yaml")
    sim_cfg = cfg["simulation"]
    ft_cfg = cfg["fixed_time_controller"]
    tick_s = sim_cfg["tick_seconds"]

    crossing = Crossing(sim_cfg)
    controller = FixedTimeController(
        phase_a_ticks=ft_cfg["phase_a_ticks"],
        phase_b_ticks=ft_cfg["phase_b_ticks"],
    )

    rng = np.random.default_rng(seed)
    state = crossing.reset()
    history: list[dict] = []

    # Cabeçalho da tabela
    header = (
        f"{'Tick':>5} | {'Fase':>4} | {'T.Fase':>6} | "
        f"{'Veh(n)':>6} | {'PedL(n)':>7} | {'PedO(n)':>7} | "
        f"{'MaxEsp.V(s)':>11} | {'MaxEsp.P(s)':>11} | {'Ação':>5}"
    )
    sep = "-" * len(header)
    print(f"\nSimulação com controlador de TEMPO FIXO  "
          f"(Fase A={ft_cfg['phase_a_ticks']} ticks, Fase B={ft_cfg['phase_b_ticks']} ticks)")
    print(f"Taxas de chegada: veh_ns={_ARRIVAL_RATES['veh_ns']}/tick, "
          f"ped={_ARRIVAL_RATES['ped_l']}/tick por lado  |  seed={seed}\n")
    print(header)
    print(sep)

    for _ in range(num_ticks):
        arrivals = sample_arrivals(rng)
        action = controller.decide(state)
        state = crossing.step(arrivals, action)
        history.append(state)

        max_veh_s = state["veh_ns"]["max_wait_ticks"] * tick_s
        max_ped_s = max(
            state["ped_l"]["max_wait_ticks"],
            state["ped_o"]["max_wait_ticks"],
        ) * tick_s
        action_label = "TROCA" if action == 1 else "  -  "

        print(
            f"{state['tick']:>5} | {state['phase']:>4} | {state['ticks_in_phase']:>6} | "
            f"{state['veh_ns']['size']:>6} | {state['ped_l']['size']:>7} | {state['ped_o']['size']:>7} | "
            f"{max_veh_s:>11.0f} | {max_ped_s:>11.0f} | {action_label:>5}"
        )

    print(sep)
    summary = compute_episode_summary(history, tick_s)
    print("\nResumo do episódio:")
    print(f"  Ticks rodados:           {summary['total_ticks']}")
    print(f"  Espera máxima veículos:  {summary['max_wait_veh_seconds']:.0f}s")
    print(f"  Espera máxima pedestres: {summary['max_wait_ped_seconds']:.0f}s")
    print(f"  Fila média veículos:     {summary['avg_queue_veh']:.2f}")
    print(f"  Fila média pedestres:    {summary['avg_queue_ped']:.2f}")


if __name__ == "__main__":
    _ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    _seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run_demo(_ticks, _seed)
