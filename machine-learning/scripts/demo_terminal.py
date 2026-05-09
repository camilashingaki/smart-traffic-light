"""
Demo terminal: roda o controlador de tempo fixo por N ticks e imprime o estado.
Script temporário para validar o critério de aceite da Fase 1.
"""

import random
import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path para importar src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation.crossing import Crossing
from src.simulation.controllers import FixedTimeController
from src.utils.config_loader import load_all_configs

# ── Configuração ──────────────────────────────────────────────────────────────
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PRINT_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SEED = 42

ARRIVAL_RATES = {          # chegadas médias por tick (simulação simples sem gerador completo)
    "veh_ns": float(sys.argv[3]) if len(sys.argv) > 3 else 2.5,
    "ped_l":  0.8,
    "ped_o":  0.7,
}

# ── Cores ANSI ────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

def phase_color(phase: str) -> str:
    return GREEN if phase == "A" else YELLOW

def bar(value: int, max_val: int = 20, width: int = 20) -> str:
    filled = min(int(value / max_val * width), width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

# ── Inicialização ─────────────────────────────────────────────────────────────
rng = random.Random(SEED)
cfgs = load_all_configs(ROOT / "configs")
sim_cfg = cfgs["config"]["simulation"]
ft_cfg  = cfgs["config"]["fixed_time_controller"]

crossing    = Crossing(sim_cfg)
controller  = FixedTimeController(ft_cfg["phase_a_ticks"], ft_cfg["phase_b_ticks"])
tick_sec    = sim_cfg["tick_seconds"]

crossing.reset()

print(f"\n{BOLD}{'='*62}{RESET}")
print(f"{BOLD}  SEMÁFORO INTELIGENTE — Demo Terminal (Fase 1){RESET}")
print(f"  {DIM}Controlador: Tempo Fixo | "
      f"Fase A={ft_cfg['phase_a_ticks']} ticks ({ft_cfg['phase_a_ticks']*tick_sec}s) | "
      f"Fase B={ft_cfg['phase_b_ticks']} ticks ({ft_cfg['phase_b_ticks']*tick_sec}s){RESET}")
print(f"  {DIM}Ticks totais: {TICKS} | Imprime a cada: {PRINT_EVERY} ticks | Seed: {SEED}{RESET}")
print(f"{BOLD}{'='*62}{RESET}\n")

header = (
    f"{'Tick':>5} {'Tempo':>7}  {'Fase':^6}  "
    f"{'veh_ns':>6} {'veh_max':>7}  "
    f"{'ped_l':>5} {'ped_o':>5} {'ped_max':>7}  "
    f"{'Ação':>6}"
)
print(f"{BOLD}{DIM}{header}{RESET}")
print(DIM + "-" * 62 + RESET)

# ── Loop principal ────────────────────────────────────────────────────────────
phase_switches = 0
prev_phase = "A"
tick_seconds_total = 0

for t in range(TICKS):
    arrivals = {
        k: max(0, round(rng.gauss(rate, rate * 0.5)))
        for k, rate in ARRIVAL_RATES.items()
    }

    state   = crossing.get_state()
    action  = controller.decide(state)
    state   = crossing.step(arrivals, action)

    tick_seconds_total = state["tick"] * tick_sec
    mins  = tick_seconds_total // 60
    secs  = tick_seconds_total % 60

    if state["phase"] != prev_phase:
        phase_switches += 1
        prev_phase = state["phase"]

    if (t + 1) % PRINT_EVERY == 0 or t == 0:
        ph   = state["phase"]
        col  = phase_color(ph)
        ph_label = f"A(car)" if ph == "A" else "B(ped)"

        action_label = f"{CYAN}TROCAR{RESET}" if action == 1 else f"{DIM}manter{RESET}"
        if state["pending_switch"]:
            action_label = f"{YELLOW}pendente{RESET}"

        veh_size     = state["veh_ns"]["size"]
        veh_max_wait = state["veh_ns"]["max_wait_ticks"] * tick_sec
        ped_l_size   = state["ped_l"]["size"]
        ped_o_size   = state["ped_o"]["size"]
        ped_max_wait = max(
            state["ped_l"]["max_wait_ticks"],
            state["ped_o"]["max_wait_ticks"],
        ) * tick_sec

        row = (
            f"{state['tick']:>5} {mins:>3}m{secs:>02d}s  "
            f"{col}{ph_label:^6}{RESET}  "
            f"{veh_size:>6} {veh_max_wait:>6}s  "
            f"{ped_l_size:>5} {ped_o_size:>5} {ped_max_wait:>6}s  "
            f"{action_label}"
        )
        print(row)

# ── Resumo final ──────────────────────────────────────────────────────────────
final = crossing.get_state()
veh_max_s  = final["veh_ns"]["max_wait_ticks"] * tick_sec
ped_max_s  = max(
    final["ped_l"]["max_wait_ticks"],
    final["ped_o"]["max_wait_ticks"],
) * tick_sec
total_mins = (TICKS * tick_sec) // 60

print(f"\n{BOLD}{'='*62}{RESET}")
print(f"{BOLD}  Resumo — {TICKS} ticks simulados ({total_mins} min){RESET}")
print(f"  Trocas de fase:          {phase_switches}")
print(f"  Carros na fila (final):  {final['veh_ns']['size']}")
print(f"  Pedestres leste (final): {final['ped_l']['size']}")
print(f"  Pedestres oeste (final): {final['ped_o']['size']}")
print(f"  Maior espera carros:     {veh_max_s}s")
print(f"  Maior espera pedestres:  {ped_max_s}s")
print(f"{BOLD}{'='*62}{RESET}\n")
