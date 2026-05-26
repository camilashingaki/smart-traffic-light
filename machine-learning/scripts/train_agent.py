"""
Script para executar o treinamento do agente PPO — Fase 5.

Rode com:
    python scripts/train_agent.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from src.rl.train import treinar


def main() -> None:
    print("\n" + "=" * 60)
    print("  SEMAFORO INTELIGENTE — Fase 5: Treinamento PPO")
    print("=" * 60)

    if not any(Path("scenarios/train").glob("*.csv")):
        print("\nERRO: Nenhum cenario encontrado em scenarios/train/")
        print("Execute primeiro: python scripts/generate_scenarios.py\n")
        sys.exit(1)

    if not any(Path("scenarios/eval").glob("*.csv")):
        print("\nERRO: Nenhum cenario encontrado em scenarios/eval/")
        print("Execute primeiro: python scripts/generate_scenarios.py\n")
        sys.exit(1)

    print("\nCenarios encontrados. Iniciando treinamento...\n")
    treinar()


if __name__ == "__main__":
    main()
