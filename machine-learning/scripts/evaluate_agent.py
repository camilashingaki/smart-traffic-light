"""
Script para executar a avaliação comparativa do agente PPO — Fase 6.

Rode com:
    python scripts/evaluate_agent.py

Antes de rodar, certifique-se de que:
    1. O venv está ativado           (source .venv/bin/activate)
    2. O benchmark foi rodado        (python scripts/run_benchmark.py)
    3. O agente foi treinado         (python scripts/train_agent.py)
    4. Os cenários de eval existem   (scenarios/eval/*.csv)

Saídas geradas em results/:
    agent_results.csv  — métricas do agente por cenário
    comparison.csv     — comparação direta agente vs baseline
    relatorio.md       — relatório final com análise completa
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

from src.rl.evaluate import avaliar


def main() -> None:
    print("\n" + "=" * 60)
    print("  SEMÁFORO INTELIGENTE — Fase 6: Avaliação")
    print("=" * 60)

    # Verifica pré-requisitos
    erros = []

    if not any(Path("scenarios/eval").glob("*.csv")):
        erros.append("Cenários de avaliação não encontrados em scenarios/eval/")
        erros.append("  → Execute: python scripts/generate_scenarios.py")

    if not Path("results/benchmark_baseline.csv").exists():
        erros.append("Baseline não encontrado em results/benchmark_baseline.csv")
        erros.append("  → Execute: python scripts/run_benchmark.py")

    model_ok = (
        Path("models/ppo_semaforo_final.zip").exists()
        or Path("models/best/best_model.zip").exists()
    )
    if not model_ok:
        erros.append("Modelo treinado não encontrado em models/")
        erros.append("  → Execute: python scripts/train_agent.py")

    if erros:
        print("\nERRO — pré-requisitos não atendidos:\n")
        for e in erros:
            print(f"  {e}")
        print()
        sys.exit(1)

    print("\nTodos os pré-requisitos encontrados. Iniciando avaliação...\n")

    avaliar()


if __name__ == "__main__":
    main()
