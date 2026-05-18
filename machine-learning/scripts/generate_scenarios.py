"""Gera todos os cenários sintéticos de treino e avaliação."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generator.scenario_generator import ScenarioGenerator
from src.utils.config_loader import load_all_configs


def main() -> None:
    cfgs = load_all_configs(ROOT / "configs")
    cfg = cfgs["config"]
    scenarios_cfg = cfgs["scenarios"]

    generator = ScenarioGenerator(scenarios_cfg)
    gen_cfg = cfg["scenario_generation"]

    total = 0
    for split_name, split_cfg in gen_cfg.items():
        output_dir = ROOT / split_cfg["output_dir"]
        families: list[str] = split_cfg["families"]
        day_types: list[str] = split_cfg["day_types"]
        seeds: list[int] = split_cfg["seeds"]
        count = len(families) * len(day_types) * len(seeds)
        print(f"\n[{split_name}] {count} cenários → {output_dir}")
        for family in families:
            for day_type in day_types:
                for seed in seeds:
                    csv_path, _ = generator.generate(family, day_type, seed, output_dir)
                    print(f"  ok  {csv_path.name}")
                    total += 1

    print(f"\nTotal: {total} cenários gerados.")


if __name__ == "__main__":
    main()
