"""Gerador de cenários sintéticos para o semáforo inteligente."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_TICKS_PER_HOUR = 720  # 3600s / 5s por tick


class ScenarioGenerator:
    """
    Gera cenários sintéticos de chegadas para o cruzamento.

    Camada 1 — Perfil-base: taxas por (tipo de dia × faixa horária × fila).
    Camada 2 — Família: multiplicadores veh/ped/var aplicados sobre o perfil-base.

    Fórmula por tick:
        lambda = base × mult × max(0, 1 + N(0, variability_base_std × var_mult))
        chegadas ~ Poisson(lambda)

    Todos os parâmetros calibráveis vêm de scenarios.yaml — nada hard-coded aqui.
    """

    def __init__(self, scenarios_cfg: dict[str, Any]) -> None:
        self._cfg = scenarios_cfg
        gen = scenarios_cfg["generation"]
        self._variability_base_std: float = float(gen["variability_base_std"])
        self._ticks_per_day: int = int(gen["ticks_per_day"])
        self._generator_version: str = str(gen["generator_version"])
        self._tick_to_band: list[str] = self._build_tick_to_band()

    # ------------------------------------------------------------------
    # Público
    # ------------------------------------------------------------------

    def generate(
        self,
        family: str,
        day_type: str,
        seed: int,
        output_dir: Path | str,
    ) -> tuple[Path, Path]:
        """
        Gera um cenário completo (1 dia = 17 280 ticks) e salva CSV + JSON.

        Parâmetros:
        - family: nome da família definida em scenarios.yaml
        - day_type: 'util' ou 'fds'
        - seed: semente do gerador aleatório (garante reprodutibilidade)
        - output_dir: diretório de saída (criado se não existir)

        Retorna: (csv_path, json_path)
        """
        rng = np.random.default_rng(seed)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        family_cfg = self._cfg["families"][family]
        veh_mult = float(family_cfg["veh_multiplier"])
        ped_mult = float(family_cfg["ped_multiplier"])
        var_mult = float(family_cfg["var_multiplier"])
        std = self._variability_base_std * var_mult

        day_rates = self._cfg["arrival_rates"][day_type]

        rows: list[dict[str, int]] = []
        for tick in range(self._ticks_per_day):
            band = self._tick_to_band[tick]
            band_rates = day_rates[band]

            lam_veh = self._draw_lambda(rng, band_rates["veh_ns"], veh_mult, std)
            lam_ped_l = self._draw_lambda(rng, band_rates["ped_l"], ped_mult, std)
            lam_ped_o = self._draw_lambda(rng, band_rates["ped_o"], ped_mult, std)

            rows.append({
                "tick": tick,
                "veh_ns": int(rng.poisson(lam_veh)),
                "ped_l": int(rng.poisson(lam_ped_l)),
                "ped_o": int(rng.poisson(lam_ped_o)),
            })

        scenario_id = f"{family}_{day_type}_seed{seed}"
        csv_path = output_dir / f"{scenario_id}.csv"
        json_path = output_dir / f"{scenario_id}.json"

        self._write_csv(csv_path, rows)
        self._write_metadata(json_path, scenario_id, family, day_type, seed)

        logger.info("Gerado: %s", scenario_id)
        return csv_path, json_path

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _build_tick_to_band(self) -> list[str]:
        """Pré-computa a faixa horária para cada tick do dia."""
        boundaries = self._cfg["time_band_boundaries"]
        sorted_bands = sorted(boundaries.items(), key=lambda x: x[1]["start"])
        result: list[str] = []
        for tick in range(self._ticks_per_day):
            hour = tick / _TICKS_PER_HOUR
            band_name = sorted_bands[-1][0]
            for name, bounds in sorted_bands:
                if bounds["start"] <= hour < bounds["end"]:
                    band_name = name
                    break
            result.append(band_name)
        return result

    @staticmethod
    def _draw_lambda(
        rng: np.random.Generator, base: float, mult: float, std: float
    ) -> float:
        """Aplica ruído gaussiano sobre a taxa base e retorna lambda >= 0."""
        noise = float(rng.normal(0.0, std)) if std > 0.0 else 0.0
        return base * mult * max(0.0, 1.0 + noise)

    def _write_csv(self, path: Path, rows: list[dict[str, int]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["tick", "veh_ns", "ped_l", "ped_o"])
            writer.writeheader()
            writer.writerows(rows)

    def _write_metadata(
        self,
        path: Path,
        scenario_id: str,
        family: str,
        day_type: str,
        seed: int,
    ) -> None:
        meta = {
            "scenario_id": scenario_id,
            "family": family,
            "day_type": day_type,
            "seed": seed,
            "ticks_per_day": self._ticks_per_day,
            "generator_version": self._generator_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
