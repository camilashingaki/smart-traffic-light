"""Testes unitários do ScenarioGenerator."""

from __future__ import annotations

import filecmp
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.generator.scenario_generator import ScenarioGenerator
from src.utils.config_loader import load_all_configs

_TICKS_PER_DAY = 17_280
_CSV_COLUMNS = {"tick", "hora", "minuto", "day_type", "time_band", "family",
                "veh_ns", "ped_l", "ped_o"}
_JSON_FIELDS = {"scenario_id", "family", "day_type", "seed", "ticks_per_day",
                "generator_version", "generated_at", "resolved_rates"}
_TIME_BANDS = {
    "madrugada", "manha_tranquila", "pico_manha", "meio_manha",
    "pico_tarde", "tarde", "pico_noite", "noite",
}


@pytest.fixture(scope="module")
def scenarios_cfg() -> dict:
    return load_all_configs("configs")["scenarios"]


@pytest.fixture(scope="module")
def gen(scenarios_cfg) -> ScenarioGenerator:
    return ScenarioGenerator(scenarios_cfg)


# ── (1) Reprodutibilidade ──────────────────────────────────────────────────────

class TestReproducibility:
    @pytest.mark.parametrize("family,day_type,seed", [
        ("equilibrado", "util", 42),
        ("pico_ped",    "fds",  777),
    ])
    def test_same_seed_byte_identical(self, gen, family, day_type, seed):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            csv1, _ = gen.generate(family, day_type, seed, base / "r1")
            csv2, _ = gen.generate(family, day_type, seed, base / "r2")
            assert filecmp.cmp(csv1, csv2, shallow=False), (
                f"CSVs diferem para {family}/{day_type}/seed={seed}"
            )


# ── (2) Estrutura do CSV ───────────────────────────────────────────────────────

class TestCsvStructure:
    @pytest.fixture(scope="class")
    def sample_df(self, gen):
        with tempfile.TemporaryDirectory() as tmp:
            csv, _ = gen.generate("equilibrado", "util", seed=1, output_dir=Path(tmp))
            yield pd.read_csv(csv)

    def test_row_count(self, sample_df):
        assert len(sample_df) == _TICKS_PER_DAY

    def test_columns(self, sample_df):
        assert set(sample_df.columns) == _CSV_COLUMNS

    def test_no_nulls(self, sample_df):
        assert sample_df.isnull().sum().sum() == 0

    def test_arrivals_non_negative(self, sample_df):
        for col in ("veh_ns", "ped_l", "ped_o"):
            assert (sample_df[col] >= 0).all(), f"{col} tem valores negativos"

    def test_arrivals_integer(self, sample_df):
        for col in ("veh_ns", "ped_l", "ped_o"):
            assert pd.api.types.is_integer_dtype(sample_df[col]), f"{col} não é inteiro"

    def test_tick_sequence(self, sample_df):
        assert list(sample_df["tick"]) == list(range(_TICKS_PER_DAY))

    def test_hora_range(self, sample_df):
        assert sample_df["hora"].between(0, 23).all()

    def test_minuto_range(self, sample_df):
        assert sample_df["minuto"].between(0, 59).all()

    def test_time_band_values(self, sample_df):
        assert set(sample_df["time_band"].unique()).issubset(_TIME_BANDS)


# ── (3) Estrutura do JSON ──────────────────────────────────────────────────────

class TestJsonStructure:
    @pytest.fixture(scope="class")
    def meta(self, gen):
        with tempfile.TemporaryDirectory() as tmp:
            _, json_path = gen.generate("pico_veic", "fds", seed=99, output_dir=Path(tmp))
            yield json.loads(json_path.read_text(encoding="utf-8"))

    def test_top_level_fields(self, meta):
        assert _JSON_FIELDS.issubset(meta.keys())

    def test_field_values(self, meta):
        assert meta["family"] == "pico_veic"
        assert meta["day_type"] == "fds"
        assert meta["seed"] == 99
        assert meta["ticks_per_day"] == _TICKS_PER_DAY

    def test_scenario_id_format(self, meta):
        assert meta["scenario_id"] == "pico_veic_fds_seed99"

    def test_resolved_rates_has_all_bands(self, meta):
        assert set(meta["resolved_rates"].keys()) == _TIME_BANDS

    def test_resolved_rates_has_all_queues(self, meta):
        for band, rates in meta["resolved_rates"].items():
            assert {"veh_ns", "ped_l", "ped_o"}.issubset(rates.keys()), (
                f"Faixa '{band}' faltando filas"
            )

    def test_resolved_rates_non_negative(self, meta):
        for band, rates in meta["resolved_rates"].items():
            for q, v in rates.items():
                assert v >= 0, f"Taxa negativa: {band}/{q}={v}"


# ── (4) Faixa horária ─────────────────────────────────────────────────────────

class TestTimeBand:
    # tick = hora_h * 720  (720 ticks/hora, cada tick = 5s)
    @pytest.mark.parametrize("tick,expected_band", [
        (0,     "madrugada"),        # 0h → [0, 5)
        (3599,  "madrugada"),        # 4.998h → ainda madrugada
        (3600,  "manha_tranquila"),  # 5h exato → [5, 7)
        (5040,  "pico_manha"),       # 7h exato → [7, 10)
        (7200,  "meio_manha"),       # 10h → [10, 12)
        (8640,  "pico_tarde"),       # 12h → [12, 14)
        (10080, "tarde"),            # 14h → [14, 17)
        (12240, "pico_noite"),       # 17h → [17, 20)
        (14400, "noite"),            # 20h → [20, 24)
        (17279, "noite"),            # último tick do dia
    ])
    def test_tick_maps_to_correct_band(self, gen, tick, expected_band):
        # O atributo público é _tick_to_band (lista pré-computada no __init__)
        result = gen._tick_to_band[tick]
        assert result == expected_band, (
            f"tick={tick} → esperado '{expected_band}', obtido '{result}'"
        )

    def test_csv_contains_correct_bands_for_sampled_ticks(self, gen):
        """Integração: verifica band no CSV gerado, não só na lista interna."""
        with tempfile.TemporaryDirectory() as tmp:
            csv, _ = gen.generate("equilibrado", "util", seed=0, output_dir=Path(tmp))
            df = pd.read_csv(csv)
            checks = [
                (0,     "madrugada"),
                (5040,  "pico_manha"),
                (12240, "pico_noite"),
            ]
            for tick, expected in checks:
                row = df.loc[df["tick"] == tick, "time_band"].iloc[0]
                assert row == expected, f"tick={tick}: esperado '{expected}', obtido '{row}'"


# ── (5) Efeito dos multiplicadores de família ─────────────────────────────────

class TestFamilyMultipliers:
    SEED = 0
    DAY_TYPE = "util"

    @pytest.fixture(scope="class")
    def dfs(self, gen):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            results = {}
            for fam in ("equilibrado", "pico_veic", "pico_ped", "baixa_mov"):
                csv, _ = gen.generate(fam, self.DAY_TYPE, self.SEED, base / fam)
                results[fam] = pd.read_csv(csv)
            yield results

    def test_pico_veic_has_more_vehicles(self, dfs):
        ratio = dfs["pico_veic"]["veh_ns"].mean() / dfs["equilibrado"]["veh_ns"].mean()
        assert ratio > 1.30, f"Ratio pico_veic/equilibrado veh_ns = {ratio:.2f}, esperado > 1.30"

    def test_pico_ped_has_more_pedestrians(self, dfs):
        ped_pico = (dfs["pico_ped"]["ped_l"] + dfs["pico_ped"]["ped_o"]).mean()
        ped_eq   = (dfs["equilibrado"]["ped_l"] + dfs["equilibrado"]["ped_o"]).mean()
        ratio = ped_pico / ped_eq
        assert ratio > 1.30, f"Ratio pico_ped/equilibrado pedestres = {ratio:.2f}, esperado > 1.30"

    def test_baixa_mov_has_less_traffic(self, dfs):
        for col in ("veh_ns", "ped_l"):
            ratio = dfs["baixa_mov"][col].mean() / dfs["equilibrado"][col].mean()
            assert ratio < 0.70, f"Ratio baixa_mov/equilibrado {col} = {ratio:.2f}, esperado < 0.70"

    def test_pico_veic_has_fewer_pedestrians(self, dfs):
        ratio = dfs["pico_veic"]["ped_l"].mean() / dfs["equilibrado"]["ped_l"].mean()
        assert ratio < 0.85, f"Ratio pico_veic/equilibrado ped_l = {ratio:.2f}, esperado < 0.85"


# ── (6) Seeds diferentes produzem CSVs diferentes ────────────────────────────

class TestSeedIsolation:
    @pytest.mark.parametrize("family,day_type,s1,s2", [
        ("equilibrado", "util", 1, 2),
        ("pico_ped",    "fds",  100, 101),
    ])
    def test_different_seeds_differ(self, gen, family, day_type, s1, s2):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            csv1, _ = gen.generate(family, day_type, s1, base / "a")
            csv2, _ = gen.generate(family, day_type, s2, base / "b")
            h1 = hashlib.sha256(csv1.read_bytes()).hexdigest()
            h2 = hashlib.sha256(csv2.read_bytes()).hexdigest()
            assert h1 != h2, (
                f"Seeds {s1} e {s2} produziram CSVs idênticos para {family}/{day_type}"
            )
