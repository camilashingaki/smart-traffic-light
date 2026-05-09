"""Testes do carregador de configuração YAML."""

from pathlib import Path

import pytest

from src.utils.config_loader import load_config, load_all_configs

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_load_config_yaml():
    cfg = load_config(CONFIGS_DIR / "config.yaml")
    assert "simulation" in cfg
    assert cfg["simulation"]["tick_seconds"] == 5
    assert cfg["simulation"]["min_green_cars_ticks"] == 3
    assert cfg["simulation"]["min_green_pedestrians_ticks"] == 2


def test_load_rl_yaml():
    cfg = load_config(CONFIGS_DIR / "rl.yaml")
    assert "reward_weights" in cfg
    assert cfg["reward_weights"]["excedeu_teto"] == -10.0
    assert cfg["teto_espera_carros"] == 90


def test_load_scenarios_yaml():
    cfg = load_config(CONFIGS_DIR / "scenarios.yaml")
    assert "arrival_rates" in cfg
    assert "families" in cfg
    assert "equilibrado" in cfg["families"]
    assert "time_band_boundaries" in cfg


def test_load_all_configs():
    all_cfg = load_all_configs(CONFIGS_DIR)
    assert "config" in all_cfg
    assert "scenarios" in all_cfg
    assert "rl" in all_cfg


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_config(CONFIGS_DIR / "nao_existe.yaml")
