"""Utilitário para carregar arquivos YAML de configuração."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Lê um arquivo YAML e retorna o conteúdo como dicionário.

    Parâmetros:
    - path: caminho para o arquivo .yaml

    Retorna: dicionário com o conteúdo do YAML
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_all_configs(configs_dir: str | Path) -> dict[str, Any]:
    """
    Carrega config.yaml, scenarios.yaml e rl.yaml de um diretório.

    Parâmetros:
    - configs_dir: diretório contendo os três arquivos YAML

    Retorna: dict com chaves 'config', 'scenarios', 'rl'
    """
    base = Path(configs_dir)
    return {
        "config": load_config(base / "config.yaml"),
        "scenarios": load_config(base / "scenarios.yaml"),
        "rl": load_config(base / "rl.yaml"),
    }
