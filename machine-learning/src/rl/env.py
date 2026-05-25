"""
Ambiente Gymnasium para o projeto de Semáforo Inteligente.

Envolve a simulação do cruzamento (crossing.py) numa interface padrão
do Gymnasium, permitindo que o algoritmo PPO (Stable-Baselines3)
interaja com ela.

Conceitos-chave (ver INSTRUCOES_SEMAFORO_INTELIGENTE.md §3):
- Tick       : 1 unidade de tempo = 5 segundos reais
- Episódio   : 30 min simulados = 360 ticks (durante treino)
- Fase A     : verde para carros (N→S), vermelho para pedestres
- Fase B     : verde para pedestres (L↔O), vermelho para carros
- Ação 0     : manter a fase atual
- Ação 1     : solicitar troca de fase (ignorada se tempo mínimo não atingido)

API real do Crossing (crossing.py):
- crossing.step(arrivals, action) executa TODO o tick de uma vez.
  arrivals = {'veh_ns': int, 'ped_l': int, 'ped_o': int}
  action   = 0 ou 1
- As filas são objetos TrafficQueue; tamanho = .size, espera = .max_wait_ticks
- Os tempos de espera estão em TICKS — multiplicar por 5 para converter em segundos
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.simulation.crossing import Crossing, Phase
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Fator de conversão: ticks → segundos
TICK_S: int = 5


class TrafficLightEnv(gym.Env):
    """
    Ambiente Gymnasium que envolve a simulação do cruzamento.

    Parâmetros
    ----------
    config_path : str
        Caminho para configs/config.yaml.
    rl_config_path : str
        Caminho para configs/rl.yaml.
    scenarios_dir : str
        Pasta com os cenários de treino (scenarios/train/).
    render_mode : str | None
        None desativa a renderização (recomendado durante treino).

    Espaço de ação
    --------------
    Discrete(2)
        0 = manter fase atual
        1 = solicitar troca de fase

    Espaço de observação
    --------------------
    Box(shape=(7,), low=0.0, high=1.0, dtype=float32)
        Vetor normalizado com o estado atual do cruzamento.
        Ver _get_obs() para descrição de cada dimensão.
    """

    # Número de dimensões do vetor de observação
    OBS_DIM: int = 7

    # Denominadores de normalização de cada componente.
    # Valores máximos razoáveis para o cruzamento modelado.
    _NORM = {
        "veh_ns":          30.0,  # carros na fila N→S
        "ped_l":           15.0,  # pedestres lado leste
        "ped_o":           15.0,  # pedestres lado oeste
        "ticks_in_phase":  36.0,  # ticks na fase atual (max razoável = 3 min)
        "max_wait_car_s":  90.0,  # maior espera carro em segundos (= teto)
        "max_wait_ped_s":  60.0,  # maior espera pedestre em segundos (= teto)
    }

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        rl_config_path: str = "configs/rl.yaml",
        scenarios_dir: str = "scenarios/train",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        # ── Carrega configurações ──────────────────────────────────────────
        _full_cfg        = load_config(config_path)
        self._cfg        = _full_cfg["simulation"]   # passa só 'simulation' para o Crossing
        self._rl_cfg     = load_config(rl_config_path)
        self.render_mode = render_mode

        # Pesos da recompensa vindos do rl.yaml
        self._w = self._rl_cfg["reward_weights"]

        # Tetos de espera em segundos
        self._teto_car_s: int = self._rl_cfg["teto_espera_carros"]    # 90
        self._teto_ped_s: int = self._rl_cfg["teto_espera_pedestres"] # 60

        # Duração do episódio de treino em ticks — chave correta do config.yaml
        self._ep_ticks: int = _full_cfg["training"]["episode_ticks"]  # 360

        # ── Espaço de ação: 0 = manter, 1 = trocar ────────────────────────
        self.action_space = spaces.Discrete(2)

        # ── Espaço de observação: 7 floats em [0, 1] ──────────────────────
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

        # ── Cruzamento (engine de simulação) ──────────────────────────────
        self.crossing = Crossing(cfg=self._cfg)

        # ── Cenários de treino disponíveis ────────────────────────────────
        self._scenarios_dir  = Path(scenarios_dir)
        self._scenario_paths = sorted(self._scenarios_dir.glob("*.csv"))
        if not self._scenario_paths:
            raise FileNotFoundError(
                f"Nenhum cenário CSV encontrado em '{scenarios_dir}'. "
                "Execute scripts/generate_scenarios.py antes de treinar."
            )
        logger.info(
            "TrafficLightEnv: %d cenários de treino carregados.",
            len(self._scenario_paths),
        )

        # ── Estado interno do episódio (inicializado em reset) ─────────────
        self._scenario_df: pd.DataFrame | None = None
        self._tick_inicio: int = 0   # linha do CSV onde o episódio começa
        self._tick_atual:  int = 0   # linha do CSV sendo lida agora
        self._ticks_ep:    int = 0   # quantos ticks já rodaram neste episódio

    # ══════════════════════════════════════════════════════════════════════
    # reset() — inicia um novo episódio
    # ══════════════════════════════════════════════════════════════════════

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reinicia o ambiente para o início de um novo episódio.

        Sorteia um cenário de treino aleatório e um ponto de início
        aleatório dentro do dia (garante que o episódio de 360 ticks
        caiba inteiramente no CSV).

        Parâmetros
        ----------
        seed : int | None
            Semente para reprodutibilidade.
        options : dict | None
            Ignorado; existe para compatibilidade com a API do Gymnasium.

        Retorna
        -------
        obs : np.ndarray, shape (7,)
            Observação inicial normalizada.
        info : dict
            Nome do cenário sorteado e tick de início.
        """
        super().reset(seed=seed)

        # Sorteia cenário aleatório
        idx  = self.np_random.integers(0, len(self._scenario_paths))
        path = self._scenario_paths[idx]
        self._scenario_df = pd.read_csv(path)

        # Sorteia ponto de início no dia (o episódio precisa caber no CSV)
        max_inicio = max(0, len(self._scenario_df) - self._ep_ticks)
        self._tick_inicio = int(self.np_random.integers(0, max_inicio + 1))
        self._tick_atual  = self._tick_inicio
        self._ticks_ep    = 0

        # Reinicia o cruzamento com filas vazias e fase A
        self.crossing.reset()

        obs  = self._get_obs()
        info = {
            "scenario":    path.name,
            "tick_inicio": self._tick_inicio,
        }

        logger.debug(
            "reset(): cenário=%s  início=tick %d",
            path.name,
            self._tick_inicio,
        )
        return obs, info

    # ══════════════════════════════════════════════════════════════════════
    # step() — avança 1 tick
    # ══════════════════════════════════════════════════════════════════════

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Avança o ambiente em 1 tick (5 segundos simulados).

        A chamada a crossing.step() executa todas as operações do tick
        internamente, na ordem correta:
          1. Aplica troca de fase pendente (se houver)
          2. Registra a nova ação (vale no próximo tick)
          3. Adiciona chegadas às filas
          4. Escoa entidades conforme fase ativa
          5. Incrementa espera de quem ficou
          6. Avança contadores

        Ações inválidas (trocar antes do tempo mínimo de verde) são
        ignoradas silenciosamente pelo crossing — sem penalidade.

        Parâmetros
        ----------
        action : int
            0 = manter fase atual.
            1 = solicitar troca de fase.

        Retorna
        -------
        obs        : np.ndarray, shape (7,)  — novo estado
        reward     : float                   — recompensa (sempre <= 0)
        terminated : bool                    — True ao completar 360 ticks
        truncated  : bool                    — sempre False
        info       : dict                    — métricas de diagnóstico
        """
        if self._scenario_df is None:
            raise RuntimeError("Chame reset() antes de step().")

        # Lê as chegadas do tick atual no CSV do cenário
        linha = self._scenario_df.iloc[self._tick_atual]
        arrivals = {
            "veh_ns": int(linha["veh_ns"]),
            "ped_l":  int(linha["ped_l"]),
            "ped_o":  int(linha["ped_o"]),
        }

        # Executa o tick inteiro de uma vez (API real do crossing)
        self.crossing.step(arrivals=arrivals, action=int(action))

        # Avança contadores internos do episódio
        self._tick_atual += 1
        self._ticks_ep   += 1

        # Calcula recompensa com o estado pós-tick
        reward = self._calcular_recompensa()

        # Episódio termina ao atingir a duração configurada
        terminated = self._ticks_ep >= self._ep_ticks
        truncated  = False

        obs  = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════════
    # render() e close()
    # ══════════════════════════════════════════════════════════════════════

    def render(self) -> None:
        """
        Renderização opcional. Não faz nada durante o treino (render_mode=None).
        A integração com o Pygame é feita externamente pelo SimulationLoop.
        """
        if self.render_mode == "human":
            logger.debug("render() chamado; renderer externo deve estar conectado.")

    def close(self) -> None:
        """Libera recursos ao encerrar o ambiente."""
        logger.debug("TrafficLightEnv encerrado.")

    # ══════════════════════════════════════════════════════════════════════
    # Métodos privados
    # ══════════════════════════════════════════════════════════════════════

    def _get_obs(self) -> np.ndarray:
        """
        Monta o vetor de observação normalizado para [0, 1].

        Dimensões do vetor:
          [0] fila de carros N→S          ÷ 30
          [1] fila de pedestres leste     ÷ 15
          [2] fila de pedestres oeste     ÷ 15
          [3] fase atual                  0.0 = Fase A  /  1.0 = Fase B
          [4] ticks na fase atual         ÷ 36
          [5] maior espera de carro (s)   ÷ 90  (teto)
          [6] maior espera de ped. (s)    ÷ 60  (teto)

        Tempos de espera: as filas guardam valores em ticks → convertidos
        para segundos multiplicando por TICK_S (5) antes de normalizar.

        Valores acima de 1.0 são truncados com np.clip para não violar
        o espaço de observação declarado.

        Retorna
        -------
        obs : np.ndarray, shape (7,), dtype float32
        """
        c = self.crossing
        n = self._NORM

        # Tempos de espera máximos convertidos de ticks para segundos
        max_wait_car_s = c.veh_ns.max_wait_ticks * TICK_S
        max_wait_ped_s = max(c.ped_l.max_wait_ticks, c.ped_o.max_wait_ticks) * TICK_S

        obs = np.array([
            c.veh_ns.size         / n["veh_ns"],
            c.ped_l.size          / n["ped_l"],
            c.ped_o.size          / n["ped_o"],
            float(c.current_phase == Phase.B),   # 0.0 = Fase A, 1.0 = Fase B
            c.ticks_in_phase      / n["ticks_in_phase"],
            max_wait_car_s        / n["max_wait_car_s"],
            max_wait_ped_s        / n["max_wait_ped_s"],
        ], dtype=np.float32)

        # Garante que nenhum valor saia do intervalo [0, 1]
        return np.clip(obs, 0.0, 1.0)

    def _calcular_recompensa(self) -> float:
        """
        Calcula a recompensa do tick atual.

        Todos os pesos são negativos — o agente aprende a maximizar a
        recompensa, o que equivale a minimizar as penalidades.

        Tempos de espera: as filas guardam ticks → convertidos para
        segundos antes de aplicar os pesos, mantendo as unidades
        coerentes com os tetos definidos em rl.yaml.

        Componentes (pesos em self._w, lidos de rl.yaml):
          espera_acumulada : soma das esperas de TODOS na fila (em s)
          tamanho_filas    : número total de agentes esperando agora
          max_espera       : maior espera individual (em s)
          desequilibrio    : |fila_carros − fila_pedestres|
          excedeu_teto     : número de agentes acima do teto de espera

        Retorna
        -------
        reward : float  (sempre <= 0)
        """
        c = self.crossing
        w = self._w

        # Espera acumulada total de todos os agentes, em segundos
        espera_acumulada = (
            c.veh_ns.total_wait_ticks
            + c.ped_l.total_wait_ticks
            + c.ped_o.total_wait_ticks
        ) * TICK_S

        # Tamanho total das filas neste tick
        tamanho_filas = c.veh_ns.size + c.ped_l.size + c.ped_o.size

        # Maior espera individual (carros ou pedestres), em segundos
        max_espera = max(
            c.veh_ns.max_wait_ticks,
            c.ped_l.max_wait_ticks,
            c.ped_o.max_wait_ticks,
        ) * TICK_S

        # Desequilíbrio entre carga de veículos e pedestres
        carga_veic    = c.veh_ns.size
        carga_ped     = c.ped_l.size + c.ped_o.size
        desequilibrio = abs(carga_veic - carga_ped)

        # Agentes acima do teto de espera (penalidade severa)
        teto_car_ticks = self._teto_car_s // TICK_S   # 90s ÷ 5 = 18 ticks
        teto_ped_ticks = self._teto_ped_s // TICK_S   # 60s ÷ 5 = 12 ticks

        violacoes = sum(
            1 for t in c.veh_ns.snapshot() if t > teto_car_ticks
        ) + sum(
            1 for t in c.ped_l.snapshot() if t > teto_ped_ticks
        ) + sum(
            1 for t in c.ped_o.snapshot() if t > teto_ped_ticks
        )

        reward = (
            w["espera_acumulada"] * espera_acumulada
            + w["tamanho_filas"]  * tamanho_filas
            + w["max_espera"]     * max_espera
            + w["desequilibrio"]  * desequilibrio
            + w["excedeu_teto"]   * violacoes
        )

        return float(reward)

    def _get_info(self) -> dict[str, Any]:
        """
        Retorna métricas de diagnóstico do tick atual.

        Não afeta o agente — usado apenas para monitoramento e logging.

        Retorna
        -------
        info : dict com métricas do cruzamento e do episódio.
        """
        c = self.crossing
        return {
            # Posição no episódio
            "ticks_episodio":     self._ticks_ep,
            "tick_absoluto":      self._tick_atual,
            # Filas
            "fila_carros":        c.veh_ns.size,
            "fila_ped_leste":     c.ped_l.size,
            "fila_ped_oeste":     c.ped_o.size,
            # Semáforo
            "fase_atual":         c.current_phase.value,
            "ticks_na_fase":      c.ticks_in_phase,
            # Esperas em segundos
            "max_espera_carro_s": c.veh_ns.max_wait_ticks * TICK_S,
            "max_espera_ped_s":   max(
                c.ped_l.max_wait_ticks, c.ped_o.max_wait_ticks
            ) * TICK_S,
            "espera_total_s": (
                c.veh_ns.total_wait_ticks
                + c.ped_l.total_wait_ticks
                + c.ped_o.total_wait_ticks
            ) * TICK_S,
        }
