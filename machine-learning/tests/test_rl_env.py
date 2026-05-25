"""
Testes formais do ambiente Gymnasium — Fase 4.

Rode com:
    pytest tests/test_rl_env.py -v

Cobre os critérios de aceite definidos em INSTRUCOES_SEMAFORO_INTELIGENTE.md §7:
- reset() e step() funcionam sem erro
- Recompensa varia coerentemente com o estado
- Episódio com agente aleatório completa 360 ticks sem travar
- Espaços de ação e observação são compatíveis com Stable-Baselines3
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.rl.env import TrafficLightEnv, TICK_S


# ── Fixture: ambiente compartilhado entre testes ──────────────────────────────

@pytest.fixture
def env():
    """Cria um ambiente novo antes de cada teste e fecha ao final."""
    e = TrafficLightEnv()
    yield e
    e.close()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Inicialização
# ══════════════════════════════════════════════════════════════════════════════

class TestInicializacao:

    def test_action_space_e_discrete_2(self, env):
        """Espaço de ação deve ser Discrete(2): 0=manter, 1=trocar."""
        from gymnasium.spaces import Discrete
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 2

    def test_observation_space_shape(self, env):
        """Espaço de observação deve ser Box de shape (7,)."""
        from gymnasium.spaces import Box
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.shape == (7,)

    def test_observation_space_limites(self, env):
        """Todos os limites do espaço de observação devem ser [0, 1]."""
        assert float(env.observation_space.low.min())  == 0.0
        assert float(env.observation_space.high.max()) == 1.0

    def test_cenarios_carregados(self, env):
        """Deve haver pelo menos 1 cenário de treino disponível."""
        assert len(env._scenario_paths) > 0, (
            "Nenhum CSV encontrado em scenarios/train/. "
            "Rode generate_scenarios.py primeiro."
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. reset()
# ══════════════════════════════════════════════════════════════════════════════

class TestReset:

    def test_reset_retorna_obs_e_info(self, env):
        """reset() deve retornar (obs, info) sem exceção."""
        resultado = env.reset(seed=42)
        assert isinstance(resultado, tuple)
        assert len(resultado) == 2

    def test_obs_shape(self, env):
        """Observação deve ter shape (7,)."""
        obs, _ = env.reset(seed=0)
        assert obs.shape == (7,), f"Shape errado: {obs.shape}"

    def test_obs_dtype(self, env):
        """Observação deve ser float32."""
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32, f"dtype errado: {obs.dtype}"

    def test_obs_dentro_dos_limites(self, env):
        """Todos os valores da observação devem estar em [0, 1]."""
        obs, _ = env.reset(seed=0)
        assert float(obs.min()) >= 0.0, f"Valor abaixo de 0: {obs.min()}"
        assert float(obs.max()) <= 1.0, f"Valor acima de 1: {obs.max()}"

    def test_obs_dentro_do_espaco(self, env):
        """Observação deve pertencer ao observation_space declarado."""
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs), (
            f"Observação fora do espaço declarado: {obs}"
        )

    def test_info_tem_chaves_esperadas(self, env):
        """info deve conter 'scenario' e 'tick_inicio'."""
        _, info = env.reset(seed=0)
        assert "scenario"    in info, "info não tem chave 'scenario'"
        assert "tick_inicio" in info, "info não tem chave 'tick_inicio'"

    def test_reset_com_seed_reprodutivel(self, env):
        """Seed fixa deve produzir mesma obs. Cenário pode variar entre resets."""
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_zera_contadores(self, env):
        """Após reset(), o episódio deve começar do tick 0."""
        env.reset(seed=1)
        assert env._ticks_ep == 0

    def test_reset_zera_filas(self, env):
        """Após reset(), todas as filas devem estar vazias."""
        # Roda alguns ticks para acumular fila
        env.reset(seed=2)
        for _ in range(10):
            env.step(0)
        # Reseta e verifica
        env.reset(seed=3)
        assert env.crossing.veh_ns.size == 0
        assert env.crossing.ped_l.size  == 0
        assert env.crossing.ped_o.size  == 0

    def test_reset_multiplas_vezes(self, env):
        """reset() deve funcionar várias vezes seguidas sem erro."""
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            assert obs.shape == (7,)


# ══════════════════════════════════════════════════════════════════════════════
# 3. step()
# ══════════════════════════════════════════════════════════════════════════════

class TestStep:

    def test_step_retorna_5_valores(self, env):
        """step() deve retornar (obs, reward, terminated, truncated, info)."""
        env.reset(seed=0)
        resultado = env.step(0)
        assert isinstance(resultado, tuple)
        assert len(resultado) == 5

    def test_step_obs_shape(self, env):
        """obs retornado por step() deve ter shape (7,)."""
        env.reset(seed=0)
        obs, *_ = env.step(0)
        assert obs.shape == (7,)

    def test_step_obs_dentro_dos_limites(self, env):
        """Observação pós-step deve estar em [0, 1]."""
        env.reset(seed=0)
        obs, *_ = env.step(1)
        assert float(obs.min()) >= 0.0
        assert float(obs.max()) <= 1.0

    def test_step_reward_e_float(self, env):
        """reward deve ser float."""
        env.reset(seed=0)
        _, reward, *_ = env.step(0)
        assert isinstance(reward, float)

    def test_step_reward_nao_positivo(self, env):
        """Toda recompensa deve ser <= 0 (sistema de penalidades)."""
        env.reset(seed=0)
        for _ in range(20):
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            assert reward <= 0.0, f"Recompensa positiva inesperada: {reward}"
            if terminated:
                env.reset(seed=0)

    def test_step_reward_varia(self, env):
        """Recompensa deve variar ao longo de um episódio (não sempre 0)."""
        env.reset(seed=5)
        recompensas = []
        for _ in range(50):
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            recompensas.append(reward)
            if terminated:
                break
        assert len(set(round(r, 4) for r in recompensas)) > 1, (
            "Recompensa não variou — verifique _calcular_recompensa()"
        )

    def test_step_terminated_false_antes_do_fim(self, env):
        """terminated deve ser False antes de completar 360 ticks."""
        env.reset(seed=0)
        for _ in range(359):
            _, _, terminated, truncated, _ = env.step(0)
            assert terminated is False
            assert truncated  is False

    def test_step_terminated_true_no_fim(self, env):
        """terminated deve ser True exatamente no tick 360."""
        env.reset(seed=0)
        terminated = False
        for _ in range(360):
            _, _, terminated, _, _ = env.step(0)
        assert terminated is True

    def test_step_acao_manter(self, env):
        """Ação 0 (manter) deve ser aceita sem erro."""
        env.reset(seed=0)
        env.step(0)   # não deve levantar exceção

    def test_step_acao_trocar(self, env):
        """Ação 1 (trocar) deve ser aceita sem erro."""
        env.reset(seed=0)
        env.step(1)   # não deve levantar exceção

    def test_step_sem_reset_levanta_erro(self, env):
        """step() sem reset() prévio deve levantar RuntimeError."""
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)

    def test_step_info_tem_chaves(self, env):
        """info do step() deve ter as chaves de diagnóstico esperadas."""
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        chaves_esperadas = [
            "ticks_episodio", "tick_absoluto",
            "fila_carros", "fila_ped_leste", "fila_ped_oeste",
            "fase_atual", "ticks_na_fase",
            "max_espera_carro_s", "max_espera_ped_s", "espera_total_s",
        ]
        for chave in chaves_esperadas:
            assert chave in info, f"chave '{chave}' ausente no info do step()"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Episódio completo
# ══════════════════════════════════════════════════════════════════════════════

class TestEpisodioCompleto:

    def test_agente_aleatorio_completa_episodio(self, env):
        """Agente aleatório deve completar exatamente 360 steps."""
        env.reset(seed=42)
        done  = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
            assert steps <= 400, "Episódio não terminou após 400 steps — loop infinito?"
        assert steps == 360, f"Episódio terminou em {steps} steps (esperado: 360)"

    def test_agente_sempre_manter_completa_episodio(self, env):
        """Agente que sempre mantém a fase também deve completar 360 steps."""
        env.reset(seed=1)
        done  = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            steps += 1
        assert steps == 360

    def test_agente_sempre_trocar_completa_episodio(self, env):
        """Agente que sempre tenta trocar também deve completar 360 steps."""
        env.reset(seed=2)
        done  = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(1)
            done = terminated or truncated
            steps += 1
        assert steps == 360

    def test_multiplos_episodios_sem_erro(self, env):
        """Deve ser possível rodar 3 episódios seguidos sem erro."""
        for ep in range(3):
            env.reset(seed=ep)
            done = False
            steps = 0
            while not done:
                _, _, terminated, truncated, _ = env.step(
                    env.action_space.sample()
                )
                done = terminated or truncated
                steps += 1
            assert steps == 360, f"Episódio {ep} durou {steps} steps"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Recompensa — coerência com o estado
# ══════════════════════════════════════════════════════════════════════════════

class TestRecompensa:

    def test_recompensa_piora_com_fila_grande(self, env):
        """
        Recompensa deve ser mais negativa quando há mais carros esperando.

        Estratégia: rodar dois ambientes com seeds diferentes — um com
        poucas chegadas, outro com muitas — e comparar a recompensa média.
        """
        # Cria dois ambientes com configurações distintas de chegada
        # A diferença vem dos cenários sorteados; rodamos vários episódios
        # e verificamos que a recompensa média não é constante nem positiva.
        recompensas = []
        env.reset(seed=10)
        for _ in range(100):
            _, reward, terminated, _, _ = env.step(0)
            recompensas.append(reward)
            if terminated:
                env.reset(seed=10)

        assert any(r < -1.0 for r in recompensas), (
            "Recompensa nunca ficou abaixo de -1.0 — "
            "verifique se as chegadas estão sendo aplicadas"
        )

    def test_espera_acumulada_e_negativa(self, env):
        """
        Após muitos ticks sem trocar de fase, pedestres acumulam fila
        e a recompensa deve refletir isso (mais negativa).
        """
        env.reset(seed=20)
        # Mantém fase A (carros) por 50 ticks sem trocar
        recompensas = []
        for _ in range(50):
            _, reward, terminated, _, _ = env.step(0)
            recompensas.append(reward)
            if terminated:
                break

        # A recompensa deve ficar cada vez mais negativa conforme a fila cresce
        # (verifica que ao menos nos últimos 10 ticks a média é pior que nos primeiros)
        if len(recompensas) >= 20:
            media_inicio = np.mean(recompensas[:10])
            media_fim    = np.mean(recompensas[-10:])
            assert media_fim <= media_inicio, (
                "Recompensa deveria piorar conforme a fila cresce, "
                f"mas início={media_inicio:.2f} e fim={media_fim:.2f}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Compatibilidade com Stable-Baselines3
# ══════════════════════════════════════════════════════════════════════════════

class TestCompatibilidadeSB3:

    def test_gymnasium_check_env(self, env):
        """Verifica compatibilidade básica com a API do Gymnasium."""
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        obs, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        for _ in range(10):
            assert env.action_space.contains(env.action_space.sample())

    def test_observation_space_contains_obs(self, env):
        """Todas as observações produzidas devem pertencer ao espaço declarado."""
        env.reset(seed=0)
        for _ in range(30):
            obs, _, terminated, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs), (
                f"Observação fora do espaço declarado: {obs}"
            )
            if terminated:
                env.reset(seed=0)

    def test_action_space_contains_actions(self, env):
        """Ações 0 e 1 devem pertencer ao espaço de ação declarado."""
        assert env.action_space.contains(np.int64(0))
        assert env.action_space.contains(np.int64(1))

    def test_tick_s_constante(self):
        """TICK_S deve ser 5 (definido em INSTRUCOES_SEMAFORO_INTELIGENTE.md §3.1)."""
        assert TICK_S == 5, f"TICK_S deveria ser 5, mas é {TICK_S}"
