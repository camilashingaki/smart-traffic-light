"""
Smoke test do ambiente Gymnasium — Fase 4.

Rode com:
    python scripts/test_env_smoke.py

Não usa pytest. É um script simples para verificar rapidamente se o
ambiente está funcionando antes de rodar os testes formais.

Critérios de aceite da Fase 4:
    [1] reset() retorna observação de shape (7,) com valores em [0, 1]
    [2] step() retorna recompensa que varia (não fica sempre 0.0)
    [3] Episódio completo com agente aleatório termina em exatamente 360 steps
"""

import sys
import logging

import numpy as np

# Configura logging para ver o que está acontecendo
logging.basicConfig(level=logging.WARNING)

# Importa o ambiente
try:
    from src.rl.env import TrafficLightEnv
except ImportError as e:
    print(f"ERRO ao importar TrafficLightEnv: {e}")
    print("Verifique se o venv está ativado e se está rodando da pasta machine-learning/")
    sys.exit(1)


def separador(titulo: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {titulo}")
    print('─' * 50)


def checar(condicao: bool, msg_ok: str, msg_erro: str) -> bool:
    if condicao:
        print(f"  ✓  {msg_ok}")
        return True
    else:
        print(f"  ✗  {msg_erro}")
        return False


def main() -> None:
    erros = 0

    # ── Cria o ambiente ───────────────────────────────────────────────────
    separador("Inicializando ambiente")
    try:
        env = TrafficLightEnv()
        print("  ✓  TrafficLightEnv criado com sucesso")
    except Exception as e:
        print(f"  ✗  Falha ao criar o ambiente: {e}")
        sys.exit(1)

    # ── Teste 1: reset() ──────────────────────────────────────────────────
    separador("Teste 1 — reset()")
    try:
        obs, info = env.reset(seed=42)

        erros += not checar(
            isinstance(obs, np.ndarray),
            "obs é np.ndarray",
            f"obs deveria ser np.ndarray, mas é {type(obs)}",
        )
        erros += not checar(
            obs.shape == (7,),
            f"obs.shape == (7,)  →  {obs}",
            f"obs.shape errado: {obs.shape} (esperado (7,))",
        )
        erros += not checar(
            obs.dtype == np.float32,
            "obs.dtype == float32",
            f"obs.dtype errado: {obs.dtype}",
        )
        erros += not checar(
            float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0,
            f"todos os valores em [0, 1]  →  min={obs.min():.3f}  max={obs.max():.3f}",
            f"valores fora de [0, 1]: min={obs.min():.3f}  max={obs.max():.3f}",
        )
        erros += not checar(
            isinstance(info, dict),
            f"info é dict com chaves: {list(info.keys())}",
            f"info deveria ser dict, mas é {type(info)}",
        )
    except Exception as e:
        print(f"  ✗  reset() levantou exceção: {e}")
        erros += 1

    # ── Teste 2: step() com 10 ações aleatórias ───────────────────────────
    separador("Teste 2 — step() com agente aleatório (10 ticks)")
    try:
        obs, _ = env.reset(seed=0)
        recompensas = []

        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            recompensas.append(reward)

            erros += not checar(
                obs.shape == (7,),
                f"tick {i+1:2d}: shape ok  reward={reward:.2f}  fase={info.get('fase_atual','?')}  "
                f"carros={info.get('fila_carros', '?')}  ped={info.get('fila_ped_leste','?')}+{info.get('fila_ped_oeste','?')}",
                f"tick {i+1}: obs.shape errado: {obs.shape}",
            )
            erros += not checar(
                isinstance(reward, float),
                "",
                f"tick {i+1}: reward deveria ser float, mas é {type(reward)}",
            )

        erros += not checar(
            len(set(round(r, 4) for r in recompensas)) > 1,
            f"recompensa varia entre ticks  →  min={min(recompensas):.2f}  max={max(recompensas):.2f}",
            "recompensa NÃO variou — verifique a função _calcular_recompensa()",
        )
        erros += not checar(
            all(r <= 0.0 for r in recompensas),
            "todas as recompensas são <= 0 (correto para penalidades)",
            f"alguma recompensa foi positiva: {[r for r in recompensas if r > 0]}",
        )
    except Exception as e:
        print(f"  ✗  step() levantou exceção: {e}")
        erros += 1

    # ── Teste 3: episódio completo ────────────────────────────────────────
    separador("Teste 3 — episódio completo com agente aleatório")
    try:
        obs, _ = env.reset(seed=7)
        done  = False
        steps = 0
        recompensa_total = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            recompensa_total += reward
            done = terminated or truncated
            steps += 1

            # Proteção contra loop infinito
            if steps > 10_000:
                print("  ✗  Episódio não terminou após 10.000 steps — loop infinito?")
                erros += 1
                break

        erros += not checar(
            steps == 360,
            f"episódio terminou em {steps} steps (esperado: 360)",
            f"episódio terminou em {steps} steps (esperado: 360)",
        )
        erros += not checar(
            terminated is True,
            f"terminated=True ao final  →  recompensa_total={recompensa_total:.1f}",
            "terminated deveria ser True ao final do episódio",
        )
    except Exception as e:
        print(f"  ✗  episódio completo levantou exceção: {e}")
        erros += 1

    # ── Resultado final ───────────────────────────────────────────────────
    separador("Resultado")
    if erros == 0:
        print("  PASSOU — Fase 4 pronta para pytest e commit.\n")
        sys.exit(0)
    else:
        print(f"  FALHOU — {erros} erro(s) encontrado(s). Corrija antes de avançar.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
