# Semáforo Inteligente

Agente de Reinforcement Learning para controle adaptativo de semáforo em cruzamento simulado.

## Objetivo

Reduzir o tempo médio e máximo de espera de carros e pedestres em comparação com um semáforo de tempo fixo (benchmark), usando PPO (Stable-Baselines3).

## Layout do cruzamento

- Rua única N→S com 2 faixas paralelas
- Travessia de pedestres L↔O
- Fase A: verde carros / Fase B: verde pedestres

## Stack

Python 3.10+ · Pygame · Matplotlib · Stable-Baselines3 · Gymnasium · PyYAML · Pandas · NumPy

## Instalação

```bash
pip install -r requirements.txt
```

## Estrutura

```
configs/    # Parâmetros YAML (não edite o código para mudar valores)
src/        # Código-fonte principal
scripts/    # Scripts de execução
tests/      # Testes unitários
results/    # Gráficos e relatórios (versionados)
scenarios/  # CSVs gerados (gitignored)
models/     # Checkpoints de treino (gitignored)
```

## Fases do projeto

1. Setup e esqueleto ← *atual*
2. Visualização Pygame + benchmark de tempo fixo
3. Gerador de cenários completo
4. Ambiente Gymnasium
5. Treinamento PPO
6. Avaliação e comparação
