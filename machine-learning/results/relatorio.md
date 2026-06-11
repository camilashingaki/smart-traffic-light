# Relatório Final — Semáforo Inteligente

> **Agente:** PPO (Stable-Baselines3)  
> **Cenários de avaliação:** 32  
> **Tetos:** carros ≤ 90 s | pedestres ≤ 60 s  

---

## 1. Resumo executivo

O agente PPO não superou o semáforo de tempo fixo em espera média de carros (+124.1%). Considere aumentar o tempo de treinamento ou ajustar os pesos da recompensa.

---

## 2. Comparação geral — agente vs baseline

| Métrica | Baseline | Agente PPO | Variação |
|---|---|---|---|
| Espera média carros (s) | 24.6 | 55.1 | +124.1% |
| Espera média pedestres (s) | 22.4 | 8.1 | -63.8% |
| Espera máxima carros (s) | 145.9 | 270.9 | +85.7% |
| Espera máxima pedestres (s) | 87.7 | 33.6 | -61.7% |
| Violações teto carros | 715.2 | 1133.3 | +58.5% |
| Violações teto pedestres | 752.2 | 5.0 | -99.3% |

---

## 3. Comparação por família de cenário

| Família | Esp.med.carros base | Esp.med.carros agente | Var. | Viol.base | Viol.agente | Var. |
|---|---|---|---|---|---|---|
| baixa_mov | 7.5 | 4.8 | -35.4% | 0.0 | 0.0 | — |
| equilibrado | 9.7 | 6.6 | -31.4% | 1.0 | 0.0 | -100.0% |
| pico_ped | 8.3 | 5.5 | -34.4% | 3008.0 | 19.9 | -99.3% |
| pico_veic | 72.9 | 203.4 | +179.2% | 2860.9 | 4533.4 | +58.5% |

---

## 4. Violações de teto de espera

| | Baseline | Agente PPO | Variação |
|---|---|---|---|
| Violações carros | 22887 | 36267 | +58.5% |
| Violações pedestres | 24072 | 159 | -99.3% |
| **Total** | **46959** | **36426** | **-22.4%** |

---

## 5. Arquivos gerados

- `results/agent_results.csv` — métricas do agente por cenário
- `results/comparison.csv` — comparação direta por cenário
- `results/relatorio.md` — este relatório

---

*Gerado automaticamente pelo módulo de avaliação da Fase 6.*
