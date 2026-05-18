# Benchmark de Tempo Fixo — Baseline

> Controlador: Fase A = 9 ticks (45 s) / Fase B = 5 ticks (25 s)  
> Cenários: 82 total  (50 treino + 32 avaliação)  
> Tetos: carros ≤ 90 s | pedestres ≤ 60 s

---

## Resumo geral

| métrica | média | std | mín | máx |
|---|---|---|---|---|
| espera_media_carros | 23.7 | 42.4 | 7.0 | 161.5 |
| espera_media_pedestres | 22.0 | 9.1 | 17.2 | 55.6 |
| espera_maxima_carros | 139.9 | 286.2 | 35.0 | 1035.0 |
| espera_maxima_pedestres | 82.8 | 74.9 | 50.0 | 425.0 |
| espera_p95_carros | 107.0 | 222.9 | 30.0 | 840.0 |
| espera_p95_pedestres | 61.1 | 33.8 | 50.0 | 190.0 |
| violacoes_teto_carros | 669.1 | 1958.6 | 0.0 | 7953.0 |
| violacoes_teto_pedestres | 656.2 | 1908.7 | 0.0 | 7604.0 |
| fila_media_veh_ns | 4.4 | 9.5 | 0.4 | 35.1 |
| fila_media_ped_l | 2.1 | 2.1 | 0.6 | 10.1 |
| fila_media_ped_o | 2.1 | 1.9 | 0.5 | 9.2 |
| throughput_total_carros | 10682.5 | 4563.7 | 4282.0 | 18847.0 |
| throughput_total_pedestres | 14611.2 | 6715.0 | 5609.0 | 27561.0 |

---

## Por família

| family | esp_media_carros | esp_media_pedestres | esp_maxima_carros | esp_maxima_pedestres | esp_p95_carros | esp_p95_pedestres | viol_teto_carros | viol_teto_pedestres | fila_veh_ns | fila_ped_l | fila_ped_o | tp_carros | tp_pedestres |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baixa_mov | 7.4 ±0.3 | 17.5 ±0.2 | 35.0 ±0.0 | 50.6 ±1.6 | 30.0 ±0.0 | 50.0 ±0.0 | 0.0 ±0.0 | 0.0 ±0.0 | 0.4 ±0.0 | 0.6 ±0.1 | 0.6 ±0.1 | 4695.9 ±330.4 | 6252.7 ±579.1 |
| equilibrado | 9.6 ±0.6 | 19.4 ±0.6 | 43.3 ±7.7 | 58.1 ±6.7 | 30.0 ±0.0 | 50.0 ±0.0 | 0.0 ±0.0 | 0.6 ±1.5 | 1.3 ±0.2 | 1.8 ±0.2 | 1.8 ±0.2 | 11696.1 ±751.1 | 15627.4 ±1421.8 |
| imprevisivel | 9.6 ±0.6 | 19.4 ±0.6 | 41.5 ±5.8 | 61.5 ±7.5 | 30.0 ±0.0 | 50.0 ±0.0 | 0.0 ±0.0 | 1.2 ±2.1 | 1.3 ±0.2 | 1.8 ±0.2 | 1.8 ±0.2 | 11738.8 ±788.7 | 15611.3 ±1467.9 |
| pico_ped | 8.3 ±0.3 | 34.0 ±14.0 | 35.0 ±0.0 | 183.3 ±113.3 | 30.0 ±0.0 | 100.6 ±57.6 | 0.0 ±0.0 | 2988.0 ±3159.3 | 0.8 ±0.1 | 5.2 ±2.7 | 5.0 ±2.4 | 8182.8 ±555.0 | 25037.9 ±2352.2 |
| pico_veic | 77.4 ±68.2 | 18.4 ±0.3 | 500.8 ±462.0 | 51.1 ±2.1 | 380.8 ±367.4 | 50.0 ±0.0 | 3048.3 ±3255.5 | 0.0 ±0.0 | 16.6 ±15.0 | 1.2 ±0.1 | 1.2 ±0.1 | 17568.4 ±1188.4 | 10971.2 ±1008.3 |

---

## Por tipo de dia

| day_type | esp_media_carros | esp_media_pedestres | esp_maxima_carros | esp_maxima_pedestres | esp_p95_carros | esp_p95_pedestres | viol_teto_carros | viol_teto_pedestres | fila_veh_ns | fila_ped_l | fila_ped_o | tp_carros | tp_pedestres |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fds | 9.1 ±1.7 | 18.9 ±1.4 | 41.0 ±10.3 | 58.5 ±13.6 | 30.0 ±0.0 | 50.0 ±0.0 | 0.0 ±0.0 | 6.4 ±14.1 | 1.1 ±0.7 | 1.5 ±0.8 | 1.5 ±0.8 | 9993.6 ±4237.3 | 13302.8 ±5991.0 |
| util | 38.4 ±56.5 | 25.0 ±12.1 | 238.8 ±381.8 | 107.1 ±99.8 | 184.0 ±297.4 | 72.2 ±45.3 | 1338.3 ±2617.2 | 1305.9 ±2551.8 | 7.6 ±12.7 | 2.8 ±2.7 | 2.7 ±2.5 | 11371.4 ±4821.6 | 15919.6 ±7204.7 |

---

## Por conjunto (treino vs avaliação)

| set | esp_media_carros | esp_media_pedestres | esp_maxima_carros | esp_maxima_pedestres | esp_p95_carros | esp_p95_pedestres | viol_teto_carros | viol_teto_pedestres | fila_veh_ns | fila_ped_l | fila_ped_o | tp_carros | tp_pedestres |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval | 24.6 ±42.3 | 22.4 ±10.0 | 145.9 ±286.3 | 87.7 ±85.6 | 113.0 ±227.1 | 62.8 ±36.4 | 715.2 ±1954.7 | 752.2 ±2082.6 | 4.5 ±9.5 | 2.2 ±2.3 | 2.2 ±2.2 | 10519.4 ±4904.0 | 14479.9 ±7193.6 |
| train | 23.2 ±42.8 | 21.7 ±8.6 | 136.0 ±289.0 | 79.7 ±67.8 | 103.2 ±222.4 | 60.0 ±32.3 | 639.6 ±1980.3 | 594.7 ±1807.8 | 4.2 ±9.6 | 2.1 ±2.0 | 2.0 ±1.8 | 10786.9 ±4380.2 | 14695.3 ±6463.8 |
