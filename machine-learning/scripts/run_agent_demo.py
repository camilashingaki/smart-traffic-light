"""
Demo visual do agente PPO treinado — Fase 5.

Carrega o modelo treinado (models/ppo_semaforo_final.zip) e mostra
visualmente cada decisão que ele toma no cruzamento em tempo real,
usando o mesmo renderer da Fase 2 (Pygame + matplotlib).

Rode com:
    python scripts/run_agent_demo.py

Controles:
    ESPAÇO      — pausa / retoma
    1 / 2 / 3 / 4 — velocidade 1x / 5x / 10x / 50x
    R           — reinicia o cenário
    F           — troca de cenário aleatório
    Q / ESC     — encerra

Requisitos:
    - Fase 4 concluída (env.py funcionando)
    - Fase 5 concluída (models/ppo_semaforo_final.zip existe)
    - pygame instalado (pip install pygame)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Garante que a raiz do projeto está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pygame
from stable_baselines3 import PPO

from src.simulation.crossing import Crossing
from src.simulation.simulation_loop import SimulationLoop
from src.utils.config_loader import load_config
from src.visualization.pygame_renderer import CrossingRenderer
from src.visualization.live_plots import LivePlots


# ── Constantes de layout ────────────────────────────────────────────────────
RENDERER_W = 768
PLOTS_W    = 512
WIN_H      = 720
FPS        = 60


class PPOController:
    """
    Adaptador que transforma o modelo PPO numa interface de controlador.

    O SimulationLoop espera um objeto com método decide(state) -> int.
    Este adaptador converte o state dict do crossing num vetor de observação
    no mesmo formato que o TrafficLightEnv usa, e chama model.predict().
    """

    # Denominadores de normalização — iguais aos do TrafficLightEnv
    _NORM = {
        "veh_ns":         30.0,
        "ped_l":          15.0,
        "ped_o":          15.0,
        "ticks_in_phase": 36.0,
        "max_wait_car_s": 90.0,
        "max_wait_ped_s": 60.0,
    }
    _TICK_S = 5

    def __init__(self, model: PPO) -> None:
        self._model = model

    def decide(self, state: dict) -> int:
        """
        Converte o state dict do crossing em observação e retorna a ação do agente.

        Parâmetros
        ----------
        state : dict
            Retorno de Crossing.get_state().

        Retorna
        -------
        action : int
            0 = manter fase atual, 1 = solicitar troca de fase.
        """
        import numpy as np
        from src.simulation.crossing import Phase

        n   = self._NORM
        ts  = self._TICK_S

        max_wait_car_s = state["veh_ns"]["max_wait_ticks"] * ts
        max_wait_ped_s = max(
            state["ped_l"]["max_wait_ticks"],
            state["ped_o"]["max_wait_ticks"],
        ) * ts

        obs = np.array([
            state["veh_ns"]["size"]  / n["veh_ns"],
            state["ped_l"]["size"]   / n["ped_l"],
            state["ped_o"]["size"]   / n["ped_o"],
            float(state["phase"] == "B"),
            state["ticks_in_phase"]  / n["ticks_in_phase"],
            max_wait_car_s           / n["max_wait_car_s"],
            max_wait_ped_s           / n["max_wait_ped_s"],
        ], dtype="float32").reshape(1, -1)

        obs = obs.clip(0.0, 1.0)
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)


def main() -> None:
    # ── Carrega configurações ─────────────────────────────────────────────
    cfg    = load_config("configs/config.yaml")
    rl_cfg = load_config("configs/rl.yaml")
    sim    = cfg["simulation"]
    viz    = cfg["visualization"]

    tick_s      = sim["tick_seconds"]
    ceil_car_s  = rl_cfg["teto_espera_carros"]
    ceil_ped_s  = rl_cfg["teto_espera_pedestres"]
    plot_interv = viz["plot_update_interval_ticks"]
    history_len = viz["history_length_ticks"]

    # ── Verifica se o modelo existe ───────────────────────────────────────
    model_path = Path("models/ppo_semaforo_final.zip")
    if not model_path.exists():
        # Tenta o melhor modelo como fallback
        best = Path("models/best/best_model.zip")
        if best.exists():
            model_path = best
            print(f"Modelo final não encontrado. Usando melhor modelo: {best}")
        else:
            print("ERRO: Nenhum modelo encontrado em models/.")
            print("Execute primeiro: python scripts/train_agent.py")
            sys.exit(1)

    print(f"Carregando modelo: {model_path}")
    model = PPO.load(str(model_path))
    controller = PPOController(model)

    # ── Carrega cenários disponíveis ──────────────────────────────────────
    scenarios_dir = Path("scenarios/eval")
    scenario_paths = sorted(scenarios_dir.glob("*.csv"))
    if not scenario_paths:
        scenarios_dir = Path("scenarios/train")
        scenario_paths = sorted(scenarios_dir.glob("*.csv"))
    if not scenario_paths:
        print("ERRO: Nenhum cenário CSV encontrado.")
        sys.exit(1)

    def sortear_cenario() -> tuple[pd.DataFrame, str]:
        path = random.choice(scenario_paths)
        return pd.read_csv(path), path.name

    scenario_df, scenario_name = sortear_cenario()

    def arrivals_fn(tick: int) -> dict[str, int]:
        """Retorna chegadas do tick atual no cenário carregado."""
        if tick >= len(scenario_df):
            return {"veh_ns": 0, "ped_l": 0, "ped_o": 0}
        row = scenario_df.iloc[tick]
        return {
            "veh_ns": int(row["veh_ns"]),
            "ped_l":  int(row["ped_l"]),
            "ped_o":  int(row["ped_o"]),
        }

    # ── Monta a simulação ─────────────────────────────────────────────────
    crossing = Crossing(cfg=sim)
    loop     = SimulationLoop(
        crossing=crossing,
        controller=controller,
        arrivals_fn=arrivals_fn,
        tick_seconds=tick_s,
    )

    # ── Inicializa Pygame ─────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((RENDERER_W + PLOTS_W, WIN_H))
    pygame.display.set_caption("Semáforo Inteligente — Agente PPO")
    clock = pygame.time.Clock()

    renderer = CrossingRenderer(cfg=cfg, panel_width=RENDERER_W, panel_height=WIN_H)
    plots    = LivePlots(
        panel_width=PLOTS_W,
        panel_height=WIN_H,
        history_length=history_len,
        tick_seconds=tick_s,
        ceil_cars_s=ceil_car_s,
        ceil_peds_s=ceil_ped_s,
    )

    # Superfície do painel de plots
    plot_surface = pygame.Surface((PLOTS_W, WIN_H))

    # Mapeamento de velocidades
    speed_map = {"1x": 1.0, "5x": 5.0, "10x": 10.0, "50x": 50.0}
    active_speed = "1x"
    loop.set_speed(1.0)

    font = pygame.font.SysFont("monospace", 13)

    def on_tick(state: dict) -> None:
        plots.record(state)

    loop._on_tick = on_tick

    print("\nDemo iniciado! Pressione Q ou ESC para encerrar.")
    print(f"Cenário: {scenario_name}\n")

    # ── Loop principal ────────────────────────────────────────────────────
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    loop.toggle_pause()
                elif event.key == pygame.K_r:
                    loop.reset()
                    plots.clear_history()
                elif event.key == pygame.K_f:
                    # Troca de cenário
                    scenario_df, scenario_name = sortear_cenario()
                    loop.reset()
                    plots.clear_history()
                    pygame.display.set_caption(
                        f"Semáforo Inteligente — Agente PPO | {scenario_name}"
                    )
                    print(f"Novo cenário: {scenario_name}")
                elif event.key == pygame.K_1:
                    active_speed = "1x";  loop.set_speed(1.0)
                elif event.key == pygame.K_2:
                    active_speed = "5x";  loop.set_speed(5.0)
                elif event.key == pygame.K_3:
                    active_speed = "10x"; loop.set_speed(10.0)
                elif event.key == pygame.K_4:
                    active_speed = "50x"; loop.set_speed(50.0)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Cliques nos botões do renderer
                pos = event.pos
                if pos[0] < RENDERER_W:
                    btns = renderer.get_button_rects()
                    for label, rect in btns.items():
                        if rect.collidepoint(pos):
                            if label == "PAUSE":
                                loop.toggle_pause()
                            elif label == "R":
                                loop.reset()
                                plots.clear_history()
                            elif label in speed_map:
                                active_speed = label
                                loop.set_speed(speed_map[label])

        # Atualiza simulação
        loop.update(dt)

        # Atualiza gráficos
        plots.update_surface(plot_interv)

        # Desenha renderer (painel esquerdo)
        renderer_surface = pygame.Surface((RENDERER_W, WIN_H))
        renderer.draw(
            surface=renderer_surface,
            state=loop.state,
            is_yellow=loop.is_yellow,
            is_paused=loop.paused,
            speed=loop.speed,
            active_speed_label=active_speed,
            mouse_pos=pygame.mouse.get_pos(),
            cfg_fixed=cfg["fixed_time_controller"],
            cfg_thresholds=cfg["thresholds"],
            tick_seconds=tick_s,
            arrivals=loop.last_arrivals,
            drains=loop.last_drains,
        )
        screen.blit(renderer_surface, (0, 0))

        # Desenha gráficos (painel direito)
        plot_surf = plots.get_surface()
        if plot_surf:
            screen.blit(plot_surf, (RENDERER_W, 0))
        else:
            # Placeholder enquanto os gráficos não renderizaram ainda
            plot_surface.fill((26, 26, 44))
            lbl = font.render("Aguardando dados...", True, (150, 150, 180))
            plot_surface.blit(lbl, (PLOTS_W // 2 - lbl.get_width() // 2, WIN_H // 2))
            screen.blit(plot_surface, (RENDERER_W, 0))

        # Mostra nome do cenário e label do agente no topo direito
        cenario_lbl = font.render(f"Cenário: {scenario_name}", True, (150, 150, 180))
        agente_lbl  = font.render("Controlador: Agente PPO", True, (100, 200, 120))
        screen.blit(cenario_lbl, (RENDERER_W + 10, 8))
        screen.blit(agente_lbl,  (RENDERER_W + 10, 24))

        pygame.display.flip()

    pygame.quit()
    plots.close()
    print("Demo encerrado.")


if __name__ == "__main__":
    main()
