"""Efeitos visuais pré-renderizados: glow radial, painéis e sombras.

Como em ``sprites.py``, os builders devolvem surfaces para serem
construídas uma vez e cacheadas — não devem rodar por frame.
"""

from __future__ import annotations

import pygame


def build_radial_glow(
    radius: int, color: tuple[int, int, int], intensity: float = 0.6
) -> pygame.Surface:
    """Constrói um halo radial para blit aditivo (``BLEND_RGB_ADD``).

    A surface é preta com círculos concêntricos cuja luminância cresce
    da borda (0) para o centro (``intensity``) seguindo (1 - r/R)², o que
    aproxima a queda de intensidade de uma fonte de luz pontual.
    Use ``surface.blit(glow, pos, special_flags=pygame.BLEND_RGB_ADD)``.
    """
    size = radius * 2
    glow = pygame.Surface((size, size))
    for rr in range(radius, 0, -1):
        falloff = intensity * (1.0 - rr / radius) ** 2
        shade = tuple(int(c * falloff) for c in color)
        pygame.draw.circle(glow, shade, (radius, radius), rr)
    return glow


def build_panel(
    width: int,
    height: int,
    fill_rgba: tuple[int, int, int, int],
    border_rgba: tuple[int, int, int, int] = (255, 255, 255, 22),
    radius: int = 10,
    inset: int = 3,
) -> pygame.Surface:
    """Painel translúcido com cantos arredondados e borda fina.

    Usado como fundo do HUD e da barra de controles: o retângulo é
    desenhado com ``inset`` px de folga para a borda arredondada não
    encostar nas margens do painel.
    """
    panel = pygame.Surface((width, height), pygame.SRCALPHA)
    rect = pygame.Rect(inset, inset, width - inset * 2, height - inset * 2)
    pygame.draw.rect(panel, fill_rgba, rect, border_radius=radius)
    pygame.draw.rect(panel, border_rgba, rect, 1, border_radius=radius)
    return panel


def build_soft_shadow(
    width: int, height: int, radius: int = 8, alpha: int = 80
) -> pygame.Surface:
    """Sombra retangular arredondada semitransparente para caixas (semáforos)."""
    shadow = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(
        shadow, (0, 0, 0, alpha),
        pygame.Rect(0, 0, width, height), border_radius=radius,
    )
    return shadow
