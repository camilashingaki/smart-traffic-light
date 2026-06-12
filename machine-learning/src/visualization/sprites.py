"""Sprites top-down pré-renderizados para o cruzamento (carros e pedestres).

Cada builder devolve uma ``pygame.Surface`` com canal alpha pronta para
blit. As surfaces devem ser construídas uma única vez e cacheadas pelo
chamador — nenhum builder é barato o bastante para rodar por frame.
"""

from __future__ import annotations

import pygame

# ── Dimensões e cores do carro ───────────────────────────────────────────────
CAR_SPRITE_PAD = 5            # margem ao redor do corpo para a sombra
_SHADOW_RGBA   = (0, 0, 0, 70)
_GLASS         = (176, 197, 213)
_HEADLIGHT     = (255, 240, 150)
_TAILLIGHT     = (255, 95, 80)

# ── Dimensões e cores do pedestre ────────────────────────────────────────────
PED_SPRITE_SIZE = 20          # lado da surface quadrada do pedestre
_SKIN           = (236, 207, 178)
_PED_SHADOW     = (0, 0, 0, 55)


def _scale_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Escurece (<1) ou clareia (>1) uma cor RGB, saturando em [0, 255]."""
    return tuple(min(255, max(0, int(c * factor))) for c in color)


def build_car_sprite(
    width: int, length: int, body_color: tuple[int, int, int], facing_down: bool
) -> pygame.Surface:
    """Constrói a silhueta top-down de um carro.

    O carro é desenhado com a frente apontada para baixo (sul) e espelhado
    verticalmente quando ``facing_down`` é False. Elementos: sombra projetada,
    corpo arredondado, teto mais claro, para-brisas dianteiro/traseiro
    (trapezoides), faróis amarelos e lanternas vermelhas.
    """
    pad = CAR_SPRITE_PAD
    w, le = width, length

    body = pygame.Surface((w, le), pygame.SRCALPHA)
    radius = max(4, w // 3)

    # corpo + contorno escuro sutil
    pygame.draw.rect(body, body_color, pygame.Rect(0, 0, w, le), border_radius=radius)
    pygame.draw.rect(
        body, _scale_color(body_color, 0.6),
        pygame.Rect(0, 0, w, le), 1, border_radius=radius,
    )

    # teto: faixa central levemente mais clara
    roof = pygame.Rect(3, int(le * 0.34), w - 6, int(le * 0.30))
    pygame.draw.rect(body, _scale_color(body_color, 1.14), roof, border_radius=3)

    # para-brisa dianteiro (frente = baixo): trapézio mais largo no lado do teto
    front_ws = [
        (5, int(le * 0.64)), (w - 5, int(le * 0.64)),
        (w - 7, int(le * 0.80)), (7, int(le * 0.80)),
    ]
    pygame.draw.polygon(body, _GLASS, front_ws)

    # para-brisa traseiro: trapézio espelhado, um pouco menor
    rear_ws = [
        (7, int(le * 0.18)), (w - 7, int(le * 0.18)),
        (w - 5, int(le * 0.32)), (5, int(le * 0.32)),
    ]
    pygame.draw.polygon(body, _GLASS, rear_ws)

    # faróis (frente) e lanternas (traseira)
    for x in (6, w - 6):
        pygame.draw.circle(body, _HEADLIGHT, (x, le - 3), 2)
        pygame.draw.circle(body, _TAILLIGHT, (x, 3), 2)

    if not facing_down:
        body = pygame.transform.flip(body, False, True)

    # composição final: sombra deslocada para baixo-direita + corpo por cima
    sprite = pygame.Surface((w + pad * 2, le + pad * 2), pygame.SRCALPHA)
    pygame.draw.ellipse(sprite, _SHADOW_RGBA, pygame.Rect(pad + 2, pad + 3, w, le))
    sprite.blit(body, (pad, pad))
    return sprite


def build_ped_sprite(shirt_color: tuple[int, int, int]) -> pygame.Surface:
    """Constrói uma figura humana top-down simples.

    Vista de cima: ombros como elipse na cor da roupa, braços como
    pontinhos laterais mais claros, cabeça circular em tom de pele e
    uma sombra leve por baixo.
    """
    s = PED_SPRITE_SIZE
    cx = cy = s // 2
    sprite = pygame.Surface((s, s), pygame.SRCALPHA)

    # sombra leve deslocada
    pygame.draw.ellipse(sprite, _PED_SHADOW, pygame.Rect(3, cy - 3, s - 5, 9))

    # ombros / tronco
    pygame.draw.ellipse(sprite, shirt_color, pygame.Rect(2, cy - 5, s - 4, 10))
    pygame.draw.ellipse(
        sprite, _scale_color(shirt_color, 0.65),
        pygame.Rect(2, cy - 5, s - 4, 10), 1,
    )

    # braços: pontinhos nas extremidades dos ombros
    arm = _scale_color(shirt_color, 1.3)
    pygame.draw.circle(sprite, arm, (3, cy), 2)
    pygame.draw.circle(sprite, arm, (s - 3, cy), 2)

    # cabeça por cima do tronco
    pygame.draw.circle(sprite, _SKIN, (cx, cy), 4)
    pygame.draw.circle(sprite, _scale_color(_SKIN, 0.6), (cx, cy), 4, 1)
    return sprite
