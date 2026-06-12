"""Funções de easing e interpolação para animações em tempo real.

Todas as funções de easing recebem um progresso temporal ``t`` normalizado
em [0, 1] e devolvem o progresso visual correspondente, também em [0, 1].
O chamador é responsável por calcular ``t = elapsed_ms / duration_ms``.
"""

from __future__ import annotations


def clamp01(t: float) -> float:
    """Restringe ``t`` ao intervalo [0, 1]."""
    return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t


def lerp(a: float, b: float, t: float) -> float:
    """Interpolação linear entre ``a`` e ``b`` com peso ``t`` em [0, 1]."""
    return a + (b - a) * t


def ease_out_cubic(t: float) -> float:
    """Rápido no início, desacelera no fim — bom para chegadas.

    Matemática: 1 - (1 - t)³. A derivada em t=0 é 3 (movimento rápido)
    e em t=1 é 0 (parada suave).
    """
    t = clamp01(t)
    inv = 1.0 - t
    return 1.0 - inv * inv * inv


def ease_in_quad(t: float) -> float:
    """Lento no início, acelera no fim — bom para escoamentos (arrancada).

    Matemática: t². A derivada em t=0 é 0 (parte do repouso) e cresce
    linearmente, simulando aceleração constante.
    """
    t = clamp01(t)
    return t * t


def ease_in_out_quad(t: float) -> float:
    """Acelera na primeira metade e desacelera na segunda — travessias.

    Matemática: 2t² para t < 0.5 e 1 - 2(1-t)² para t ≥ 0.5, emendadas
    com derivada contínua em t = 0.5.
    """
    t = clamp01(t)
    if t < 0.5:
        return 2.0 * t * t
    inv = 1.0 - t
    return 1.0 - 2.0 * inv * inv
