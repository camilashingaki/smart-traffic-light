"""Seleção de fonte sans-serif do sistema com fallback para monospace.

Tenta, nesta ordem: SF Pro Display, Helvetica Neue, Inter, Arial.
Se nenhuma existir no sistema, mantém a monospace original do projeto.
"""

from __future__ import annotations

import pygame

# Nomes já normalizados como o pygame espera (minúsculas, sem espaços)
_UI_FONT_CHAIN = ["sfprodisplay", "helveticaneue", "inter", "arial"]
_FALLBACK = "monospace"

_resolved_name: str | None = None
_resolved = False


def ui_font_name() -> str:
    """Devolve o nome da primeira fonte da cadeia disponível no sistema."""
    global _resolved_name, _resolved
    if not _resolved:
        pygame.font.init()
        _resolved_name = next(
            (name for name in _UI_FONT_CHAIN if pygame.font.match_font(name)),
            _FALLBACK,
        )
        _resolved = True
    return _resolved_name or _FALLBACK


def load_ui_font(size: int, bold: bool = False) -> pygame.font.Font:
    """Carrega a fonte de interface no tamanho pedido (com fallback)."""
    return pygame.font.SysFont(ui_font_name(), size, bold=bold)
