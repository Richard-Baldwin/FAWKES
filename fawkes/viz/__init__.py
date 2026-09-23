"""L5 - trajectory pictures (stdlib SVG; the replaytest heritage, minimal start)."""

from __future__ import annotations

from pathlib import Path


def trajectory_svg(
    positions: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    width_mm: int = 4500,
    height_mm: int = 3000,
    scale: float = 0.2,
) -> str:
    """Render one episode's path as SVG (world metres -> mm canvas)."""

    def tx(x):
        return (x * 1000 + width_mm / 2) * scale

    def ty(y):
        return (height_mm / 2 - y * 1000) * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width_mm*scale)}" height="{int(height_mm*scale)}">',
        f'<rect width="100%" height="100%" fill="#0e1116"/>',
        f'<rect x="{tx(-width_mm/2000)}" y="{ty(width_mm/2000)}" width="{width_mm*scale}" height="{height_mm*scale}" fill="none" stroke="#2c3440" stroke-width="2"/>',
    ]
    for tx_, ty_ in targets:
        parts.append(f'<circle cx="{tx(tx_)}" cy="{ty(ty_)}" r="5" fill="none" stroke="#e6b450" stroke-width="2"/>')
    if len(positions) > 1:
        pts = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in positions)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#4ec9b0" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "".join(parts)


def write_svg(path: str | Path, svg: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path
