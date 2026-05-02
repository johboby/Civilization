"""Character Portrait Generator - Procedural SVG-based character portraits.

Generates unique character portraits using SVG for use in the web frontend.
Each portrait is deterministically generated from the character's name and stats.
"""

from __future__ import annotations

import hashlib
from typing import Optional


# Color palettes for different aspects
SKIN_COLORS = [
    "#f5d6b8", "#e8c4a0", "#d4a574", "#c49060", "#b07848",
    "#a06838", "#8a5828", "#f0d0a8", "#e0b888", "#d0a070",
]

HAIR_COLORS = [
    "#1a1a1a", "#2c2c2c", "#3d2b1f", "#5a3825", "#8b6914",
    "#a0522d", "#654321", "#c0c0c0", "#f5f5dc", "#4a3728",
]

CLOTHING_COLORS = [
    "#8b0000", "#00008b", "#006400", "#8b8b00", "#4b0082",
    "#2f4f4f", "#800000", "#191970", "#556b2f", "#8b4513",
]

ACCESSORY_COLORS = [
    "#ffd700", "#c0c0c0", "#b87333", "#cd7f32", "#e5e4e2",
]

EYE_SHAPES = ["round", "narrow", "wide", "sharp"]
FACE_SHAPES = ["oval", "round", "square", "long"]
HAIR_STYLES = ["topknot", "flowing", "short", "bald", "braided", "helmet"]
FACIAL_HAIR = ["none", "beard", "mustache", "goatee", "full"]
ACCESSORIES = ["none", "crown", "headband", "earring", "scar"]


def _hash_seed(name: str, extra: str = "") -> int:
    """Generate a deterministic hash from a name."""
    h = hashlib.md5((name + extra).encode()).hexdigest()
    return int(h[:8], 16)


def _pick(items: list, seed: int, offset: int = 0) -> any:
    """Deterministically pick from a list using seed."""
    return items[(seed + offset) % len(items)]


def generate_portrait_svg(
    name: str,
    role: str = "free",
    command: int = 50,
    force: int = 50,
    intelligence: int = 50,
    politics: int = 50,
    charisma: int = 50,
    width: int = 80,
    height: int = 100,
) -> str:
    """Generate a unique SVG portrait for a character.

    Args:
        name: Character name (used as seed for deterministic generation).
        role: Character role (affects clothing/accessories).
        command-charisma: Stats (affect visual appearance).
        width: SVG width.
        height: SVG height.

    Returns:
        SVG string.
    """
    seed = _hash_seed(name)
    s2 = _hash_seed(name, "extra")

    skin = _pick(SKIN_COLORS, seed, 0)
    hair_color = _pick(HAIR_COLORS, seed, 1)
    cloth_color = _pick(CLOTHING_COLORS, seed, 2)
    eye_shape = _pick(EYE_SHAPES, seed, 3)
    face_shape = _pick(FACE_SHAPES, seed, 4)
    hair_style = _pick(HAIR_STYLES, seed, 5)
    facial_hair = _pick(FACIAL_HAIR, s2, 0)
    accessory = _pick(ACCESSORIES, s2, 1)
    acc_color = _pick(ACCESSORY_COLORS, s2, 2)

    # Role-specific overrides
    if role == "ruler":
        accessory = "crown"
        acc_color = "#ffd700"
    elif role == "general":
        hair_style = "helmet" if force > 60 else hair_style
    elif role == "strategist":
        facial_hair = "goatee" if intelligence > 60 else facial_hair

    cx, cy = width // 2, height // 2

    # Face dimensions based on face_shape
    face_rx = width * 0.32
    face_ry = height * 0.36
    if face_shape == "round":
        face_rx = width * 0.30
        face_ry = width * 0.30
    elif face_shape == "square":
        face_rx = width * 0.30
        face_ry = height * 0.32
    elif face_shape == "long":
        face_rx = width * 0.26
        face_ry = height * 0.40

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        # Background
        f'<rect width="{width}" height="{height}" fill="#1a1a2e" rx="6"/>',
        # Clothing/shoulders
        f'<ellipse cx="{cx}" cy="{height + 5}" rx="{width * 0.45}" ry="{height * 0.25}" fill="{cloth_color}"/>',
        # Neck
        f'<rect x="{cx - 6}" y="{cy + face_ry * 0.7}" width="12" height="{height * 0.15}" fill="{skin}" rx="3"/>',
        # Face
        f'<ellipse cx="{cx}" cy="{cy - 5}" rx="{face_rx}" ry="{face_ry}" fill="{skin}"/>',
    ]

    # Eyes
    eye_y = cy - height * 0.08
    eye_offset = face_rx * 0.35
    eye_w = 4 if eye_shape in ("narrow", "sharp") else 5
    eye_h = 3 if eye_shape == "narrow" else (5 if eye_shape == "wide" else 4)
    svg_parts.append(
        f'<ellipse cx="{cx - eye_offset}" cy="{eye_y}" rx="{eye_w}" ry="{eye_h}" fill="#1a1a1a"/>'
    )
    svg_parts.append(
        f'<ellipse cx="{cx + eye_offset}" cy="{eye_y}" rx="{eye_w}" ry="{eye_h}" fill="#1a1a1a"/>'
    )
    # Eye whites
    svg_parts.append(
        f'<ellipse cx="{cx - eye_offset}" cy="{eye_y}" rx="{eye_w - 1}" ry="{eye_h - 1}" fill="white"/>'
    )
    svg_parts.append(
        f'<ellipse cx="{cx + eye_offset}" cy="{eye_y}" rx="{eye_w - 1}" ry="{eye_h - 1}" fill="white"/>'
    )
    # Pupils
    svg_parts.append(
        f'<circle cx="{cx - eye_offset}" cy="{eye_y}" r="2" fill="#2c1810"/>'
    )
    svg_parts.append(
        f'<circle cx="{cx + eye_offset}" cy="{eye_y}" r="2" fill="#2c1810"/>'
    )

    # Eyebrows (thicker for higher force/command)
    brow_thickness = 1.5 + (command / 100) * 1.5
    svg_parts.append(
        f'<line x1="{cx - eye_offset - 5}" y1="{eye_y - 6}" x2="{cx - eye_offset + 5}" y2="{eye_y - 7}" '
        f'stroke="{hair_color}" stroke-width="{brow_thickness}" stroke-linecap="round"/>'
    )
    svg_parts.append(
        f'<line x1="{cx + eye_offset - 5}" y1="{eye_y - 7}" x2="{cx + eye_offset + 5}" y2="{eye_y - 6}" '
        f'stroke="{hair_color}" stroke-width="{brow_thickness}" stroke-linecap="round"/>'
    )

    # Nose
    nose_y = cy + height * 0.02
    svg_parts.append(
        f'<path d="M{cx - 3} {nose_y} Q{cx} {nose_y + 6} {cx + 3} {nose_y}" '
        f'fill="none" stroke="#00000030" stroke-width="1"/>'
    )

    # Mouth (wider smile for higher charisma)
    mouth_y = cy + height * 0.12
    mouth_w = 5 + (charisma / 100) * 6
    mouth_curve = 2 + (charisma / 100) * 3
    svg_parts.append(
        f'<path d="M{cx - mouth_w} {mouth_y} Q{cx} {mouth_y + mouth_curve} {cx + mouth_w} {mouth_y}" '
        f'fill="none" stroke="#8b4513" stroke-width="1.5" stroke-linecap="round"/>'
    )

    # Hair
    if hair_style == "topknot":
        svg_parts.append(
            f'<ellipse cx="{cx}" cy="{cy - face_ry - 3}" rx="{face_rx * 0.4}" ry="10" fill="{hair_color}"/>'
        )
        svg_parts.append(
            f'<path d="M{cx - face_rx * 0.8} {cy - face_ry * 0.5} '
            f'Q{cx - face_rx * 0.9} {cy - face_ry * 1.1} {cx} {cy - face_ry - 8} '
            f'Q{cx + face_rx * 0.9} {cy - face_ry * 1.1} {cx + face_rx * 0.8} {cy - face_ry * 0.5}" '
            f'fill="{hair_color}"/>'
        )
    elif hair_style == "flowing":
        svg_parts.append(
            f'<path d="M{cx - face_rx} {cy - face_ry * 0.3} '
            f'Q{cx - face_rx * 1.2} {cy - face_ry} {cx} {cy - face_ry * 1.1} '
            f'Q{cx + face_rx * 1.2} {cy - face_ry} {cx + face_rx} {cy - face_ry * 0.3}" '
            f'fill="{hair_color}"/>'
        )
        # Side hair
        svg_parts.append(
            f'<path d="M{cx - face_rx * 0.9} {cy - face_ry * 0.2} '
            f'Q{cx - face_rx * 1.1} {cy + face_ry * 0.3} {cx - face_rx * 0.8} {cy + face_ry * 0.6}" '
            f'fill="none" stroke="{hair_color}" stroke-width="5"/>'
        )
        svg_parts.append(
            f'<path d="M{cx + face_rx * 0.9} {cy - face_ry * 0.2} '
            f'Q{cx + face_rx * 1.1} {cy + face_ry * 0.3} {cx + face_rx * 0.8} {cy + face_ry * 0.6}" '
            f'fill="none" stroke="{hair_color}" stroke-width="5"/>'
        )
    elif hair_style == "short":
        svg_parts.append(
            f'<ellipse cx="{cx}" cy="{cy - face_ry * 0.6}" '
            f'rx="{face_rx * 0.95}" ry="{face_ry * 0.5}" fill="{hair_color}"/>'
        )
    elif hair_style == "helmet":
        svg_parts.append(
            f'<path d="M{cx - face_rx * 0.9} {cy - face_ry * 0.2} '
            f'L{cx - face_rx * 0.9} {cy - face_ry * 0.8} '
            f'Q{cx} {cy - face_ry * 1.4} {cx + face_rx * 0.9} {cy - face_ry * 0.8} '
            f'L{cx + face_rx * 0.9} {cy - face_ry * 0.2}" '
            f'fill="#708090" stroke="#505050" stroke-width="1"/>'
        )
        # Helmet ridge
        svg_parts.append(
            f'<line x1="{cx}" y1="{cy - face_ry * 1.3}" x2="{cx}" y2="{cy - face_ry * 0.2}" '
            f'stroke="#606060" stroke-width="3"/>'
        )
    elif hair_style == "braided":
        svg_parts.append(
            f'<ellipse cx="{cx}" cy="{cy - face_ry * 0.7}" '
            f'rx="{face_rx * 0.9}" ry="{face_ry * 0.4}" fill="{hair_color}"/>'
        )
        # Braid
        svg_parts.append(
            f'<path d="M{cx + face_rx * 0.7} {cy - face_ry * 0.3} '
            f'Q{cx + face_rx * 1.2} {cy + face_ry * 0.2} {cx + face_rx * 0.9} {cy + face_ry * 0.8}" '
            f'fill="none" stroke="{hair_color}" stroke-width="4"/>'
        )

    # Facial hair
    if facial_hair == "beard":
        svg_parts.append(
            f'<path d="M{cx - face_rx * 0.5} {mouth_y + 2} '
            f'Q{cx} {mouth_y + 18} {cx + face_rx * 0.5} {mouth_y + 2}" '
            f'fill="{hair_color}" opacity="0.8"/>'
        )
    elif facial_hair == "mustache":
        svg_parts.append(
            f'<path d="M{cx - 8} {mouth_y - 3} Q{cx} {mouth_y - 1} {cx + 8} {mouth_y - 3}" '
            f'fill="none" stroke="{hair_color}" stroke-width="2.5"/>'
        )
    elif facial_hair == "goatee":
        svg_parts.append(
            f'<ellipse cx="{cx}" cy="{mouth_y + 8}" rx="5" ry="7" fill="{hair_color}" opacity="0.8"/>'
        )
    elif facial_hair == "full":
        svg_parts.append(
            f'<path d="M{cx - face_rx * 0.6} {eye_y + 10} '
            f'Q{cx - face_rx * 0.7} {mouth_y + 15} {cx} {mouth_y + 20} '
            f'Q{cx + face_rx * 0.7} {mouth_y + 15} {cx + face_rx * 0.6} {eye_y + 10}" '
            f'fill="{hair_color}" opacity="0.7"/>'
        )

    # Accessories
    if accessory == "crown":
        crown_y = cy - face_ry - 5
        svg_parts.append(
            f'<path d="M{cx - 12} {crown_y} L{cx - 10} {crown_y - 10} '
            f'L{cx - 4} {crown_y - 4} L{cx} {crown_y - 12} '
            f'L{cx + 4} {crown_y - 4} L{cx + 10} {crown_y - 10} '
            f'L{cx + 12} {crown_y} Z" fill="{acc_color}" stroke="#b8860b" stroke-width="0.5"/>'
        )
        # Gem
        svg_parts.append(f'<circle cx="{cx}" cy="{crown_y - 6}" r="2" fill="#e74c3c"/>')
    elif accessory == "headband":
        hb_y = cy - face_ry * 0.5
        svg_parts.append(
            f'<rect x="{cx - face_rx}" y="{hb_y - 3}" '
            f'width="{face_rx * 2}" height="5" fill="{acc_color}" rx="2"/>'
        )
    elif accessory == "scar":
        scar_y = eye_y + 5
        svg_parts.append(
            f'<line x1="{cx - 10}" y1="{scar_y - 8}" x2="{cx - 2}" y2="{scar_y + 8}" '
            f'stroke="#8b4513" stroke-width="1.5" opacity="0.6"/>'
        )

    # Role indicator border
    role_colors = {
        "ruler": "#ffd700",
        "general": "#e74c3c",
        "strategist": "#3498db",
        "governor": "#2ecc71",
        "diplomat": "#9b59b6",
        "spy": "#555555",
        "free": "#888888",
    }
    border_color = role_colors.get(role, "#888888")
    svg_parts.append(
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" '
        f'fill="none" stroke="{border_color}" stroke-width="2" rx="6"/>'
    )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def generate_portrait_data_uri(name: str, **kwargs) -> str:
    """Generate a portrait as a data URI for embedding in HTML."""
    svg = generate_portrait_svg(name, **kwargs)
    import base64
    encoded = base64.b64encode(svg.encode('utf-8')).decode('ascii')
    return f"data:image/svg+xml;base64,{encoded}"
