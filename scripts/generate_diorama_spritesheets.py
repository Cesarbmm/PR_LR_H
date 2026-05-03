from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw


FRAME_SIZE = 64
FRAMES_PER_ANIMATION = 4
ANIMATIONS = ["idle", "walk", "scan", "handoff", "wait"]
OUTLINE = (23, 28, 35, 255)
SKIN = (239, 228, 210, 255)

ROLE_SPECS: Dict[str, Dict[str, object]] = {
    "courier": {
        "body": (239, 229, 112, 255),
        "trim": (202, 170, 75, 255),
        "limb": (76, 83, 96, 255),
        "accent": (255, 220, 122, 255),
        "backpack": True,
        "device": True,
        "stripe": True,
    },
    "customer": {
        "body": (184, 226, 179, 255),
        "trim": (123, 174, 121, 255),
        "limb": (74, 90, 82, 255),
        "accent": (255, 188, 122, 255),
    },
    "supervisor": {
        "body": (111, 175, 236, 255),
        "trim": (82, 129, 181, 255),
        "limb": (66, 78, 94, 255),
        "accent": (245, 234, 206, 255),
        "clipboard": True,
    },
    "rival": {
        "body": (238, 184, 96, 255),
        "trim": (194, 140, 68, 255),
        "limb": (85, 78, 68, 255),
        "accent": (255, 230, 170, 255),
        "backpack": True,
        "parcel": True,
        "stripe": True,
    },
    "thief": {
        "body": (232, 115, 95, 255),
        "trim": (148, 70, 58, 255),
        "limb": (66, 50, 55, 255),
        "accent": (255, 148, 122, 255),
        "hood": True,
        "parcel": True,
    },
}


def _rgba(color: Tuple[int, int, int, int], alpha_scale: float = 1.0) -> Tuple[int, int, int, int]:
    return (color[0], color[1], color[2], max(0, min(255, int(color[3] * alpha_scale))))


def _frame_phase(index: int) -> float:
    return [-1.0, -0.35, 0.35, 1.0][index % FRAMES_PER_ANIMATION]


def _draw_shadow(draw: ImageDraw.ImageDraw, bob: float) -> None:
    draw.ellipse((14, 49 + bob * 0.15, 50, 58 + bob * 0.15), fill=(0, 0, 0, 48))


def _draw_limb(
    draw: ImageDraw.ImageDraw,
    points: Tuple[Tuple[float, float], Tuple[float, float]],
    color: Tuple[int, int, int, int],
    width: int,
) -> None:
    draw.line(points, fill=OUTLINE, width=width + 3)
    draw.line(points, fill=color, width=width)


def _draw_round_rect(
    draw: ImageDraw.ImageDraw,
    box: Tuple[float, float, float, float],
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int] = OUTLINE,
    radius: int = 6,
    outline_width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=outline_width)


def _draw_parcel(draw: ImageDraw.ImageDraw, x: float, y: float, accent: Tuple[int, int, int, int]) -> None:
    box = (x - 5, y - 4, x + 7, y + 6)
    _draw_round_rect(draw, box, (219, 183, 108, 255), radius=2, outline_width=2)
    draw.line((x + 1, y - 4, x + 1, y + 6), fill=accent, width=2)
    draw.line((x - 5, y + 1, x + 7, y + 1), fill=accent, width=2)


def _draw_sprite_frame(role_key: str, spec: Dict[str, object], animation: str, frame_index: int) -> Image.Image:
    img = Image.new("RGBA", (FRAME_SIZE, FRAME_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    phase = _frame_phase(frame_index)
    cx = FRAME_SIZE / 2
    bob = 0.0
    torso_tilt = 0.0
    arm_swing = 0.0
    leg_swing = 0.0
    arm_raise = 0.0
    wait_drop = 0.0
    parcel_offset = 0.0
    if animation == "walk":
        bob = abs(phase) * 1.8
        torso_tilt = phase * 1.2
        arm_swing = phase * 7.0
        leg_swing = phase * 6.0
    elif animation == "scan":
        bob = 0.7
        torso_tilt = -1.6
        arm_raise = 13.0
        arm_swing = -4.0
        leg_swing = phase * 1.2
    elif animation == "handoff":
        bob = 0.6
        torso_tilt = -1.1
        arm_raise = 9.0
        arm_swing = -9.0
        leg_swing = 1.4 * phase
        parcel_offset = 7.0 + phase * 1.8
    elif animation == "wait":
        bob = 1.2
        torso_tilt = -0.8
        wait_drop = 3.5
        arm_swing = -1.2
    else:
        bob = 0.4 + frame_index * 0.1

    body = spec["body"]
    trim = spec["trim"]
    limb = spec["limb"]
    accent = spec["accent"]
    foot_y = 52 + bob
    hip_y = 38 + bob + wait_drop
    shoulder_y = 26 + bob + wait_drop * 0.8
    head_y = 16 + bob + wait_drop * 0.7

    _draw_shadow(draw, bob)

    left_leg = ((cx - 5, hip_y), (cx - 7 - leg_swing, foot_y))
    right_leg = ((cx + 5, hip_y), (cx + 7 + leg_swing, foot_y))
    _draw_limb(draw, left_leg, limb, 6)
    _draw_limb(draw, right_leg, limb, 6)

    torso = [
        (cx - 12 + torso_tilt, shoulder_y - 2),
        (cx + 12 + torso_tilt, shoulder_y - 2),
        (cx + 15 + torso_tilt * 0.4, hip_y - 3),
        (cx, hip_y + 8),
        (cx - 15 + torso_tilt * 0.4, hip_y - 3),
    ]
    draw.polygon(torso, fill=body, outline=OUTLINE)
    trim_band = [
        (cx - 8 + torso_tilt * 0.7, shoulder_y + 8),
        (cx + 8 + torso_tilt * 0.7, shoulder_y + 8),
        (cx + 10 + torso_tilt * 0.4, hip_y - 2),
        (cx - 10 + torso_tilt * 0.4, hip_y - 2),
    ]
    draw.polygon(trim_band, fill=_rgba(trim, 0.95), outline=None)

    if spec.get("stripe"):
        draw.line((cx - 6 + torso_tilt * 0.3, shoulder_y + 2, cx + 6 + torso_tilt * 0.3, shoulder_y + 11), fill=accent, width=3)

    if spec.get("backpack"):
        _draw_round_rect(draw, (cx - 16 + torso_tilt * 0.2, shoulder_y - 1, cx - 8 + torso_tilt * 0.2, hip_y + 3), _rgba(trim, 0.95), radius=4)

    left_arm_end = (cx - 16 - arm_swing * 0.35, shoulder_y + 14)
    right_arm_end = (cx + 16 + arm_swing + arm_raise * 0.25, shoulder_y + 14 - arm_raise)
    _draw_limb(draw, ((cx - 10, shoulder_y), left_arm_end), limb, 5)
    _draw_limb(draw, ((cx + 10, shoulder_y), right_arm_end), limb, 5)

    if spec.get("clipboard"):
        _draw_round_rect(draw, (left_arm_end[0] - 4, left_arm_end[1] - 1, left_arm_end[0] + 4, left_arm_end[1] + 8), (245, 235, 210, 255), radius=2)
        draw.line((left_arm_end[0] - 2, left_arm_end[1] + 2, left_arm_end[0] + 2, left_arm_end[1] + 2), fill=(96, 118, 148, 255), width=1)

    if spec.get("device") or animation == "scan":
        _draw_round_rect(draw, (right_arm_end[0] - 2, right_arm_end[1] - 1, right_arm_end[0] + 4, right_arm_end[1] + 5), (96, 208, 244, 255), radius=2)
        if animation == "scan":
            beam_x = right_arm_end[0] + 10
            draw.line((right_arm_end[0] + 2, right_arm_end[1] + 2, beam_x, right_arm_end[1] - 2), fill=(97, 208, 244, 220), width=2)
            draw.arc((beam_x - 4, right_arm_end[1] - 7, beam_x + 7, right_arm_end[1] + 4), -35, 35, fill=(97, 208, 244, 220), width=2)

    if spec.get("parcel") or animation == "handoff":
        _draw_parcel(draw, right_arm_end[0] + parcel_offset, right_arm_end[1] + 5, accent)

    head_box = (cx - 8, head_y - 7, cx + 8, head_y + 10)
    draw.ellipse(head_box, fill=SKIN, outline=OUTLINE, width=2)
    if spec.get("hood"):
        hood_box = (cx - 10, head_y - 9, cx + 10, head_y + 11)
        draw.arc(hood_box, 190, 350, fill=_rgba(trim, 0.95), width=6)
    else:
        draw.pieslice((cx - 8, head_y - 9, cx + 8, head_y + 4), 180, 360, fill=_rgba(limb, 0.92), outline=None)
    draw.ellipse((cx - 4, head_y, cx - 2, head_y + 2), fill=OUTLINE)
    draw.ellipse((cx + 2, head_y, cx + 4, head_y + 2), fill=OUTLINE)

    if animation == "wait":
        draw.ellipse((cx + 10, head_y - 16, cx + 20, head_y - 6), fill=(255, 188, 122, 70), outline=(255, 188, 122, 160))
        draw.ellipse((cx + 18, head_y - 23, cx + 25, head_y - 16), fill=(255, 188, 122, 50), outline=None)
    elif animation == "handoff":
        draw.line((cx + 20, shoulder_y + 8, cx + 28, shoulder_y + 8), fill=(130, 222, 165, 220), width=2)

    if role_key == "courier" and animation in {"walk", "scan"}:
        draw.line((cx - 17, hip_y + 2, cx - 22, hip_y + 6), fill=accent, width=2)
        draw.line((cx + 17, hip_y + 2, cx + 22, hip_y + 6), fill=accent, width=2)

    return img


def _build_sheet(role_key: str, spec: Dict[str, object]) -> Image.Image:
    sheet = Image.new(
        "RGBA",
        (FRAME_SIZE * FRAMES_PER_ANIMATION, FRAME_SIZE * len(ANIMATIONS)),
        (0, 0, 0, 0),
    )
    for row, animation in enumerate(ANIMATIONS):
        for col in range(FRAMES_PER_ANIMATION):
            frame = _draw_sprite_frame(role_key, spec, animation, col)
            sheet.alpha_composite(frame, (col * FRAME_SIZE, row * FRAME_SIZE))
    return sheet


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "godot_broadcast" / "assets" / "diorama" / "spritesheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for role_key, spec in ROLE_SPECS.items():
        sheet = _build_sheet(role_key, spec)
        out_path = out_dir / f"{role_key}_sheet.png"
        sheet.save(out_path)
        print(out_path.relative_to(repo_root))


if __name__ == "__main__":
    main()
