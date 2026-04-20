#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _extract_frames(raw: Any) -> Tuple[List[Dict[str, Any]], str]:
    """
    Returns:
      frames, shape_kind

    shape_kind:
      - "list"
      - "frames"
      - "timeline"
    """
    if isinstance(raw, list):
        frames = [fr for fr in raw if isinstance(fr, dict)]
        return frames, "list"

    if isinstance(raw, dict):
        if isinstance(raw.get("frames"), list):
            frames = [fr for fr in raw["frames"] if isinstance(fr, dict)]
            return frames, "frames"
        if isinstance(raw.get("timeline"), list):
            frames = [fr for fr in raw["timeline"] if isinstance(fr, dict)]
            return frames, "timeline"

    raise ValueError("unsupported mouth json shape: expected list / {frames:[...]} / {timeline:[...]}")


def _wrap_like(raw: Any, shape_kind: str, frames_new: List[Dict[str, Any]]) -> Any:
    if shape_kind == "list":
        return frames_new

    if not isinstance(raw, dict):
        raise ValueError("internal error: raw must be dict for wrapped output")

    out = dict(raw)
    if shape_kind == "frames":
        out["frames"] = frames_new
        return out
    if shape_kind == "timeline":
        out["timeline"] = frames_new
        return out

    raise ValueError(f"unknown shape_kind: {shape_kind}")


def _safe_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _last_t_ms(frames: List[Dict[str, Any]]) -> int | None:
    for fr in reversed(frames):
        t = _safe_int(fr.get("t_ms"))
        if t is not None:
            return t
    return None


def _validate_step_ms(step_ms: int) -> None:
    if step_ms <= 0:
        raise ValueError("step_ms must be > 0")


def _validate_audio_ms(audio_ms: int) -> None:
    if audio_ms <= 0:
        raise ValueError("audio_ms must be > 0")


def build_clamped_mouth(
    *,
    mouth_json: Path,
    out_json: Path,
    audio_ms: int,
    step_ms: int = 40,
    keep_meta_fields: bool = True,
) -> Dict[str, Any]:
    _validate_audio_ms(audio_ms)
    _validate_step_ms(step_ms)

    raw = _load_json(mouth_json)
    frames, shape_kind = _extract_frames(raw)

    target_frames = math.ceil(audio_ms / step_ms)
    cutoff_ms = target_frames * step_ms  # keep frames with t_ms < cutoff_ms

    clamped: List[Dict[str, Any]] = []
    dropped_no_t_ms = 0
    dropped_out_of_range = 0

    for fr in frames:
        t = _safe_int(fr.get("t_ms"))
        if t is None:
            dropped_no_t_ms += 1
            continue
        if t < 0:
            dropped_out_of_range += 1
            continue
        if t >= cutoff_ms:
            dropped_out_of_range += 1
            continue
        clamped.append(dict(fr))

    if keep_meta_fields:
        out_obj = _wrap_like(raw, shape_kind, clamped)
    else:
        if shape_kind == "list":
            out_obj = clamped
        else:
            out_obj = {"frames": clamped}

    stats = {
        "mouth_json": str(mouth_json),
        "out_json": str(out_json),
        "audio_ms": int(audio_ms),
        "step_ms": int(step_ms),
        "target_frames": int(target_frames),
        "cutoff_ms": int(cutoff_ms),
        "frames_n_before": int(len(frames)),
        "frames_n_after": int(len(clamped)),
        "last_t_ms_before": _last_t_ms(frames),
        "last_t_ms_after": _last_t_ms(clamped),
        "dropped_no_t_ms": int(dropped_no_t_ms),
        "dropped_out_of_range": int(dropped_out_of_range),
        "shape_kind": shape_kind,
    }

    _dump_json(out_json, out_obj)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouth_json", required=True, help="Input mouth timeline json")
    ap.add_argument("--out_json", required=True, help="Output clamped mouth json")
    ap.add_argument("--audio_ms", required=True, type=int, help="SSOT session audio length in ms")
    ap.add_argument("--step_ms", type=int, default=40, help="Frame step in ms (default: 40)")
    ap.add_argument(
        "--no_keep_meta_fields",
        action="store_true",
        help="If set, do not preserve wrapper/meta fields; output plain frames wrapper",
    )
    args = ap.parse_args()

    mouth_json = Path(args.mouth_json).resolve()
    out_json = Path(args.out_json).resolve()

    if not mouth_json.exists():
        raise FileNotFoundError(f"missing mouth_json: {mouth_json}")

    stats = build_clamped_mouth(
        mouth_json=mouth_json,
        out_json=out_json,
        audio_ms=int(args.audio_ms),
        step_ms=int(args.step_ms),
        keep_meta_fields=not bool(args.no_keep_meta_fields),
    )

    print("[build_clamped_mouth][OK]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())