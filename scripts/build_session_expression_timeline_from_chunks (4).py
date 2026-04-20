#!/usr/bin/env python3
# scripts/build_session_expression_timeline_from_chunks.py
#
# expression_chunks.v1.json -> session_expression_timeline_v0.1 互換（M0が読む形）
#
# 出力:
# {
#   "schema_version": "session_expression_timeline_v0.1",
#   "session_id": "...",
#   "step_ms": 40,
#   "timeline": [ {t_ms, expression, source, ...}, ... ],
#   "meta": {...}
# }
#
# 追加:
# - auto blink 挿入
#   * emo_id == 9_1 -> 8000ms
#   * emo_id == 9_2 -> 5000ms
#   * その他       -> 10000ms
# - blink は chunk_end_ms までの時間軸で生成
# - blink の直後に 1フレーム(=step_ms) 後の restore event を追加
#   例:
#     t=8000  -> blink
#     t=8040  -> sleepy に戻す

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _blink_interval_ms_from_emo_id(emo_id: str) -> int:
    emo_id = str(emo_id or "")
    if emo_id == "9_1":
        return 8000
    if emo_id == "9_2":
        return 5000
    return 10000


def _sort_and_dedup_timeline(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    timeline.sort(
        key=lambda e: (
            int(e.get("t_ms", 0)),
            str(e.get("expression", "")),
            str(e.get("source", "")),
        )
    )
    dedup: List[Dict[str, Any]] = []
    prev: Tuple[int, str, str] | None = None
    for e in timeline:
        k = (
            int(e.get("t_ms", 0)),
            str(e.get("expression", "")),
            str(e.get("source", "")),
        )
        if k == prev:
            continue
        dedup.append(e)
        prev = k
    return dedup


def _build_absolute_timeline(
    chunks: List[Dict[str, Any]],
    *,
    drop_hold: bool,
) -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []

    for ch in chunks:
        cs = int(ch.get("chunk_start_ms", 0))
        events = ch.get("events", [])
        if not isinstance(events, list):
            continue
        for ev in events:
            if drop_hold and str(ev.get("source", "")) == "hold":
                continue
            rel_t = int(ev.get("t_ms", 0))
            abs_t = cs + rel_t
            out_ev = dict(ev)
            out_ev["t_ms"] = abs_t
            timeline.append(out_ev)

    return _sort_and_dedup_timeline(timeline)


def _insert_blinks_for_chunk(
    chunk: Dict[str, Any],
    abs_events: List[Dict[str, Any]],
    *,
    step_ms: int,
    blink_duration_ms: int,
) -> List[Dict[str, Any]]:
    """
    1 chunk 内の感情イベントをもとに、chunk_end_ms まで blink を自動生成する。

    ルール:
    - 区間ごとに最新の emo_id / expression を採用
    - emo_id ごとに blink 間隔を変える
    - blink は初回 t=0 には入れない
    - blink の直後に restore event を入れる
    """
    chunk_start_ms = int(chunk.get("chunk_start_ms", 0))
    chunk_end_ms = int(chunk.get("chunk_end_ms", chunk_start_ms))

    if chunk_end_ms <= chunk_start_ms:
        return []

    local_events = [
        dict(e) for e in abs_events
        if chunk_start_ms <= int(e.get("t_ms", 0)) < chunk_end_ms
    ]
    local_events.sort(key=lambda e: int(e.get("t_ms", 0)))

    if not local_events:
        return []

    blink_events: List[Dict[str, Any]] = []

    for i, ev in enumerate(local_events):
        seg_start = int(ev.get("t_ms", 0))
        seg_end = (
            int(local_events[i + 1].get("t_ms", 0))
            if i + 1 < len(local_events)
            else chunk_end_ms
        )

        if seg_end <= seg_start:
            continue

        emo_id = str(ev.get("emo_id", ""))
        base_expression = str(ev.get("expression", "normal"))
        emotion_hint = ev.get("emotion_hint", None)

        interval_ms = _blink_interval_ms_from_emo_id(emo_id)

        # 初回 blink は区間開始 + interval から
        t = seg_start + interval_ms

        while t < seg_end:
            t_aligned = (t // step_ms) * step_ms

            if t_aligned > seg_start and t_aligned < seg_end:
                blink_events.append(
                    {
                        "t_ms": t_aligned,
                        "expression": "blink",
                        "source": "auto_blink",
                        "emo_id": emo_id,
                        "emotion_hint": emotion_hint,
                    }
                )

                restore_t = t_aligned + blink_duration_ms
                if restore_t < seg_end:
                    restore_ev = {
                        "t_ms": restore_t,
                        "expression": base_expression,
                        "source": "auto_blink_restore",
                        "emo_id": emo_id,
                    }
                    if emotion_hint is not None:
                        restore_ev["emotion_hint"] = emotion_hint
                    blink_events.append(restore_ev)

            t += interval_ms

    return blink_events


def _insert_blinks(
    *,
    chunks: List[Dict[str, Any]],
    base_timeline: List[Dict[str, Any]],
    step_ms: int,
    blink_duration_ms: int,
) -> List[Dict[str, Any]]:
    out = list(base_timeline)

    for ch in chunks:
        blink_events = _insert_blinks_for_chunk(
            ch,
            base_timeline,
            step_ms=step_ms,
            blink_duration_ms=blink_duration_ms,
        )
        out.extend(blink_events)

    out.sort(
        key=lambda e: (
            int(e.get("t_ms", 0)),
            1 if str(e.get("source", "")) == "auto_blink" else 2 if str(e.get("source", "")) == "auto_blink_restore" else 0,
            str(e.get("expression", "")),
            str(e.get("source", "")),
        )
    )
    return _sort_and_dedup_timeline(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session_id", default="sess_from_chunks")
    ap.add_argument("--drop_hold", action="store_true", help="drop source=='hold' events")

    ap.add_argument("--auto_blink", dest="auto_blink", action="store_true")
    ap.add_argument("--no_auto_blink", dest="auto_blink", action="store_false")
    ap.set_defaults(auto_blink=False)

    ap.add_argument(
        "--blink_duration_ms",
        type=int,
        default=None,
        help="blink duration before restore; default=step_ms",
    )

    args = ap.parse_args()

    in_path = Path(args.in_chunks)
    out_path = Path(args.out)

    obj = load_json(in_path)
    step_ms = int(obj.get("step_ms", 40))
    chunks = obj.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise SystemExit("[ERROR] chunks empty")

    blink_duration_ms = int(args.blink_duration_ms) if args.blink_duration_ms is not None else step_ms
    if blink_duration_ms <= 0:
        raise SystemExit("[ERROR] blink_duration_ms must be > 0")

    base_timeline = _build_absolute_timeline(
        chunks,
        drop_hold=bool(args.drop_hold),
    )

    final_timeline = base_timeline
    if args.auto_blink:
        final_timeline = _insert_blinks(
            chunks=chunks,
            base_timeline=base_timeline,
            step_ms=step_ms,
            blink_duration_ms=blink_duration_ms,
        )

    out_obj = {
        "schema_version": "session_expression_timeline_v0.1",
        "session_id": args.session_id,
        "step_ms": step_ms,
        "timeline": final_timeline,
        "meta": {
            "source": str(in_path).replace("\\", "/"),
            "events_n": len(final_timeline),
            "drop_hold": bool(args.drop_hold),
            "auto_blink": bool(args.auto_blink),
            "blink_duration_ms": blink_duration_ms,
        }
    }

    save_json(out_path, out_obj)
    print("[build_session_expression_timeline_from_chunks][OK]")
    print(" in :", in_path.as_posix())
    print(" out:", out_path.as_posix())
    print(" step_ms:", step_ms)
    print(" timeline:", len(final_timeline))
    print(" auto_blink:", bool(args.auto_blink))
    print(" blink_duration_ms:", blink_duration_ms)


if __name__ == "__main__":
    main()