"""Automated tileset solidity, slope/step semantics, and exit inference across all 191 ROM courses.

This tool executes two offline reverse-engineering passes over the NSMB DS ROM:

1. Empirical & Structural Tileset Solidity Inference:
   - Combines every measured teleport-drop column in `out/floor_probe/*.json` (8 courses)
     and every measured horizontal collision cell in `out/wall_probe/*.json` (5 courses)
     with the parsed `(tileset_byte, obj_id)` rectangles from `course/*_bgdat.bin` and
     the course tileset ID stored in `Blocks[0]` of `course/*.bin`.
   - Computes empirical conditional solidity probabilities `P(solid | course_ts, ts_byte, obj)`
     and `P(passable_horizontal | course_ts, ts_byte, obj)` (which distinguishes walkable slopes
     and background trims from vertical walls).
   - Extends to unprobed tilesets via bottom-band structural recurrence across all 191 courses
     (identifying full-span floor objects whose bottom edge sits consistently on the level's
     walking floor band while excluding sky/ceiling markers at `ty == 0`).

2. Non-Flagpole Exit & Goal Inference:
   - Parses all 14 header blocks of `course/*.bin` across all 191 courses, including
     `Blocks[0]` (tileset/header), `Blocks[1]` (zone bounding boxes), `Blocks[5]` (entrances/warps),
     `Blocks[6]` (sprites), and `Blocks[7..13]` (views/paths/progress zones).
   - Identifies explicit flagpole goals (sprite 32), boss/castle goals, right-edge warp pipes/doors
     (in `Blocks[5]`), and zone-end boundaries (`Blocks[1]`), providing an inferred exit coordinate
     `inferred_exit_x_px` and exit mechanism for all 191 courses without mutating the legacy
     `Course.exit_found` contract.

Outputs:
   - `out/dataset/tileset_solidity.json`
"""
from __future__ import annotations

import collections
import glob
import json
import os
import struct
import sys

import ndspy.rom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from course import GROUND_OBJS, SPRITE_NAMES, TILE  # noqa: E402
from course_id import course_ids  # noqa: E402
from eval_provenance import sha1_file  # noqa: E402
import paths  # noqa: E402


def _parse_course_raw(rom: ndspy.rom.NintendoDSRom, cid: str) -> dict:
    name = f"course/{cid}.bin"
    bg_name = f"course/{cid}_bgdat.bin"
    raw = bytes(rom.getFileByName(name))
    bg = bytes(rom.getFileByName(bg_name))
    blocks = [struct.unpack_from("<II", raw, i * 8) for i in range(14)]
    get = lambda idx: raw[blocks[idx][0] : blocks[idx][0] + blocks[idx][1]]

    # Block 0: area options / tileset ID (commonly byte 12 or u16 at offset 12 in NSMB DS Block 0)
    b0 = get(0)
    tileset_id = int(b0[12]) if len(b0) > 12 else 0

    # Block 1: camera/zone views (16-byte records: x, y, w, h, id, ...)
    b1 = get(1)
    zones = []
    if len(b1) >= 16:
        rec_size = 16 if len(b1) % 16 == 0 else (20 if len(b1) % 20 == 0 else 16)
        for i in range(len(b1) // rec_size):
            zx, zy, zw, zh = struct.unpack_from("<HHHH", b1, i * rec_size)
            zones.append({"x_px": int(zx), "y_px": int(zy), "w_px": int(zw), "h_px": int(zh)})

    # Block 5: entrances / pipe / door warps (20B records)
    b5 = get(5)
    entrances = []
    for i in range(len(b5) // 20):
        ex, ey, cam_x, cam_y, eid, dest_area, dest_ent, etype, flags = struct.unpack_from(
            "<HHHHBBBBH", b5, i * 20
        )
        entrances.append({
            "index": i,
            "x_px": int(ex),
            "y_px": int(ey),
            "id": int(eid),
            "dest_area": int(dest_area),
            "dest_ent": int(dest_ent),
            "type": int(etype),
            "flags": int(flags),
        })

    # Block 6: sprites (12B records + 0xFFFFFFFF trailer)
    b6 = get(6)
    sprites = []
    n_rec = (len(b6) - 4) // 12 if len(b6) >= 4 and (len(b6) - 4) % 12 == 0 else len(b6) // 12
    for i in range(max(0, n_rec)):
        t, x, y = struct.unpack_from("<HHH", b6, i * 12)
        sprites.append({
            "type": int(t),
            "tx": int(x),
            "ty": int(y),
            "x_px": int(x * TILE),
            "y_px": int(y * TILE),
            "name": SPRITE_NAMES.get(int(t), f"id{int(t)}"),
        })

    # bgdat: 10B records (obj u8, ts u8, tx u16, ty u16, w u16, h u16)
    rects = []
    pos = 0
    while pos + 10 <= len(bg):
        obj, ts, x, y, w, h = struct.unpack_from("<BBHHHH", bg, pos)
        if obj == 0xFF:
            break
        rects.append({
            "obj": int(obj),
            "ts": int(ts),
            "tx": int(x),
            "ty": int(y),
            "w": int(w),
            "h": int(h),
        })
        pos += 10

    return {
        "cid": cid,
        "name": name,
        "tileset_id": tileset_id,
        "zones": zones,
        "entrances": entrances,
        "sprites": sprites,
        "rects": rects,
    }


def _load_floor_probes() -> dict[str, dict]:
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "out", "floor_probe", "*.json"))):
        cid = os.path.basename(p).replace(".json", "")
        with open(p, encoding="utf-8") as f:
            out[cid] = json.load(f)
    return out


def _load_wall_probes() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, "out", "wall_probe", "*.json"))):
        base = os.path.basename(p).replace(".json", "")
        cid = "_".join(base.split("_")[:2])
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        out[cid].extend(data.get("cells", []))
    return out


def build_tileset_solidity(rom_path: str | None = None) -> dict:
    rom_abs = paths.resolve(rom_path or paths.ROM_REL)
    rom = ndspy.rom.NintendoDSRom.fromFile(rom_abs)
    cids = course_ids()

    floor_probes = _load_floor_probes()
    wall_probes = _load_wall_probes()

    # Key: (tileset_id, ts_byte, obj_id)
    obj_stats: dict[tuple[int, int, int], dict] = {}

    def _stat(key: tuple[int, int, int]) -> dict:
        if key not in obj_stats:
            obj_stats[key] = {
                "tileset_id": key[0],
                "ts_byte": key[1],
                "obj": key[2],
                "in_static_ground_objs": bool(key[2] in GROUND_OBJS),
                "total_rects": 0,
                "total_cols_covered": 0,
                "bottom_band_cols": 0,
                "sky_ceiling_cols": 0,
                "probe_landed_hits": 0,
                "probe_pit_misses": 0,
                "wall_blocked_hits": 0,
                "wall_free_hits": 0,
                "courses_seen": set(),
            }
        return obj_stats[key]

    parsed_courses = {}
    for cid in cids:
        parsed_courses[cid] = _parse_course_raw(rom, cid)

    # Pass 1: accumulate structural & empirical probe statistics per (tileset_id, ts_byte, obj)
    for cid, cdata in parsed_courses.items():
        ts_id = cdata["tileset_id"]
        rects = cdata["rects"]

        # Estimate the course's walking floor band from rects with ty >= 18
        bottom_ys = [r["ty"] + r["h"] - 1 for r in rects if r["ty"] >= 18 and r["h"] >= 1]
        median_bottom = sorted(bottom_ys)[len(bottom_ys) // 2] if bottom_ys else 30
        floor_band_lo = max(18, median_bottom - 8)

        # Map (col, row) -> list of (ts_byte, obj, rect)
        cell_to_objs: dict[tuple[int, int], list[tuple[int, int, dict]]] = collections.defaultdict(list)
        col_to_objs: dict[int, list[tuple[int, int, dict]]] = collections.defaultdict(list)

        for r in rects:
            key = (ts_id, r["ts"], r["obj"])
            st = _stat(key)
            st["total_rects"] += 1
            st["courses_seen"].add(cid)
            for xx in range(r["tx"], r["tx"] + r["w"]):
                st["total_cols_covered"] += 1
                if r["ty"] < 18:
                    st["sky_ceiling_cols"] += 1
                elif (r["ty"] + r["h"] - 1) >= floor_band_lo:
                    st["bottom_band_cols"] += 1
                col_to_objs[xx].append((r["ts"], r["obj"], r))
                for yy in range(r["ty"], r["ty"] + r["h"]):
                    cell_to_objs[(xx, yy)].append((r["ts"], r["obj"], r))

        # Match floor_probe columns
        if cid in floor_probes:
            fp = floor_probes[cid]
            for col_rec in fp.get("columns", []):
                col = int(col_rec["col"])
                note = col_rec.get("note")
                measured = col_rec.get("measured")
                if note == "landed" and measured is not None:
                    m_row = int(round(float(measured)))
                    # Credit objects whose top or body spans (col, m_row) or (col, m_row - 1 .. m_row + 1)
                    matched_keys = set()
                    for dr in (0, -1, 1):
                        for ts_b, obj_id, r in cell_to_objs.get((col, m_row + dr), []):
                            if r["ty"] >= 18:
                                matched_keys.add((ts_id, ts_b, obj_id))
                    for k in matched_keys:
                        _stat(k)["probe_landed_hits"] += 1
                elif note == "pit":
                    # Any object present at y >= 18 in a measured pit column is non-solid!
                    pit_keys = set()
                    for ts_b, obj_id, r in col_to_objs.get(col, []):
                        if r["ty"] >= 18:
                            pit_keys.add((ts_id, ts_b, obj_id))
                    for k in pit_keys:
                        _stat(k)["probe_pit_misses"] += 1

        # Match wall_probe cells
        if cid in wall_probes:
            for wcell in wall_probes[cid]:
                col = int(wcell["col"])
                row = int(wcell["row"])
                blocked = bool(wcell.get("blocked"))
                for ts_b, obj_id, r in cell_to_objs.get((col, row), []):
                    if r["ty"] >= 18:
                        st = _stat((ts_id, ts_b, obj_id))
                        if blocked:
                            st["wall_blocked_hits"] += 1
                        else:
                            st["wall_free_hits"] += 1

    # Pass 2: classify each (tileset_id, ts_byte, obj)
    # Rules:
    # - Empirical solid if probe_landed_hits > 0 and probe_landed_hits / (probe_landed_hits + probe_pit_misses) >= 0.5
    # - Empirical non-solid if probe_pit_misses > probe_landed_hits and probe_pit_misses >= 2
    # - Walkable slope / step-ramp if empirical solid AND (wall_free_hits > wall_blocked_hits or obj in volcano slope range 19..23)
    # - Structural solid fallback (when unprobed): in GROUND_OBJS or (bottom_band_cols >= 6 and bottom_band_cols / max(1, total_cols_covered) >= 0.5 and sky_ceiling_cols == 0)
    catalog = {}
    per_tileset_solid_objs: dict[int, set[int]] = collections.defaultdict(lambda: set(GROUND_OBJS))
    per_tileset_slope_objs: dict[int, set[int]] = collections.defaultdict(set)
    per_course_slope_objs: dict[str, set[int]] = collections.defaultdict(set)

    for key, st in sorted(obj_stats.items()):
        ts_id, ts_b, obj_id = key
        landed = st["probe_landed_hits"]
        pit_miss = st["probe_pit_misses"]
        probed_total = landed + pit_miss
        p_solid_emp = (landed / probed_total) if probed_total > 0 else None

        bottom_ratio = st["bottom_band_cols"] / max(1, st["total_cols_covered"])
        sky_ratio = st["sky_ceiling_cols"] / max(1, st["total_cols_covered"])

        if p_solid_emp is not None:
            is_solid = bool(p_solid_emp >= 0.5)
            source = "empirical_probe"
        else:
            is_solid = bool(
                st["in_static_ground_objs"]
                or (st["bottom_band_cols"] >= 6 and bottom_ratio >= 0.45 and sky_ratio < 0.25)
            )
            source = "static_ground_objs" if st["in_static_ground_objs"] else (
                "structural_bottom_band" if is_solid else "non_solid"
            )

        # Detect slope / non-blocking ramp semantics:
        # In NSMB DS tilesets, sloped terrain tiles allow horizontal advance (wall_free_hits > 0)
        # or change elevation smoothly across adjacent columns (e.g., volcano/ghost obj 19..23 in H01_1,
        # grass slopes 0x06..0x08 / 0x48..0x4B in A01_1).
        is_slope = False
        if is_solid:
            if st["wall_free_hits"] > st["wall_blocked_hits"]:
                is_slope = True
            elif obj_id in (0x06, 0x07, 0x08, 0x48, 0x49, 0x4A, 0x4B):
                is_slope = True
            elif "H01_1" in st["courses_seen"] and obj_id in (19, 20, 21, 22, 23):
                # In H01_1, obj 19, 21, 22, 23 are volcano slopes (20 is 1x1 block/bridge tile)
                if obj_id in (19, 21, 22, 23):
                    is_slope = True

        if is_solid:
            per_tileset_solid_objs[ts_id].add(obj_id)
        if is_slope:
            per_tileset_slope_objs[ts_id].add(obj_id)
            for c_seen in st["courses_seen"]:
                per_course_slope_objs[c_seen].add(obj_id)

        k_str = f"ts{ts_id}_b{ts_b}_obj{obj_id}"
        catalog[k_str] = {
            "tileset_id": ts_id,
            "ts_byte": ts_b,
            "obj": obj_id,
            "is_solid": is_solid,
            "is_slope": is_slope,
            "classification_source": source,
            "p_solid_empirical": round(p_solid_emp, 4) if p_solid_emp is not None else None,
            "probe_landed_hits": landed,
            "probe_pit_misses": pit_miss,
            "wall_blocked_hits": st["wall_blocked_hits"],
            "wall_free_hits": st["wall_free_hits"],
            "total_rects": st["total_rects"],
            "total_cols_covered": st["total_cols_covered"],
            "bottom_band_ratio": round(bottom_ratio, 4),
            "courses_count": len(st["courses_seen"]),
        }

    # Pass 3: Compute per-course inferred coverage and non-flagpole exit inference across all 191 courses
    courses_summary = {}
    static_plannable_count = 0
    inferred_plannable_count = 0

    for cid, cdata in parsed_courses.items():
        ts_id = cdata["tileset_id"]
        rects = cdata["rects"]
        sprites = cdata["sprites"]
        entrances = cdata["entrances"]
        zones = cdata["zones"]

        xs = [r["tx"] + r["w"] for r in rects] + [r["tx"] for r in rects]
        span = (max(xs) - min(xs)) if xs else 0
        span_px = float(max((r["tx"] + r["w"] for r in rects), default=0) * TILE)

        # Static cover (exact match to Course.ground_cover)
        static_cols = set()
        inferred_cols = set()
        solid_set = per_tileset_solid_objs[ts_id]

        for r in rects:
            is_stat = r["obj"] in GROUND_OBJS
            is_inf = is_stat or (r["obj"] in solid_set) or ((r["ty"] >= 18) and (r["ty"] + r["h"] >= 26))
            for xx in range(r["tx"], r["tx"] + r["w"]):
                for yy in range(r["ty"], r["ty"] + r["h"]):
                    if yy >= 18:
                        if is_stat:
                            static_cols.add(xx)
                        if is_inf:
                            inferred_cols.add(xx)

        # Apply floor_probe overrides if present and enabled
        if cid in floor_probes and floor_probes[cid].get("apply_overrides") is not False:
            cols_p = floor_probes[cid].get("columns", [])
            dom = next((c.get("floor_row_dominant") for c in cols_p if c.get("floor_row_dominant") is not None), None)
            if dom is not None:
                for c in cols_p:
                    m, note, p = c.get("measured"), c.get("note"), c.get("parser")
                    if note == "landed" and m is not None and p is None:
                        if (dom - m) <= 5 and (m - dom) <= 3:
                            static_cols.add(int(c["col"]))
                            inferred_cols.add(int(c["col"]))
                    elif note == "pit" and p is not None:
                        static_cols.discard(int(c["col"]))

        static_cov = (len(static_cols) / span) if span else 0.0
        inferred_cov = (len(inferred_cols) / span) if span else 0.0

        # Exit detection:
        # 1. Explicit flagpole (sprite 32 or "Flag" in name within span_px)
        flag_cands = [s for s in sprites if s["type"] == 32 or "Flag" in s["name"]]
        in_span_flags = [s for s in flag_cands if s["x_px"] <= span_px]
        static_exit_found = bool(in_span_flags)

        if in_span_flags:
            inferred_exit_px = float(max(s["x_px"] for s in in_span_flags))
            exit_kind = "flagpole_sprite_32"
            inferred_exit_found = True
        else:
            # 2. Check right-most warp entrance / pipe / door in Blocks[5] beyond spawn
            spawn_px = float(entrances[0]["x_px"]) if entrances else 0.0
            right_entrances = [
                e for e in entrances[1:]
                if e["x_px"] > spawn_px + 256 and e["x_px"] <= span_px
            ]
            # 3. Check zone right edge in Blocks[1]
            zone_right_px = max((z["x_px"] + z["w_px"] for z in zones), default=0)
            if right_entrances:
                best_ent = max(right_entrances, key=lambda e: e["x_px"])
                inferred_exit_px = float(best_ent["x_px"])
                exit_kind = f"warp_entrance_type_{best_ent['type']}"
                inferred_exit_found = True
            elif zone_right_px > spawn_px + 256 and zone_right_px <= span_px + 64:
                inferred_exit_px = float(zone_right_px - 32.0)
                exit_kind = "zone_right_boundary"
                inferred_exit_found = True
            elif inferred_cols:
                inferred_exit_px = float(max(inferred_cols) * TILE)
                exit_kind = "rightmost_solid_column"
                inferred_exit_found = True
            else:
                inferred_exit_px = 4032.0
                exit_kind = "fallback_4032"
                inferred_exit_found = False

        if static_cov >= 0.6 and static_exit_found:
            static_plannable_count += 1
        if inferred_cov >= 0.6 and inferred_exit_found:
            inferred_plannable_count += 1

        courses_summary[cid] = {
            "tileset_id": ts_id,
            "span_tiles": span,
            "span_px": span_px,
            "static_ground_coverage": round(static_cov, 4),
            "inferred_ground_coverage": round(inferred_cov, 4),
            "static_exit_found": static_exit_found,
            "inferred_exit_found": inferred_exit_found,
            "inferred_exit_x_px": round(inferred_exit_px, 1),
            "inferred_exit_kind": exit_kind,
            "inferred_solid_objs": sorted(per_tileset_solid_objs[ts_id]),
            "inferred_slope_objs": sorted(per_course_slope_objs.get(cid, per_tileset_slope_objs[ts_id])),
        }

    result = {
        "schema": 1,
        "rom_sha1": sha1_file(rom_abs),
        "summary": {
            "total_courses": len(cids),
            "probed_floor_courses": sorted(floor_probes.keys()),
            "probed_wall_courses": sorted(wall_probes.keys()),
            "total_catalog_entries": len(catalog),
            "static_plannable_courses": static_plannable_count,
            "inferred_plannable_courses": inferred_plannable_count,
            "examples": {
                k: courses_summary[k]
                for k in ("A01_1", "A02_1", "A05_1", "A08_1", "B01_1", "D01_1", "E01_1", "F01_1", "G01_1", "H01_1")
                if k in courses_summary
            },
        },
        "tileset_solid_objs": {str(k): sorted(v) for k, v in sorted(per_tileset_solid_objs.items())},
        "tileset_slope_objs": {str(k): sorted(v) for k, v in sorted(per_tileset_slope_objs.items())},
        "courses": courses_summary,
        "object_catalog": catalog,
    }
    return result


def main():
    out_path = os.path.join(ROOT, "out", "dataset", "tileset_solidity.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = build_tileset_solidity()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    s = data["summary"]
    print(f"Wrote {out_path}")
    print(
        f"Courses: {s['total_courses']} | "
        f"Static plannable: {s['static_plannable_courses']}/{s['total_courses']} | "
        f"Inferred plannable: {s['inferred_plannable_courses']}/{s['total_courses']}"
    )
    for k, ex in s["examples"].items():
        print(
            f"  {k}: static_cov={ex['static_ground_coverage']*100:5.1f}% -> "
            f"inferred_cov={ex['inferred_ground_coverage']*100:5.1f}% | "
            f"exit={ex['inferred_exit_x_px']}px ({ex['inferred_exit_kind']}) | "
            f"slopes={ex['inferred_slope_objs']}"
        )


if __name__ == "__main__":
    main()
