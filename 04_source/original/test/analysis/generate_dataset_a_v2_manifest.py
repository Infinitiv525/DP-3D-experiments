#!/usr/bin/env python3
"""Create the frozen, deterministic Dataset A v2 scenario and obstacle tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

NU = 1.0e-6
DOMAIN = (-0.05, 0.15, -0.05, 0.05, -0.02, 0.02)

FAMILY_LAYOUT = {
    "spanning_cylinder": (12, ["train"] * 8 + ["validation", "test_ID"] + ["test_OOD"] * 2),
    "sphere": (12, ["train"] * 8 + ["validation", "test_ID"] + ["test_OOD"] * 2),
    "finite_cylinder": (12, ["train"] * 8 + ["validation", "test_ID"] + ["test_OOD"] * 2),
    "box": (12, ["train"] * 8 + ["validation", "test_ID"] + ["test_OOD"] * 2),
    "cylinder_pair": (16, ["train"] * 10 + ["validation"] * 2 + ["test_ID"] * 2 + ["test_OOD"] * 2),
    "mixed_pair": (8, ["train"] * 4 + ["validation"] * 2 + ["test_ID"] * 2),
}


def f(value: float) -> str:
    return f"{value:.8g}"


def base_parameters(family_index: int, local_index: int, split: str) -> tuple[float, float, float, float]:
    if split == "test_OOD":
        u = [0.0155, 0.0275][local_index % 2]
        length = [0.0075, 0.0125][(local_index // 2 + family_index) % 2]
        cx = [-0.012, 0.012][local_index % 2]
        cy = [-0.014, 0.014][(local_index + family_index) % 2]
    else:
        u = 0.0175 + 0.00125 * ((3 * local_index + family_index) % 7)
        length = 0.0085 + 0.0005 * ((5 * local_index + 2 * family_index) % 7)
        cx = -0.0075 + 0.0025 * ((2 * local_index + family_index) % 7)
        cy = -0.009 + 0.003 * ((4 * local_index + family_index) % 7)
    return u, length, cx, cy


def obstacle(sid: str, oid: int, shape: str, cx: float, cy: float, cz: float,
             sx: float, sy: float, sz: float, ax: float = 0.0,
             ay: float = 0.0, az: float = 1.0, rotation: float = 0.0) -> dict[str, str]:
    return {
        "scenario_id": sid, "obstacle_id": str(oid), "shape": shape,
        "center_x_m": f(cx), "center_y_m": f(cy), "center_z_m": f(cz),
        "size_x_m": f(sx), "size_y_m": f(sy), "size_z_m": f(sz),
        "axis_x": f(ax), "axis_y": f(ay), "axis_z": f(az),
        "rotation_deg": f(rotation), "wall_patch": "obstacle",
    }


def make_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    scenarios: list[dict[str, str]] = []
    obstacles: list[dict[str, str]] = []
    family_codes = {"spanning_cylinder": "sc", "sphere": "sp", "finite_cylinder": "fc",
                    "box": "bx", "cylinder_pair": "cp", "mixed_pair": "mp"}
    for family_index, (family, (count, splits)) in enumerate(FAMILY_LAYOUT.items()):
        assert count == len(splits)
        for i, split in enumerate(splits):
            sid = f"a2_{family_codes[family]}_{i + 1:03d}"
            u, length, cx, cy = base_parameters(family_index, i, split)
            arrangement = "single"
            ood_axis = "none" if split != "test_OOD" else "velocity_size_position_extrapolation"

            if family == "spanning_cylinder":
                obstacles.append(obstacle(sid, 1, "cylinder", cx, cy, 0.0,
                                          length, length, 0.042))
            elif family == "sphere":
                obstacles.append(obstacle(sid, 1, "sphere", cx, cy, 0.0,
                                          length, length, length))
            elif family == "finite_cylinder":
                tilt = math.radians([0, 12, 20, 28][i % 4])
                axis = (math.sin(tilt), 0.0, math.cos(tilt))
                finite_length = min(0.030, 2.0 * length + 0.002 * (i % 3))
                obstacles.append(obstacle(sid, 1, "cylinder", cx, cy, 0.0,
                                          length, length, finite_length, *axis))
            elif family == "box":
                obstacles.append(obstacle(sid, 1, "box", cx, cy, 0.0,
                                          1.15 * length, 0.85 * length,
                                          1.6 * length, rotation=[0, 15, 30, 45][i % 4]))
            elif family == "cylinder_pair":
                arrangement = ["tandem", "side_by_side", "staggered"][i % 3]
                gap = (1.45 + 0.18 * (i % 4)) * length
                if arrangement == "tandem":
                    offsets = [(-gap / 2, 0.0), (gap / 2, 0.0)]
                elif arrangement == "side_by_side":
                    offsets = [(0.0, -gap / 2), (0.0, gap / 2)]
                else:
                    offsets = [(-gap / 2, -0.35 * gap), (gap / 2, 0.35 * gap)]
                for oid, (dx, dy) in enumerate(offsets, 1):
                    d = length * (1.0 if oid == 1 else 0.82 + 0.06 * (i % 3))
                    obstacles.append(obstacle(sid, oid, "cylinder", cx + dx, cy + dy,
                                              0.0, d, d, 0.042))
                if split == "test_OOD":
                    ood_axis = "pair_spacing_plus_parameter_extrapolation"
            elif family == "mixed_pair":
                arrangement = ["sphere_box_tandem", "sphere_finite_staggered"][i % 2]
                gap = (1.7 + 0.15 * (i % 3)) * length
                obstacles.append(obstacle(sid, 1, "sphere", cx - gap / 2, cy - 0.2 * gap,
                                          0.0, length, length, length))
                if i % 2 == 0:
                    obstacles.append(obstacle(sid, 2, "box", cx + gap / 2, cy + 0.2 * gap,
                                              0.0, 1.05 * length, 0.8 * length,
                                              1.4 * length, rotation=30.0))
                else:
                    tilt = math.radians(18.0)
                    obstacles.append(obstacle(sid, 2, "cylinder", cx + gap / 2,
                                              cy + 0.2 * gap, 0.0, 0.9 * length,
                                              0.9 * length, 2.0 * length,
                                              math.sin(tilt), 0.0, math.cos(tilt)))

            t_scale = length / u
            scenarios.append({
                "scenario_id": sid, "split": split, "family": family,
                "arrangement": arrangement, "u_inlet_m_s": f(u),
                "reference_length_m": f(length), "reynolds": f(u * length / NU),
                "nu_m2_s": f(NU), "t_end_s": f(400.0 * t_scale),
                "startup_dense_end_s": f(80.0 * t_scale),
                "developed_dense_start_s": f(320.0 * t_scale),
                "dense_write_interval_s": f(0.4 * t_scale),
                "sparse_write_interval_s": f(4.0 * t_scale),
                "expected_retained_states": "461", "ood_axis": ood_axis,
                "test_locked": "true" if split.startswith("test") else "false",
            })
    return scenarios, obstacles


def validate(scenarios: list[dict[str, str]], obstacles: list[dict[str, str]]) -> dict[str, object]:
    split_counts = Counter(row["split"] for row in scenarios)
    family_counts = Counter(row["family"] for row in scenarios)
    assert len(scenarios) == 72
    assert split_counts == Counter({"train": 46, "validation": 8, "test_ID": 8, "test_OOD": 10})
    assert family_counts == Counter({"cylinder_pair": 16, "spanning_cylinder": 12,
                                     "sphere": 12, "finite_cylinder": 12,
                                     "box": 12, "mixed_pair": 8})
    assert len({row["scenario_id"] for row in scenarios}) == len(scenarios)
    ids = {row["scenario_id"] for row in scenarios}
    assert {row["scenario_id"] for row in obstacles} == ids
    for row in obstacles:
        x, y, z = (float(row[k]) for k in ("center_x_m", "center_y_m", "center_z_m"))
        sx, sy, sz = (float(row[k]) for k in ("size_x_m", "size_y_m", "size_z_m"))
        assert -0.035 < x - sx / 2 and x + sx / 2 < 0.06
        assert -0.04 < y - sy / 2 and y + sy / 2 < 0.04
        if row["shape"] != "cylinder" or sz < 0.04:
            assert -0.0195 < z - sz / 2 and z + sz / 2 < 0.0195
    return {"scenario_count": len(scenarios), "obstacle_count": len(obstacles),
            "split_counts": dict(split_counts), "family_counts": dict(family_counts),
            "status": "MANIFEST_VALID"}


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios, obstacles = make_rows()
    audit = validate(scenarios, obstacles)
    write_csv(args.output_dir / "dataset_a_v2_scenarios.csv", scenarios)
    write_csv(args.output_dir / "dataset_a_v2_obstacles.csv", obstacles)
    (args.output_dir / "dataset_a_v2_manifest_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
