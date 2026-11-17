#!/usr/bin/env python3
"""Emit an OpenCASCADE Gmsh geometry for one Dataset A v2 scenario."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DOMAIN = (-0.05, 0.15, -0.05, 0.05, -0.02, 0.02)


def read_rows(path: Path, scenario_id: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["scenario_id"] == scenario_id]
    if not rows:
        raise SystemExit(f"unknown scenario_id: {scenario_id}")
    return rows


def emit(rows: list[dict[str, str]], mesh_scale: float = 1.0,
         wake_transition_m: float = 0.0) -> str:
    if not math.isfinite(mesh_scale) or mesh_scale <= 0:
        raise ValueError("mesh_scale must be finite and positive")
    if not math.isfinite(wake_transition_m) or wake_transition_m < 0:
        raise ValueError("wake_transition_m must be finite and nonnegative")
    xmin, xmax, ymin, ymax, zmin, zmax = DOMAIN
    eps = 1.0e-7
    lines = [
        'SetFactory("OpenCASCADE");', "Mesh.MshFileVersion = 2.2;", "Mesh.Binary = 0;",
        f"Mesh.MeshSizeFactor = {mesh_scale:.12g};",
        "Mesh.Algorithm = 6;", "Mesh.Algorithm3D = 10;", "Mesh.Optimize = 1;",
        "Mesh.OptimizeNetgen = 1;", "Geometry.OCCBoundsUseStl = 1;",
        f"Box(1) = {{{xmin}, {ymin}, {zmin}, {xmax-xmin}, {ymax-ymin}, {zmax-zmin}}};",
    ]
    tags = []
    for index, row in enumerate(rows, 10):
        shape = row["shape"]
        cx, cy, cz = (float(row[k]) for k in ("center_x_m", "center_y_m", "center_z_m"))
        sx, sy, sz = (float(row[k]) for k in ("size_x_m", "size_y_m", "size_z_m"))
        ax, ay, az = (float(row[k]) for k in ("axis_x", "axis_y", "axis_z"))
        if shape == "sphere":
            lines.append(f"Sphere({index}) = {{{cx}, {cy}, {cz}, {sx/2}}};")
        elif shape == "cylinder":
            norm = math.sqrt(ax*ax + ay*ay + az*az)
            vx, vy, vz = sz*ax/norm, sz*ay/norm, sz*az/norm
            bx, by, bz = cx-vx/2, cy-vy/2, cz-vz/2
            lines.append(f"Cylinder({index}) = {{{bx}, {by}, {bz}, {vx}, {vy}, {vz}, {sx/2}}};")
        elif shape == "box":
            lines.append(f"Box({index}) = {{{cx-sx/2}, {cy-sy/2}, {cz-sz/2}, {sx}, {sy}, {sz}}};")
            angle = math.radians(float(row["rotation_deg"]))
            if abs(angle) > 1e-12:
                lines.append(f"Rotate {{{{0, 0, 1}}, {{{cx}, {cy}, {cz}}}, {angle}}} {{ Volume{{{index}}}; }}")
        else:
            raise SystemExit(f"unsupported shape: {shape}")
        tags.append(index)
    lines += [
        f"fluid[] = BooleanDifference{{ Volume{{1}}; Delete; }}{{ Volume{{{','.join(map(str,tags))}}}; Delete; }};",
        "Coherence;", "eps = 1e-7;",
        f"inlet[] = Surface In BoundingBox {{{xmin-eps}, {ymin-eps}, {zmin-eps}, {xmin+eps}, {ymax+eps}, {zmax+eps}}};",
        f"outlet[] = Surface In BoundingBox {{{xmax-eps}, {ymin-eps}, {zmin-eps}, {xmax+eps}, {ymax+eps}, {zmax+eps}}};",
        f"sideBottom[] = Surface In BoundingBox {{{xmin-eps}, {ymin-eps}, {zmin-eps}, {xmax+eps}, {ymin+eps}, {zmax+eps}}};",
        f"sideTop[] = Surface In BoundingBox {{{xmin-eps}, {ymax-eps}, {zmin-eps}, {xmax+eps}, {ymax+eps}, {zmax+eps}}};",
        f"front[] = Surface In BoundingBox {{{xmin-eps}, {ymin-eps}, {zmin-eps}, {xmax+eps}, {ymax+eps}, {zmin+eps}}};",
        f"back[] = Surface In BoundingBox {{{xmin-eps}, {ymin-eps}, {zmax-eps}, {xmax+eps}, {ymax+eps}, {zmax+eps}}};",
        "obstacle[] = {};",
    ]
    for index, row in enumerate(rows):
        cx, cy, cz = (float(row[k]) for k in ("center_x_m", "center_y_m", "center_z_m"))
        sx, sy, sz = (float(row[k]) for k in ("size_x_m", "size_y_m", "size_z_m"))
        radius = 0.51 * math.sqrt(sx*sx + sy*sy + sz*sz)
        lines += [
            f"obstaclePart{index}[] = Surface In BoundingBox "
            f"{{{cx-radius}, {cy-radius}, {cz-radius}, {cx+radius}, {cy+radius}, {cz+radius}}};",
            f"obstacle[] += obstaclePart{index}[];",
        ]
    lines += [
        'Physical Volume("fluid") = {fluid[]};', 'Physical Surface("inlet") = {inlet[]};',
        'Physical Surface("outlet") = {outlet[]};', 'Physical Surface("sideBottom") = {sideBottom[]};',
        'Physical Surface("sideTop") = {sideTop[]};', 'Physical Surface("front") = {front[]};',
        'Physical Surface("back") = {back[]};', 'Physical Surface("obstacle") = {obstacle[]};',
        f"Periodic Surface {{back[]}} = {{front[]}} Translate {{0, 0, {zmax-zmin}}};",
        "Field[1] = Distance;", "Field[1].FacesList = {obstacle[]};", "Field[1].Sampling = 150;",
        "Field[2] = Threshold;", "Field[2].InField = 1;", "Field[2].SizeMin = 0.00038;",
        "Field[2].SizeMax = 0.0030;", "Field[2].DistMin = 0.0015;", "Field[2].DistMax = 0.030;",
    ]
    xs = [float(r["center_x_m"]) for r in rows]
    ys = [float(r["center_y_m"]) for r in rows]
    scale = max(float(r["size_x_m"]) for r in rows)
    transition = [f"Field[3].Thickness = {wake_transition_m:.12g};"] if wake_transition_m > 0 else []
    lines += [
        "Field[3] = Box;", "Field[3].VIn = 0.0010;", "Field[3].VOut = 0.0030;",
        f"Field[3].XMin = {min(xs)-scale};", f"Field[3].XMax = {min(0.14,max(xs)+10*scale)};",
        f"Field[3].YMin = {max(ymin,min(ys)-2*scale)};", f"Field[3].YMax = {min(ymax,max(ys)+2*scale)};",
        f"Field[3].ZMin = {zmin};", f"Field[3].ZMax = {zmax};",
        *transition,
        "Field[4] = Min;", "Field[4].FieldsList = {2, 3};", "Background Field = 4;",
        "Mesh.MeshSizeExtendFromBoundary = 0;", "Mesh.MeshSizeFromPoints = 0;",
        "Mesh.MeshSizeFromCurvature = 12;", "Mesh 3;",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh-scale", type=float, default=1.0)
    parser.add_argument("--wake-transition-m", type=float, default=0.0)
    args = parser.parse_args()
    args.output.write_text(emit(read_rows(args.obstacles, args.scenario_id), args.mesh_scale,
                               args.wake_transition_m), encoding="utf-8")


if __name__ == "__main__":
    main()
