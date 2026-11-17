"""Export and audit one full remote CFD pilot; no production-data acceptance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--budget-core-h", type=float,
                        help="Explicit budget assumption, not a verified account balance")
    args = parser.parse_args()
    start = time.monotonic()
    expected_tau = np.r_[np.arange(201)*0.4, 80+np.arange(1,61)*4,
                            320+np.arange(1,201)*0.4]
    names = []
    for path in (args.case / "processor0").iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if path.is_dir():
            names.append((value, path.name))
    names.sort()
    observed = np.array([v for v, _ in names])
    if len(names) != 461 or not np.allclose(observed*0.02375/0.01, expected_tau, atol=1e-5, rtol=0):
        raise RuntimeError(f"Retained time schedule mismatch: found {len(names)} states")
    for rank in range(32):
        if any(not (args.case/f"processor{rank}"/name).is_dir() for _,name in names):
            raise RuntimeError(f"Incomplete time series on processor {rank}")
    compact = args.result / "compact"
    compact.mkdir(exist_ok=True)
    records = []
    mask = None
    coordinates = None
    for tau,(t,name) in zip(expected_tau,names):
        # Retain reconstructed and decomposed data on PERUN during this pilot.
        if t != 0:
            with (args.result/"logs/log.reconstruct_export").open("a") as log:
                subprocess.run(["reconstructPar","-case",str(args.case),"-time",name],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        with (args.result/"export_stdout.jsonl").open("a") as log:
            subprocess.run([sys.executable,str(args.exporter),"--case-dir",str(args.case),
                            "--time",name,"--diameter","0.01","--scenario-id","a2_mp_001",
                            "--output-dir",str(compact),"--dimensions","128","64","32",
                            "--bounds","-0.05","0.15","-0.05","0.05","-0.02","0.02"],
                           stdout=log,check=True)
        path = compact/f"a2_mp_001_t{t:g}.npz"
        with np.load(path,allow_pickle=False) as data:
            if data["U"].shape != (128,64,32,3) or data["p"].shape != (128,64,32):
                raise RuntimeError("Unexpected raster shape")
            if not np.isfinite(data["U"]).all() or not np.isfinite(data["p"]).all():
                raise RuntimeError("Non-finite export")
            if mask is None:
                mask=data["valid"].copy()
                coordinates=tuple(data[axis].copy() for axis in ("x","y","z"))
            if not np.array_equal(mask,data["valid"]):
                raise RuntimeError("Fluid mask changed despite static geometry")
            if not all(np.array_equal(old,data[axis]) for old,axis in zip(coordinates,("x","y","z"))):
                raise RuntimeError("Grid coordinates changed")
            record={"tau":float(tau),"time_s":float(data["time_s"]),"filename":path.name,
                    "bytes":path.stat().st_size,"sha256":digest(path),
                    "max_speed_over_inlet":float(np.linalg.norm(data["U"],axis=-1).max()/0.02375),
                    "pressure_min":float(data["p"].min()),"pressure_max":float(data["p"].max())}
        records.append(record)
        print(f"EXPORTED {len(records)}/461 tau={tau:g}",flush=True)
    with (args.result/"series_index.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    fluxes={}
    for patch in ("inletFlux","outletFlux","frontFlux","backFlux"):
        by_time={}
        for path in sorted((args.case/"postProcessing"/patch).glob("*/*.dat")):
            rows=np.loadtxt(path,comments="#",ndmin=2)
            if not np.isfinite(rows).all(): raise RuntimeError("Non-finite flux history")
            by_time.update({float(row[0]):float(row[1]) for row in rows})
        if not by_time: raise RuntimeError(f"Missing flux history: {patch}")
        fluxes[patch]=by_time
    shared=sorted(set.intersection(*(set(v) for v in fluxes.values())))
    if not shared: raise RuntimeError("No matching flux sample times")
    flux_error=max(abs(fluxes['inletFlux'][t]+fluxes['outletFlux'][t])/9.5e-5 for t in shared)
    periodic_error=max(abs(fluxes['frontFlux'][t]+fluxes['backFlux'][t])/9.5e-5 for t in shared)
    force_files=list((args.case/"postProcessing/obstacleForces").glob("*/*.dat"))
    force_rows=0
    for path in force_files:
        rows=np.loadtxt(path,comments="#",ndmin=2)
        if not np.isfinite(rows).all(): raise RuntimeError("Non-finite force history")
        force_rows+=len(rows)
    if not force_rows: raise RuntimeError("Missing force history")
    with (args.result/"segment_costs.tsv").open() as stream:
        segments=list(csv.DictReader(stream,delimiter="\t"))
    solver_h=sum(float(row['core_hours']) for row in segments)
    solver_summary=dict(line.split('=',1) for line in (args.result/'solver_summary.txt').read_text().splitlines() if '=' in line)
    solver_stage_h=float(solver_summary['solver_stage_wall_seconds'])*32/3600
    export_s=time.monotonic()-start
    compact_bytes=sum(row['bytes'] for row in records)
    budget=args.budget_core_h
    if budget is not None and (not np.isfinite(budget) or budget <= 0):
        raise ValueError("Budget assumption must be finite and positive")
    report={"status":"FULL_COST_PILOT_EXPORTED_AND_AUDITED",
            "scenario_id":"a2_mp_001","tau_range":[0,400],"state_count":len(records),
            "solver_core_hours":solver_h,"export_wall_seconds":export_s,
            "solver_stage_allocated_core_hours":solver_stage_h,
            "export_allocated_core_hours":export_s/3600,
            "measured_solver_plus_export_core_hours":solver_stage_h+export_s/3600,
            "compact_npz_bytes":compact_bytes,
            "max_sampled_inlet_outlet_imbalance_over_inlet":flux_error,
            "max_sampled_periodic_imbalance_over_inlet":periodic_error,
            "sampled_flux_screen_passed":flux_error<1e-4 and periodic_error<1e-4,
            "finite_force_history_rows":force_rows,
            "maximum_exported_speed_over_inlet":max(r['max_speed_over_inlet'] for r in records),
            "budget_assumed_cpu_core_h":budget,"remaining_budget_verified":False,
            "illustrative_72_core_h":72*(solver_stage_h+export_s/3600),
            "illustrative_50_core_h":50*(solver_stage_h+export_s/3600),
            "per_scenario_ceiling_assumed_budget_72":budget/72 if budget is not None else None,
            "per_scenario_ceiling_assumed_budget_50":budget/50 if budget is not None else None,
            "production_mesh_accepted":False,
            "limitations":["Single training geometry; different geometries may cost more or less.",
                           "Budget estimates exclude meshing, setup, cancelled runs, prior usage and other checks.",
                           "Long numerical stability and finite outputs do not prove physical accuracy or stationarity.",
                           "Coarse pressure screening differed about 12.85 percent from finer startup reference."]}
    (args.result/"full_pilot_evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
