"""Compute field and scenario-parameter normalization from train scenarios only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from train_baseline_smoke import calculate_training_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_split = str(config["training_split"])
    parameters: list[list[float]] = []
    scenarios: list[str] = []
    protected: list[str] = []
    with Path(config["scenario_table"]).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == train_split:
                scenarios.append(row["scenario_id"])
                parameters.append(
                    [
                        float(row["u_m_s"]),
                        float(row["diameter_m"]),
                        float(row["center_x_m"]),
                        float(row["center_y_m"]),
                    ]
                )
            else:
                protected.append(row["scenario_id"])
    statistics = calculate_training_statistics(Path(config["dataset_root"]), scenarios)
    parameter_array = np.asarray(parameters, dtype=np.float64)
    parameter_std = np.std(parameter_array, axis=0)
    if np.any(parameter_std <= 0):
        raise RuntimeError("A scenario parameter has zero training variance")
    statistics["scenario_parameter_names"] = ["u_m_s", "diameter_m", "center_x_m", "center_y_m"]
    statistics["scenario_parameter_mean"] = np.mean(parameter_array, axis=0).tolist()
    statistics["scenario_parameter_std"] = parameter_std.tolist()
    statistics["protected_scenarios_not_read"] = protected
    statistics["dataset_root"] = str(config["dataset_root"])
    output = Path(config["training_statistics"])
    output.write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(statistics, indent=2))
    print("FNO_TRAINING_STATISTICS_OK")


if __name__ == "__main__":
    main()
