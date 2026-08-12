"""Testable data access and export helpers for both dashboard versions."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "Results" / "baseline_metrics.csv"
LOGS_DIR = ROOT / "Logs" / "Baselines"
FINAL_RUNS = ROOT / "Results" / "Final_Experiment" / "benchmark_2025_runs.csv"
FINAL_SUMMARY = ROOT / "Results" / "Final_Experiment" / "benchmark_2025_summary.csv"
FINAL_LOGS = ROOT / "Logs" / "Final_Experiment"
ABLATION_SUMMARY = ROOT / "Results" / "Ablation_Robustness" / "ablation_robustness_summary.csv"
FACTORIAL_STATS = ROOT / "Statistics" / "factorial_rm_anova_primary.csv"
PAIRWISE_STATS = ROOT / "Statistics" / "pairwise_sacsi_primary.csv"
STATISTICAL_FINDINGS = ROOT / "Statistics" / "statistical_findings.json"
WEATHER_2025 = ROOT / "00_Dataset" / "Processed" / "benchmark_2025.csv"
SF20 = ROOT / "00_Dataset" / "Processed" / "synthetic_forecast_sf20.csv"


def controller_slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def method_slug(name: str) -> str:
    return name.lower().replace(" + ", "_plus_").replace("-", "_").replace(" ", "_")


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(METRICS_PATH)


def load_logs(split: str, controllers: list[str]) -> pd.DataFrame:
    frames = []
    for controller in controllers:
        path = LOGS_DIR / f"{split}_{controller_slug(controller)}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Run baseline benchmark first: missing {path.name}")
        frames.append(pd.read_csv(path, parse_dates=["timestamp"]))
    return pd.concat(frames, ignore_index=True)


def load_final_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(FINAL_RUNS)
    summary = pd.read_csv(FINAL_SUMMARY).sort_values("rank").reset_index(drop=True)
    if len(runs) != 45 or summary["method"].nunique() != 9:
        raise ValueError("Final registry must contain 45 runs and nine methods")
    return runs, summary


def load_selected_registry(methods: list[str], seed: int) -> pd.DataFrame:
    runs, _ = load_final_registry()
    baseline = runs["method_type"].eq("baseline") & runs["method"].isin(methods)
    rl = (
        runs["method_type"].eq("rl")
        & runs["method"].isin(methods)
        & pd.to_numeric(runs["seed"], errors="coerce").eq(seed)
    )
    selected = runs.loc[baseline | rl].copy()
    if set(selected["method"]) != set(methods):
        raise ValueError("Selected method/seed is missing from the final registry")
    return selected


def _weather_context() -> pd.DataFrame:
    weather = pd.read_csv(WEATHER_2025, usecols=["timestamp", "precipitation_mm"], parse_dates=["timestamp"])
    weather = weather.rename(columns={"precipitation_mm": "actual_precipitation_mm"})
    forecast = pd.read_csv(
        SF20,
        usecols=["timestamp", "forecast_precipitation_mm"],
        parse_dates=["timestamp"],
    )
    forecast = forecast.loc[forecast["timestamp"].dt.year == 2025]
    return weather.merge(forecast, on="timestamp", how="left", validate="one_to_one")


def load_final_logs(methods: list[str], seed: int) -> pd.DataFrame:
    registry = load_selected_registry(methods, seed)
    frames = []
    for row in registry.itertuples(index=False):
        suffix = f"_seed{seed}" if row.method_type == "rl" else ""
        path = FINAL_LOGS / f"benchmark_2025_{method_slug(row.method)}{suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Run final benchmark first: missing {path.name}")
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        frame["controller"] = row.method
        frame["display_seed"] = str(seed) if row.method_type == "rl" else "deterministic"
        frames.append(frame)
    logs = pd.concat(frames, ignore_index=True)
    logs = logs.merge(_weather_context(), on="timestamp", how="left", validate="many_to_one")
    if logs[["actual_precipitation_mm", "forecast_precipitation_mm"]].isna().any().any():
        raise ValueError("Weather enrichment failed on final common support")
    logs = logs.sort_values(["controller", "timestamp"]).reset_index(drop=True)
    logs["cumulative_irrigation_mm"] = logs.groupby("controller")["irrigation_mm"].cumsum()
    if not logs.groupby("controller")["timestamp"].nunique().eq(8_760).all():
        raise ValueError("Every selected method must have 8,760 benchmark hours")
    return logs


def load_ablation_summary() -> pd.DataFrame:
    return pd.read_csv(ABLATION_SUMMARY)


def load_statistics() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return (
        pd.read_csv(FACTORIAL_STATS),
        pd.read_csv(PAIRWISE_STATS),
        json.loads(STATISTICAL_FINDINGS.read_text(encoding="utf-8")),
    )


def filter_dates(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end_exclusive = pd.Timestamp(end) + pd.Timedelta(days=1)
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end_exclusive)].copy()


def export_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def export_json(frame: pd.DataFrame) -> bytes:
    return frame.to_json(orient="records", date_format="iso", indent=2).encode("utf-8")


def export_xlsx(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="simulation")
    return output.getvalue()


def export_xlsx_bundle(sheets: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=name[:31])
    return output.getvalue()


def export_json_bundle(frames: dict[str, pd.DataFrame], metadata: dict) -> bytes:
    payload = {
        "metadata": metadata,
        **{name: json.loads(frame.to_json(orient="records", date_format="iso")) for name, frame in frames.items()},
    }
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def export_experiment_zip(files: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def png_chart_config(filename: str) -> dict:
    """Browser-side PNG export; avoids a server Chrome/Kaleido dependency."""
    return {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png", "filename": filename, "height": 720, "width": 1280, "scale": 2,
        },
    }
