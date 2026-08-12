"""Unified final dashboard for the nine-method SACSI-POMDP experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from Dashboard.data import (  # noqa: E402
    export_csv,
    export_experiment_zip,
    export_json_bundle,
    export_xlsx_bundle,
    filter_dates,
    load_ablation_summary,
    load_final_logs,
    load_final_registry,
    load_selected_registry,
    load_statistics,
    localize_columns,
    png_chart_config,
    translate,
)
from evaluation import compute_metrics  # noqa: E402


st.set_page_config(page_title="SACSI-POMDP Final Dashboard", layout="wide")
with st.sidebar:
    language = st.selectbox(
        "Language / Bahasa", ("English", "Bahasa Indonesia"), key="ui_language"
    )
t = lambda text: translate(text, language)
st.title(t("SACSI-POMDP · Unified Final Dashboard"))
st.caption(t("Retrospective benchmark 2025 · 9 methods · 10 matched RL seeds · target band 0.22–0.32"))


@st.cache_data
def final_registry():
    return load_final_registry()


@st.cache_data
def final_logs(methods, seed):
    return load_final_logs(list(methods), seed)


@st.cache_data
def ablation_summary():
    return load_ablation_summary()


@st.cache_data
def statistics_tables():
    return load_statistics()


runs, benchmark_summary = final_registry()
method_order = benchmark_summary["method"].tolist()
rl_methods = set(runs.loc[runs["method_type"] == "rl", "method"])

with st.sidebar:
    st.header(t("Final Experiment View"))
    single_label, compare_label = t("Single method"), t("Compare 2–4 methods")
    mode = st.radio("Mode", (single_label, compare_label), key=f"view_mode_{language}")
    single_method = st.selectbox(
        t("Method"), method_order, index=0, disabled=mode != single_label,
        key=f"single_method_{language}",
    )
    compared_methods = st.multiselect(
        t("Methods"), method_order,
        default=["Threshold-Based", "SAC + Forecast", "SACSI Full"],
        max_selections=4,
        disabled=mode == single_label,
        key=f"compared_methods_{language}",
    )
    if mode == single_label:
        selected = [single_method]
    else:
        selected = compared_methods
        if len(selected) < 2:
            st.info(t("Select at least two methods."))
            st.stop()
    has_rl = bool(set(selected) & rl_methods)
    seed_label = st.selectbox(
        t("Matched RL seed"), ("11", "22", "33", "44", "55", "66", "77", "88", "99", "110"),
        disabled=not has_rl, key=f"matched_seed_{language}",
    )
    seed = int(seed_label)
    st.caption(t("The seed selector affects RL methods only; deterministic baselines remain one trajectory."))

logs = final_logs(tuple(selected), seed)
minimum_date, maximum_date = logs["timestamp"].min().date(), logs["timestamp"].max().date()
with st.sidebar:
    dates = st.date_input(
        t("Displayed period"),
        value=(minimum_date, min(minimum_date + pd.Timedelta(days=30), maximum_date)),
        min_value=minimum_date,
        max_value=maximum_date,
    )
if len(dates) != 2:
    st.info(t("Select both start and end dates."))
    st.stop()
view = filter_dates(logs, dates[0], dates[1])
if view.empty:
    st.warning(t("No data in the selected period."))
    st.stop()

metric_rows = [
    {"method": method, **compute_metrics(group)}
    for method, group in view.groupby("controller", sort=False)
]
metric_table = pd.DataFrame(metric_rows)
selected_registry = load_selected_registry(selected, seed)
ablation = ablation_summary()
factorial, pairwise, findings = statistics_tables()

overview_tab, trajectory_tab, ablation_tab, statistics_tab, export_tab = st.tabs((
    t("Overview"), t("Trajectories"), t("Ablation & Robustness"), t("Statistics"), t("Exports"),
))

with overview_tab:
    cards = st.columns(4)
    cards[0].metric(t("Selected methods"), len(selected))
    cards[1].metric(t("Displayed hours"), f"{view['timestamp'].nunique():,}")
    cards[2].metric(t("Best target occupancy"), f"{metric_table['time_in_target_pct'].max():.2f}%")
    cards[3].metric(t("Lowest irrigation"), f"{metric_table['total_irrigation_mm'].min():.2f} mm")

    st.subheader(t("Nine-method final registry"))
    registry_columns = [
        "rank", "method", "n_runs", "time_in_target_pct_mean", "time_in_target_pct_std",
        "total_irrigation_mm_mean", "total_irrigation_mm_std", "rmse_band_mean",
    ]
    st.dataframe(localize_columns(benchmark_summary[registry_columns], language), hide_index=True, width="stretch")

    ranking = px.bar(
        benchmark_summary.sort_values("time_in_target_pct_mean"),
        x="time_in_target_pct_mean", y="method", orientation="h",
        error_x="time_in_target_pct_std", color="method",
        labels={"time_in_target_pct_mean": t("Mean Time in Target (%)"), "method": t("Method")},
    )
    ranking.update_layout(showlegend=False)
    st.plotly_chart(ranking, width="stretch", config=png_chart_config("final_benchmark_ranking"))
    st.caption(t("PNG: hover over a chart and click the camera icon in the Plotly toolbar."))

with trajectory_tab:
    st.subheader(t("Soil moisture and target band"))
    soil = px.line(view, x="timestamp", y="theta", color="controller", labels={"controller": t("Method")})
    soil.add_hrect(y0=0.22, y1=0.32, fillcolor="green", opacity=0.12, line_width=0)
    soil.add_hline(y=0.22, line_dash="dash", line_color="green")
    soil.add_hline(y=0.32, line_dash="dash", line_color="green")
    soil.update_layout(yaxis_title=t("Volumetric soil moisture (m³/m³)"))
    st.plotly_chart(soil, width="stretch", config=png_chart_config("soil_moisture_trajectory"))

    left, right = st.columns(2)
    with left:
        irrigation = px.line(
            view, x="timestamp", y="irrigation_mm", color="controller",
            labels={"irrigation_mm": t("Irrigation (mm/h)"), "controller": t("Method")},
        )
        st.plotly_chart(irrigation, width="stretch", config=png_chart_config("irrigation_trajectory"))
    with right:
        cumulative = px.line(
            view, x="timestamp", y="cumulative_irrigation_mm", color="controller",
            labels={"cumulative_irrigation_mm": t("Cumulative irrigation (mm)"), "controller": t("Method")},
        )
        st.plotly_chart(cumulative, width="stretch", config=png_chart_config("cumulative_water"))

    precipitation = view.drop_duplicates("timestamp")[
        ["timestamp", "actual_precipitation_mm", "forecast_precipitation_mm"]
    ]
    rain = go.Figure()
    rain.add_bar(x=precipitation["timestamp"], y=precipitation["actual_precipitation_mm"], name=t("Actual"))
    rain.add_scatter(
        x=precipitation["timestamp"], y=precipitation["forecast_precipitation_mm"],
        name="SF20 h+1 proxy", mode="lines",
    )
    rain.update_layout(yaxis_title=t("Precipitation (mm/h)"), barmode="overlay")
    st.plotly_chart(rain, width="stretch", config=png_chart_config("rain_actual_forecast"))

    st.subheader(t("Metrics for displayed period"))
    shown = [
        "method", "total_irrigation_mm", "time_in_target_pct", "violation_rate_pct",
        "deficit_rate_pct", "surplus_rate_pct", "rmse_band", "action_smoothness",
        "runoff_total_mm", "drainage_total_mm", "max_abs_mass_balance_error_mm",
    ]
    st.dataframe(localize_columns(metric_table[shown], language), hide_index=True, width="stretch")

with ablation_tab:
    labels = {
        "context_ablation": t("Context ablation"),
        "forecast_robustness": t("Forecast robustness SF10–SF30"),
        "sequence_sensitivity": t("Sequence sensitivity k6–k48"),
    }
    experiment_label = st.selectbox(
        t("Experiment"), list(labels.values()), key=f"ablation_experiment_{language}"
    )
    experiment = {label: code for code, label in labels.items()}[experiment_label]
    experiment_data = ablation.loc[ablation["experiment"] == experiment].copy()
    target_chart = px.bar(
        experiment_data, x="condition", y="time_in_target_pct_mean",
        error_y="time_in_target_pct_std", color="condition",
        labels={"condition": t("Condition"), "time_in_target_pct_mean": t("Mean Time in Target (%)")},
    )
    target_chart.update_layout(showlegend=False)
    st.plotly_chart(target_chart, width="stretch", config=png_chart_config(f"{experiment}_target"))
    st.dataframe(localize_columns(experiment_data, language), hide_index=True, width="stretch")
    st.caption(t("All conditions use the same 10 locked seeds; sequence sensitivity is inference-window sensitivity without retraining."))

with statistics_tab:
    st.subheader(t("Primary endpoint: Time in Target (%)"))
    st.dataframe(localize_columns(factorial, language), hide_index=True, width="stretch")
    st.subheader(t("Pre-specified paired SACSI comparisons"))
    st.dataframe(localize_columns(pairwise, language), hide_index=True, width="stretch")
    if findings["sacsi_superiority_supported"]:
        st.success(t("SACSI superiority is supported for all pre-specified comparisons."))
    else:
        st.info(t("SACSI superiority over all variants is not supported. Technical validity remains separate from superiority."))
    st.caption(t("Deterministic baselines are trajectory references and are not assigned fake stochastic seeds."))

with export_tab:
    st.subheader(t("Export selected experiment view"))
    metadata = {
        "period": [str(dates[0]), str(dates[1])],
        "methods": selected,
        "rl_seed": seed if has_rl else None,
        "benchmark": "retrospective 2025",
    }
    sheets = {
        "trajectory": view,
        "display_metrics": metric_table,
        "selected_registry": selected_registry,
        "benchmark_summary": benchmark_summary,
        "ablation": ablation,
        "factorial_stats": factorial,
        "pairwise_stats": pairwise,
    }
    csv_bytes = export_csv(view)
    xlsx_bytes = export_xlsx_bundle(sheets)
    json_bytes = export_json_bundle(sheets, metadata)
    zip_bytes = export_experiment_zip({
        "trajectory.csv": csv_bytes,
        "experiment.xlsx": xlsx_bytes,
        "experiment.json": json_bytes,
        "benchmark_summary.csv": export_csv(benchmark_summary),
        "ablation_summary.csv": export_csv(ablation),
        "factorial_statistics.csv": export_csv(factorial),
        "pairwise_statistics.csv": export_csv(pairwise),
    })
    filename = f"sacsi_2025_{dates[0]}_{dates[1]}_seed{seed}"
    columns = st.columns(4)
    columns[0].download_button(t("Download CSV"), csv_bytes, f"{filename}.csv", "text/csv")
    columns[1].download_button(
        t("Download XLSX"), xlsx_bytes, f"{filename}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    columns[2].download_button(t("Download JSON"), json_bytes, f"{filename}.json", "application/json")
    columns[3].download_button(t("Download ZIP"), zip_bytes, f"{filename}.zip", "application/zip")
    st.info(t("PNG export is available from the camera icon on every chart toolbar."))

st.caption(t(
    "Forecast display: SF-20 h+1 controlled proxy, not an archived operational forecast. "
    "Checkpoint selection used validation 2024 only; 2025 was retrospective evaluation."
))
