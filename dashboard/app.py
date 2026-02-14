from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.io import (
    collect_artifact_paths,
    format_timestamp,
    infer_stage_status,
    parse_curve_string,
    read_csv_optional,
    read_json_optional,
    repo_branch,
    repo_commit,
)


st.set_page_config(
    page_title="Radiology XAI Thesis Dashboard",
    page_icon=":microscope:",
    layout="wide",
)


STATUS_LABELS = {
    "done": "Done",
    "partial": "Partial",
    "not_started": "Not Started",
}
STATUS_EMOJI = {
    "done": "✅",
    "partial": "🟨",
    "not_started": "⬜",
}

THESIS_CROSSWALK: dict[str, dict[str, object]] = {
    "pipeline_overview": {
        "title": "Pipeline Overview",
        "focus": "End-to-end pipeline logic, with saliency and concept families under a unified faithfulness protocol.",
        "sections": [
            "Methodology -> Overview",
            "Methodology -> Unified Faithfulness Protocol",
            "Experimental Design -> Implementation Status at Draft Time",
        ],
        "files": [
            "thesis/sections/05_methodology.tex",
            "thesis/sections/06_experimental_design.tex",
        ],
    },
    "data_and_config": {
        "title": "Data and Config",
        "focus": "Dataset assumptions, cohort construction, and reproducibility controls.",
        "sections": [
            "Experimental Design -> Datasets and Splits",
            "Experimental Design -> Cohort Construction",
            "Reproducibility Checklist -> Data Processing",
        ],
        "files": [
            "thesis/sections/06_experimental_design.tex",
            "thesis/sections/A_reproducibility.tex",
        ],
    },
    "e1_models": {
        "title": "E1 Models",
        "focus": "Diagnostic baseline behavior used as foundation for explanation evaluation.",
        "sections": [
            "Methodology -> Common Diagnostic Backbone",
            "Experimental Design -> Primary Metrics -> Diagnostic Metrics",
            "Results -> Diagnostic Performance",
        ],
        "files": [
            "thesis/sections/05_methodology.tex",
            "thesis/sections/06_experimental_design.tex",
            "thesis/sections/07_results.tex",
        ],
    },
    "faithfulness_e23_e7": {
        "title": "Faithfulness (E2/E3 + E7)",
        "focus": "Saliency-versus-concept explanation benchmarking with component-first interpretation and secondary NFI summaries.",
        "sections": [
            "Methodology -> Explanation Families",
            "Methodology -> Core Test 1: Sanity Checks",
            "Methodology -> Deletion and Insertion Tests",
            "Methodology -> Nuisance Robustness",
            "Methodology -> Cross-Family Normalized Faithfulness Index",
            "Results -> Faithfulness Benchmark Results",
        ],
        "files": [
            "thesis/sections/05_methodology.tex",
            "thesis/sections/07_results.tex",
        ],
    },
    "e8_randomization": {
        "title": "E8 Randomization",
        "focus": "Randomization stress tests and sanity sensitivity analysis.",
        "sections": [
            "Methodology -> Core Test 1: Sanity Checks",
            "Experimental Design -> Ablation and Sensitivity Studies",
            "Experimental Design -> Failure Analysis and Quality Gates",
        ],
        "files": [
            "thesis/sections/05_methodology.tex",
            "thesis/sections/06_experimental_design.tex",
        ],
    },
    "sample_inspector": {
        "title": "Sample Inspector",
        "focus": "Case-level behavior inspection for debugging and qualitative validation.",
        "sections": [
            "Results -> Faithfulness Benchmark Results",
            "Discussion -> Interpretation of Findings",
            "Discussion -> Failure Modes and Limitations",
        ],
        "files": [
            "thesis/sections/07_results.tex",
            "thesis/sections/08_discussion.tex",
        ],
    },
}


def _status_text(status: str) -> str:
    return f"{STATUS_EMOJI.get(status, '⬜')} {STATUS_LABELS.get(status, status)}"


def _read_text_excerpt(path: Path, max_lines: int = 80) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""

    lines = text.splitlines()
    return "\n".join(lines[:max_lines])


def _latex_inline_to_markdown(text: str) -> str:
    value = text

    for _ in range(4):
        previous = value
        value = re.sub(r"\\textbf\{([^{}]+)\}", r"**\1**", value)
        value = re.sub(r"\\textit\{([^{}]+)\}", r"*\1*", value)
        value = re.sub(r"\\emph\{([^{}]+)\}", r"*\1*", value)
        value = re.sub(r"\\verb\|([^|]+)\|", r"`\1`", value)
        if value == previous:
            break

    value = value.replace(r"\textbf{", "").replace(r"\textit{", "").replace(r"\emph{", "")
    value = re.sub(r"\\cite\w*\{([^{}]+)\}", r"[citation: \1]", value)
    value = re.sub(r"\\label\{[^{}]*\}", "", value)
    value = value.replace(r"\%", "%").replace(r"\_", "_").replace(r"\&", "&")
    value = value.replace(r"\{", "{").replace(r"\}", "}")
    value = re.sub(r"\\addcontentsline\{[^{}]*\}\{[^{}]*\}\{[^{}]*\}", "", value)
    value = re.sub(r"\\(noindent|clearpage|par)\b", "", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _latex_excerpt_to_markdown(excerpt: str) -> str:
    lines = excerpt.splitlines()
    rendered: list[str] = []
    in_equation = False
    equation_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            if in_equation:
                equation_lines.append("")
            elif rendered and rendered[-1] != "":
                rendered.append("")
            continue

        if line.startswith("%"):
            continue

        if line.startswith(r"\begin{equation}"):
            in_equation = True
            equation_lines = []
            continue

        if line.startswith(r"\end{equation}"):
            in_equation = False
            if equation_lines:
                rendered.append("```latex")
                rendered.extend(equation_lines)
                rendered.append("```")
            continue

        if in_equation:
            equation_lines.append(raw_line.rstrip())
            continue

        match = re.match(r"\\chapter\*?\{(.+)\}", line)
        if match:
            rendered.append(f"### {match.group(1).strip()}")
            continue

        match = re.match(r"\\section\{(.+)\}", line)
        if match:
            rendered.append(f"#### {match.group(1).strip()}")
            continue

        match = re.match(r"\\subsection\{(.+)\}", line)
        if match:
            rendered.append(f"##### {match.group(1).strip()}")
            continue

        match = re.match(r"\\paragraph\{(.+)\}", line)
        if match:
            rendered.append(f"**{match.group(1).strip()}**")
            continue

        if line in {r"\begin{itemize}", r"\end{itemize}", r"\begin{enumerate}", r"\end{enumerate}"}:
            continue

        if line.startswith(r"\item"):
            cleaned_item = _latex_inline_to_markdown(line[len(r"\item"):].strip())
            if cleaned_item:
                rendered.append(f"- {cleaned_item}")
            continue

        if "$" in line:
            rendered.append(f"`{line}`")
            continue

        cleaned = _latex_inline_to_markdown(line)
        if cleaned:
            rendered.append(cleaned)

    compact: list[str] = []
    for item in rendered:
        if item == "" and compact and compact[-1] == "":
            continue
        compact.append(item)
    return "\n".join(compact).strip()


def _render_thesis_reference(project_root: Path, key: str) -> None:
    entry = THESIS_CROSSWALK.get(key)
    if entry is None:
        return

    title = str(entry.get("title", "Thesis Crosswalk"))
    focus = str(entry.get("focus", ""))
    sections = entry.get("sections", [])
    file_entries = entry.get("files", [])
    section_items = sections if isinstance(sections, list) else []
    rel_files = file_entries if isinstance(file_entries, list) else []

    with st.expander(f"Thesis Crosswalk: {title}", expanded=False):
        if focus:
            st.caption(focus)

        existing_files: list[Path] = []
        if rel_files:
            for rel_path in rel_files:
                chapter_path = project_root / str(rel_path)
                if chapter_path.exists():
                    existing_files.append(chapter_path)

        if existing_files:
            option_map = {path.name: path for path in existing_files}
            preview_mode = st.radio(
                "Preview mode",
                options=["Readable", "Raw LaTeX"],
                horizontal=True,
                key=f"thesis_preview_mode_{key}",
            )
            selected = st.selectbox(
                "Preview chapter source excerpt",
                options=list(option_map.keys()),
                key=f"thesis_preview_{key}",
            )
            preview_lines = st.slider(
                "Excerpt lines",
                min_value=30,
                max_value=220,
                value=90,
                step=10,
                key=f"thesis_preview_lines_{key}",
            )
            selected_path = option_map[selected]
            excerpt = _read_text_excerpt(selected_path, max_lines=preview_lines)
            if excerpt:
                if preview_mode == "Readable":
                    rendered = _latex_excerpt_to_markdown(excerpt)
                    if rendered:
                        st.markdown(rendered)
                    else:
                        st.info("No readable excerpt content found.")
                else:
                    st.code(excerpt, language="tex")
        else:
            st.info("No chapter source files found for this crosswalk entry.")

        st.markdown("---")

        if section_items:
            st.markdown("**Best Sections To Cite**")
            for item in section_items:
                st.markdown(f"- {item}")

        pdf_path = project_root / "thesis" / "main.pdf"
        if pdf_path.exists():
            st.link_button("Open Full Thesis PDF", pdf_path.as_uri(), use_container_width=False)

        if rel_files:
            st.markdown("**Chapter Source Files**")
            for rel_path in rel_files:
                chapter_path = project_root / str(rel_path)
                if chapter_path.exists():
                    st.markdown(f"- [{rel_path}]({chapter_path.as_uri()})")
                else:
                    st.markdown(f"- {rel_path} (missing)")


def _data_source_mode(paths: dict[str, Path | None]) -> str:
    existing_paths = [path for path in paths.values() if path is not None and path.exists()]
    if not existing_paths:
        return "none"

    smoke_hits = sum("/outputs/smoke/" in str(path) for path in existing_paths)
    report_hits = sum("/outputs/reports/" in str(path) for path in existing_paths)

    if smoke_hits and not report_hits:
        return "synthetic_smoke"
    if report_hits and not smoke_hits:
        return "reports_data"
    return "mixed"


def _method_snapshot(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "No method summary available yet."

    if "method" not in frame.columns:
        return "Method summary is present but missing `method` column."

    component_labels = [
        ("sanity", "Sanity"),
        ("perturbation", "Perturbation"),
        ("robustness", "Robustness"),
    ]
    component_bits: list[str] = []
    for column, label in component_labels:
        if column not in frame.columns:
            continue
        ranked = frame[["method", column]].dropna()
        if ranked.empty:
            continue
        best_row = ranked.sort_values(by=column, ascending=False).iloc[0]
        component_bits.append(f"{label}: `{best_row['method']}` ({float(best_row[column]):.3f})")

    nfi_bit = ""
    if "nfi" in frame.columns:
        nfi_ranked = frame[["method", "nfi"]].dropna()
        if not nfi_ranked.empty:
            nfi_best = nfi_ranked.sort_values(by="nfi", ascending=False).iloc[0]
            nfi_bit = f"NFI summary: `{nfi_best['method']}` ({float(nfi_best['nfi']):.3f})"

    if component_bits and nfi_bit:
        return " | ".join(component_bits + [nfi_bit])
    if component_bits:
        return " | ".join(component_bits)
    if nfi_bit:
        return nfi_bit
    return "Method summary is present but missing non-empty component and NFI values."


def _advisor_snapshot_text(
    project_root: Path,
    stage_df: pd.DataFrame,
    data_mode: str,
    e23_summary: pd.DataFrame | None,
    e7_summary: pd.DataFrame | None,
    e8_summary: pd.DataFrame | None,
) -> str:
    done_count = int((stage_df["status"] == "done").sum()) if "status" in stage_df.columns else 0
    total_count = int(len(stage_df))
    mode_label = {
        "none": "No artifacts detected",
        "synthetic_smoke": "Synthetic smoke artifacts only",
        "reports_data": "Reports artifacts (dataset-backed runs)",
        "mixed": "Mixed smoke + reports artifacts",
    }.get(data_mode, data_mode)

    lines: list[str] = [
        "# Radiology XAI Dashboard Snapshot",
        "",
        f"- Project root: `{project_root}`",
        f"- Data mode: {mode_label}",
        f"- Completed stages: {done_count}/{total_count}",
        "",
        "## Stage Status",
    ]

    for _, row in stage_df.iterrows():
        lines.append(
            "- {stage}: {status} ({done}/{total})".format(
                stage=row.get("stage", "Unknown"),
                status=row.get("status", "unknown"),
                done=row.get("completed_artifacts", 0),
                total=row.get("total_artifacts", 0),
            )
        )

        lines.extend(
        [
            "",
            "## Method Snapshots",
            f"- E2/E3: {_method_snapshot(e23_summary)}",
            f"- E7: {_method_snapshot(e7_summary)}",
            f"- E8: {_method_snapshot(e8_summary)}",
            "",
            "## Interpretation Notes",
            "- Component metrics (sanity, perturbation, robustness) are primary evidence; interpret those first.",
            "- NFI is a secondary relative summary score in [0,1]; higher is better within a fixed split/protocol.",
            "- Current thesis core scope is saliency + concept families; text tracks are optional future work.",
            "- Smoke runs validate pipeline wiring, file contracts, and metric computation paths.",
            "- Smoke runs do not establish clinical validity or model generalization.",
        ]
    )

    return "\n".join(lines)


def _show_method_summary(title: str, frame: pd.DataFrame | None) -> None:
    st.subheader(title)
    if frame is None or frame.empty:
        st.info("No data found yet.")
        return

    st.dataframe(frame, use_container_width=True, hide_index=True)

    component_cols = [column for column in ["sanity", "perturbation", "robustness"] if column in frame.columns]
    if "method" in frame.columns and len(component_cols) >= 2:
        st.caption("Primary component metrics")
        component_df = frame[["method"] + component_cols].dropna().set_index("method")
        st.bar_chart(component_df, use_container_width=True)

    if "method" in frame.columns and "nfi" in frame.columns:
        st.caption("Secondary aggregate summary (NFI)")
        chart_df = frame[["method", "nfi"]].dropna().set_index("method")
        st.bar_chart(chart_df, use_container_width=True)


def _show_pairwise(title: str, frame: pd.DataFrame | None) -> None:
    st.subheader(title)
    if frame is None or frame.empty:
        st.info("No pairwise output found yet.")
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _curve_plot_rows(sample_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "method" not in sample_frame.columns:
        return pd.DataFrame()

    for _, row in sample_frame.iterrows():
        method_name = str(row["method"])
        deletion = parse_curve_string(row.get("deletion_curve", ""))
        insertion = parse_curve_string(row.get("insertion_curve", ""))

        for index, value in enumerate(deletion):
            rows.append({"method": method_name, "curve_type": "deletion", "step": index, "score": float(value)})
        for index, value in enumerate(insertion):
            rows.append({"method": method_name, "curve_type": "insertion", "step": index, "score": float(value)})

    return pd.DataFrame(rows)


def _render_sample_inspector(sample_scores: pd.DataFrame | None, source_name: str) -> None:
    st.subheader(f"Sample Inspector ({source_name})")
    if sample_scores is None or sample_scores.empty:
        st.info("No sample-level scores available.")
        return

    id_col = "study_id" if "study_id" in sample_scores.columns else ("sample_id" if "sample_id" in sample_scores.columns else None)
    if id_col is None:
        st.warning("No study/sample id column found in this file.")
        return

    id_values = sorted(sample_scores[id_col].dropna().astype(str).unique().tolist())
    if not id_values:
        st.info("No sample ids found.")
        return

    selected_id = st.selectbox("Select sample ID", options=id_values, index=0)
    selected = sample_scores[sample_scores[id_col].astype(str) == selected_id].copy()

    display_cols = [
        col
        for col in [
            id_col,
            "method",
            "sanity_score",
            "perturbation_score",
            "nuisance_similarity",
            "sample_nfi",
            "base_prediction",
            "perturbed_prediction",
        ]
        if col in selected.columns
    ]
    if display_cols:
        st.dataframe(selected[display_cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(selected, use_container_width=True, hide_index=True)

    curves = _curve_plot_rows(selected)
    if not curves.empty:
        for curve_type in ["deletion", "insertion"]:
            sub = curves[curves["curve_type"] == curve_type]
            pivot = sub.pivot_table(index="step", columns="method", values="score", aggfunc="mean")
            st.caption(f"{curve_type.title()} curves")
            st.line_chart(pivot, use_container_width=True)


def _tail_lines(text: str, max_lines: int = 120) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _run_pipeline_command(
    project_root: Path,
    command: list[str],
    env_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    started = time.time()
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
    )
    duration = time.time() - started

    return {
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "duration_seconds": float(duration),
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
    }


def _render_run_actions(project_root: Path) -> None:
    with st.sidebar.expander("Run Actions", expanded=True):
        st.caption("Trigger thesis scripts directly from the dashboard.")

        fast_smoke_mode = st.checkbox(
            "Fast synthetic mode",
            value=True,
            help="Uses smaller synthetic split sizes for quicker test runs.",
        )

        if st.button("Run Synthetic Smoke", use_container_width=True):
            env: dict[str, str] = {}
            if fast_smoke_mode:
                env.update(
                    {
                        "SMOKE_NUM_TRAIN": "8",
                        "SMOKE_NUM_VAL": "4",
                        "SMOKE_NUM_TEST": "4",
                        "SMOKE_E8_RUNS": "4",
                    }
                )
            with st.spinner("Running synthetic smoke pipeline..."):
                result = _run_pipeline_command(
                    project_root=project_root,
                    command=["bash", "scripts/run_synthetic_smoke.sh"],
                    env_overrides=env,
                )
            st.session_state["last_action_name"] = "Run Synthetic Smoke"
            st.session_state["last_action_result"] = result

        if st.button("Run E8 Template Demo", use_container_width=True):
            env = {"E8_INPUT_CSV": "configs/eval/e8_randomization_input_template.csv"}
            with st.spinner("Running E8 randomization demo..."):
                result = _run_pipeline_command(
                    project_root=project_root,
                    command=["bash", "scripts/run_e8_randomization.sh"],
                    env_overrides=env,
                )
            st.session_state["last_action_name"] = "Run E8 Template Demo"
            st.session_state["last_action_result"] = result

        if st.button("Run E4 Proxy", use_container_width=True):
            with st.spinner("Running E4 concept proxy generator..."):
                result = _run_pipeline_command(
                    project_root=project_root,
                    command=["bash", "scripts/run_e4_concept.sh"],
                )
            st.session_state["last_action_name"] = "Run E4 Proxy"
            st.session_state["last_action_result"] = result

        with st.expander("Future Work Actions (Text Methods)", expanded=False):
            st.caption("Optional actions for text-family prototypes, outside the current core thesis scope.")

            if st.button("Run E5 Proxy (Future)", use_container_width=True):
                with st.spinner("Running E5 constrained-text proxy generator..."):
                    result = _run_pipeline_command(
                        project_root=project_root,
                        command=["bash", "scripts/run_e5_text_constrained.sh"],
                    )
                st.session_state["last_action_name"] = "Run E5 Proxy (Future)"
                st.session_state["last_action_result"] = result

            if st.button("Run E6 Proxy (Future)", use_container_width=True):
                with st.spinner("Running E6 unconstrained-text proxy generator..."):
                    result = _run_pipeline_command(
                        project_root=project_root,
                        command=["bash", "scripts/run_e6_text_unconstrained.sh"],
                    )
                st.session_state["last_action_name"] = "Run E6 Proxy (Future)"
                st.session_state["last_action_result"] = result

        if st.button("Assemble Family Artifacts", use_container_width=True):
            with st.spinner("Assembling available explanation-family artifacts..."):
                result = _run_pipeline_command(
                    project_root=project_root,
                    command=["bash", "scripts/run_assemble_family_artifacts.sh"],
                )
            st.session_state["last_action_name"] = "Assemble Family Artifacts"
            st.session_state["last_action_result"] = result

        if st.button("Run Full Pipeline (After Data)", use_container_width=True):
            with st.spinner("Running full pipeline..."):
                result = _run_pipeline_command(
                    project_root=project_root,
                    command=["bash", "scripts/run_after_data.sh"],
                )
            st.session_state["last_action_name"] = "Run Full Pipeline (After Data)"
            st.session_state["last_action_result"] = result

        if st.button("Clear Last Action", use_container_width=True):
            st.session_state.pop("last_action_name", None)
            st.session_state.pop("last_action_result", None)

        last_name = st.session_state.get("last_action_name")
        last_result = st.session_state.get("last_action_result")
        if last_name and isinstance(last_result, dict):
            success = int(last_result.get("returncode", 1)) == 0
            st.markdown(f"**Last Action:** {last_name}")
            st.write(f"Status: {'✅ Success' if success else '❌ Failed'}")
            st.write(f"Duration: {float(last_result.get('duration_seconds', 0.0)):.1f}s")
            st.code(str(last_result.get("command", "")), language="bash")

            logs = f"{last_result.get('stdout', '')}\n{last_result.get('stderr', '')}".strip()
            if logs:
                show_logs = st.checkbox(
                    "Show command logs",
                    value=(not success),
                    key="show_last_action_logs",
                )
                if show_logs:
                    st.code(_tail_lines(logs), language="bash")


def main() -> None:
    st.title("Radiology XAI Pipeline Dashboard")
    st.caption(
        "Live view of thesis pipeline artifacts (E0 -> E1 -> E2/E3 -> E4 -> E7 -> E8). "
        "Text tracks (E5/E6) are optional future work."
    )

    default_root = Path(__file__).resolve().parents[1]
    project_root_str = st.sidebar.text_input("Project root", value=str(default_root))
    project_root = Path(project_root_str).expanduser().resolve()
    st.sidebar.write(f"Resolved root: `{project_root}`")

    if st.sidebar.button("Refresh"):
        st.rerun()

    if not project_root.exists():
        st.error(f"Project root does not exist: {project_root}")
        return

    _render_run_actions(project_root)

    artifact_paths = collect_artifact_paths(project_root=project_root)
    stage_df = infer_stage_status(artifact_paths)
    data_mode = _data_source_mode(artifact_paths)

    tabs = st.tabs(
        [
            "Pipeline Overview",
            "Data and Config",
            "E1 Models",
            "Faithfulness (E2/E3 + E7)",
            "E8 Randomization",
            "Sample Inspector",
        ]
    )

    with tabs[0]:
        _render_thesis_reference(project_root, "pipeline_overview")

        done_count = int((stage_df["status"] == "done").sum()) if "status" in stage_df.columns else 0
        total_count = int(len(stage_df))
        progress_pct = (100.0 * done_count / total_count) if total_count else 0.0

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Stages Completed", f"{done_count}/{total_count}")
        col_b.metric("Pipeline Progress", f"{progress_pct:.0f}%")
        col_c.metric(
            "Data Mode",
            {
                "none": "No data",
                "synthetic_smoke": "Smoke",
                "reports_data": "Reports",
                "mixed": "Mixed",
            }.get(data_mode, "Unknown"),
        )

        st.caption(
            "E0 and E1 CNN can remain `Not Started` during synthetic smoke runs; "
            "that is expected until dataset-backed scripts are executed."
        )

        stage_map = pd.DataFrame(
            [
                {"stage": "E0 Audit", "purpose": "Data quality and split integrity checks", "expected_output": "cohort/split/label audit tables"},
                {"stage": "E1 Baseline", "purpose": "Reference predictive model", "expected_output": "split metrics + training history"},
                {"stage": "E1 CNN", "purpose": "Image backbone benchmark", "expected_output": "CNN metrics + run metadata"},
                {"stage": "E2/E3", "purpose": "Explanation scoring primitives", "expected_output": "method + sample-level faithfulness"},
                {"stage": "E4", "purpose": "Concept-family artifact generation", "expected_output": "concept-aligned explanation artifacts"},
                {"stage": "E7", "purpose": "Unified cross-family benchmark", "expected_output": "component summaries + secondary NFI/pairwise deltas"},
                {"stage": "E8", "purpose": "Randomization sanity stress test", "expected_output": "pass rates + run variability"},
                {"stage": "Future", "purpose": "Optional text-family tracks (E5/E6)", "expected_output": "constrained vs unconstrained text rationale artifacts"},
            ]
        )
        st.subheader("Pipeline Map")
        st.dataframe(stage_map, use_container_width=True, hide_index=True)

        st.subheader("Stage Status")
        status_frame = stage_df.copy()
        status_frame["status_display"] = status_frame["status"].map(_status_text)
        st.dataframe(
            status_frame[["stage", "status_display", "completed_artifacts", "total_artifacts", "last_updated"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Artifact Presence")
        artifact_rows: list[dict[str, object]] = []
        for key, path in artifact_paths.items():
            artifact_rows.append(
                {
                    "artifact_key": key,
                    "exists": bool(path is not None and path.exists()),
                    "path": str(path) if path is not None else "",
                    "last_updated": format_timestamp(path),
                }
            )
        artifact_frame = pd.DataFrame(artifact_rows).sort_values(by=["exists", "artifact_key"], ascending=[False, True])
        st.dataframe(artifact_frame, use_container_width=True, hide_index=True)

        e23_summary = read_csv_optional(artifact_paths["e23_method_summary"])
        e7_summary = read_csv_optional(artifact_paths["e7_method_summary"])
        e8_summary = read_csv_optional(artifact_paths["e8_method_summary"])
        snapshot_text = _advisor_snapshot_text(
            project_root=project_root,
            stage_df=stage_df,
            data_mode=data_mode,
            e23_summary=e23_summary,
            e7_summary=e7_summary,
            e8_summary=e8_summary,
        )
        st.download_button(
            "Download Advisor Snapshot (.md)",
            data=snapshot_text,
            file_name="dashboard_snapshot.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with tabs[1]:
        _render_thesis_reference(project_root, "data_and_config")

        st.subheader("Repository and Runtime Context")
        st.caption(
            "Use this tab to verify reproducibility context (branch/commit) and dataset audit outputs."
        )
        st.write(f"Branch: `{repo_branch(project_root)}`")
        st.write(f"Commit: `{repo_commit(project_root)}`")

        split_counts = read_csv_optional(artifact_paths["e0_split_counts"])
        prevalence = read_csv_optional(artifact_paths["e0_label_prevalence"])
        concept_coverage = read_csv_optional(artifact_paths["e0_concept_coverage"])

        st.subheader("E0 Split Counts")
        if split_counts is not None:
            st.dataframe(split_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No split-count output yet.")

        st.subheader("E0 Label Prevalence (Preview)")
        if prevalence is not None:
            st.dataframe(prevalence.head(30), use_container_width=True, hide_index=True)
        else:
            st.info("No label-prevalence output yet.")

        st.subheader("Concept Coverage")
        if concept_coverage is not None:
            st.dataframe(concept_coverage, use_container_width=True, hide_index=True)
        else:
            st.info("No concept coverage output yet (optional).")

        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.caption("E1 CNN Run Meta")
            e1_meta = read_json_optional(artifact_paths["e1_cnn_run_meta"])
            if e1_meta is not None:
                st.json(e1_meta)
            else:
                st.info("Missing")
        with meta_cols[1]:
            st.caption("E2/E3 Generation Meta")
            e23_meta = read_json_optional(artifact_paths["e23_generation_meta"])
            if e23_meta is not None:
                st.json(e23_meta)
            else:
                st.info("Missing")
        with meta_cols[2]:
            st.caption("E8 Randomization Meta")
            e8_meta = read_json_optional(artifact_paths["e8_meta"])
            if e8_meta is not None:
                st.json(e8_meta)
            else:
                st.info("Missing")

    with tabs[2]:
        _render_thesis_reference(project_root, "e1_models")

        st.subheader("E1 Track")
        st.caption(
            "E1 establishes predictive reference behavior before comparing explanation faithfulness."
        )
        available_tracks: list[str] = []
        if read_csv_optional(artifact_paths["e1_metrics_summary"]) is not None:
            available_tracks.append("Baseline")
        if read_csv_optional(artifact_paths["e1_cnn_metrics_summary"]) is not None:
            available_tracks.append("CNN")

        if not available_tracks:
            st.info("No E1 outputs found yet.")
        else:
            default_index = available_tracks.index("CNN") if "CNN" in available_tracks else 0
            track = st.radio("Track", options=available_tracks, index=default_index, horizontal=True)

            if track == "Baseline":
                summary = read_csv_optional(artifact_paths["e1_metrics_summary"])
                by_label = read_csv_optional(artifact_paths["e1_metrics_by_label"])
                history = read_csv_optional(artifact_paths["e1_train_history"])
                meta = read_json_optional(artifact_paths["e1_model_card"])
            else:
                summary = read_csv_optional(artifact_paths["e1_cnn_metrics_summary"])
                by_label = read_csv_optional(artifact_paths["e1_cnn_metrics_by_label"])
                history = read_csv_optional(artifact_paths["e1_cnn_train_history"])
                meta = read_json_optional(artifact_paths["e1_cnn_run_meta"])

            st.subheader("Summary by Split")
            if summary is not None:
                st.dataframe(summary, use_container_width=True, hide_index=True)
            else:
                st.info("Summary not found.")

            st.subheader("Per-label Metrics")
            if by_label is not None:
                st.dataframe(by_label, use_container_width=True, hide_index=True)
            else:
                st.info("Per-label metrics not found.")

            st.subheader("Training Curve")
            if history is not None and not history.empty and "epoch" in history.columns:
                y_col = "train_loss" if "train_loss" in history.columns else ("loss" if "loss" in history.columns else None)
                if y_col is not None:
                    chart_df = history[["epoch", y_col]].dropna().set_index("epoch")
                    st.line_chart(chart_df, use_container_width=True)
                else:
                    st.dataframe(history, use_container_width=True, hide_index=True)
            else:
                st.info("Training history not found.")

            if track == "CNN":
                st.subheader("Dataset Usage")
                dataset_counts = read_csv_optional(artifact_paths["e1_cnn_dataset_counts"])
                if dataset_counts is not None:
                    st.dataframe(dataset_counts, use_container_width=True, hide_index=True)
                else:
                    st.info("Dataset-count output not found.")

            st.subheader("Run Metadata")
            if meta is not None:
                st.json(meta)
            else:
                st.info("Run metadata not found.")

    with tabs[3]:
        _render_thesis_reference(project_root, "faithfulness_e23_e7")

        st.caption(
            "Compare saliency and concept methods under common tests: sanity, perturbation "
            "(deletion/insertion), and nuisance robustness. Interpret component metrics first; use "
            "NFI as a secondary summary."
        )
        e4_artifacts = read_csv_optional(artifact_paths.get("e4_artifacts"))
        e5_artifacts = read_csv_optional(artifact_paths.get("e5_artifacts"))
        e6_artifacts = read_csv_optional(artifact_paths.get("e6_artifacts"))

        artifact_rows: list[dict[str, object]] = []
        for label, frame in [
            ("E4 Concept", e4_artifacts),
            ("E5 Text Constrained (Future)", e5_artifacts),
            ("E6 Text Unconstrained (Future)", e6_artifacts),
        ]:
            artifact_rows.append(
                {
                    "family": label,
                    "available": bool(frame is not None and not frame.empty),
                    "num_rows": int(frame.shape[0]) if frame is not None else 0,
                    "num_methods": int(frame["method"].nunique()) if frame is not None and "method" in frame.columns else 0,
                }
            )
        st.subheader("Concept + Optional Text Artifact Availability")
        st.dataframe(pd.DataFrame(artifact_rows), use_container_width=True, hide_index=True)

        e23_summary = read_csv_optional(artifact_paths["e23_method_summary"])
        e23_pairwise = read_csv_optional(artifact_paths["e23_pairwise"])
        e7_summary = read_csv_optional(artifact_paths["e7_method_summary"])
        e7_pairwise = read_csv_optional(artifact_paths["e7_pairwise"])

        cols = st.columns(2)
        with cols[0]:
            _show_method_summary("E2/E3 Saliency Method Summary", e23_summary)
        with cols[1]:
            _show_method_summary("E7 Unified Method Summary", e7_summary)

        cols = st.columns(2)
        with cols[0]:
            _show_pairwise("E2/E3 Pairwise NFI Deltas", e23_pairwise)
        with cols[1]:
            _show_pairwise("E7 Pairwise NFI Deltas", e7_pairwise)

    with tabs[4]:
        _render_thesis_reference(project_root, "e8_randomization")

        st.caption(
            "E8 checks whether explanations degrade under randomization as expected for faithful methods."
        )
        e8_summary = read_csv_optional(artifact_paths["e8_method_summary"])
        e8_run_scores = read_csv_optional(artifact_paths["e8_run_scores"])
        e8_sample_var = read_csv_optional(artifact_paths["e8_sample_variability"])

        st.subheader("E8 Method Summary")
        if e8_summary is not None:
            st.dataframe(e8_summary, use_container_width=True, hide_index=True)
            if "method" in e8_summary.columns and "pass_rate" in e8_summary.columns:
                st.bar_chart(e8_summary[["method", "pass_rate"]].set_index("method"), use_container_width=True)
        else:
            st.info("No E8 summary found.")

        st.subheader("Run-level Sanity Scores")
        if e8_run_scores is not None:
            st.dataframe(e8_run_scores.head(200), use_container_width=True, hide_index=True)
            if {"method", "sanity_score"}.issubset(e8_run_scores.columns):
                numeric_run = pd.to_numeric(
                    e8_run_scores["run_id"].astype(str).str.extract(r"(\d+)$")[0],
                    errors="coerce",
                )
                if numeric_run.notna().any():
                    chart_frame = e8_run_scores.copy()
                    chart_frame["run_num"] = numeric_run
                    line = chart_frame.pivot_table(index="run_num", columns="method", values="sanity_score", aggfunc="mean")
                    st.line_chart(line, use_container_width=True)
        else:
            st.info("No E8 run-scores file found.")

        st.subheader("Sample-level Variability")
        if e8_sample_var is not None:
            if "std_sanity_similarity" in e8_sample_var.columns:
                e8_sample_var = e8_sample_var.sort_values(by="std_sanity_similarity", ascending=False)
            st.dataframe(e8_sample_var.head(100), use_container_width=True, hide_index=True)
        else:
            st.info("No E8 sample-variability file found.")

    with tabs[5]:
        _render_thesis_reference(project_root, "sample_inspector")

        st.caption(
            "Inspect one sample at a time to validate whether method behavior matches the aggregate metrics."
        )
        e7_samples = read_csv_optional(artifact_paths["e7_sample_scores"])
        e23_samples = read_csv_optional(artifact_paths["e23_sample_scores"])
        if e7_samples is not None and not e7_samples.empty:
            _render_sample_inspector(e7_samples, source_name="E7")
        elif e23_samples is not None and not e23_samples.empty:
            _render_sample_inspector(e23_samples, source_name="E2/E3")
        else:
            st.info("No sample-level outputs found yet.")


if __name__ == "__main__":
    main()
