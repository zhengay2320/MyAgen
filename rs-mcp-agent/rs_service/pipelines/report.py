from __future__ import annotations

from pathlib import Path
from typing import Any

from rs_service.core.manifest import read_json, write_manifest
from rs_service.pipelines.base import prepare_output_dir


def _format_flags(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "- No quality flags were raised.\n"
    return "".join(f"- [{item.get('severity', 'info')}] {item.get('code')}: {item.get('message')}\n" for item in flags)


def generate_report(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    title: str = "Remote Sensing Processing Report",
) -> dict[str, Any]:
    out_dir = prepare_output_dir(output_dir, task="report")
    source = read_json(manifest_path)
    report_path = out_dir / "report.md"
    lines = [
        f"# {title}",
        "",
        f"- Source task: `{source.get('task')}`",
        f"- Source job: `{source.get('job_id')}`",
        f"- Source status: `{source.get('status')}`",
        "",
        "## Inputs",
        "",
    ]
    for key, value in source.get("inputs", {}).items():
        if key.endswith("raster") and isinstance(value, dict):
            lines.append(f"- {key}: {value.get('width')} x {value.get('height')}, {value.get('count')} band(s), CRS={value.get('crs')}")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Outputs", ""])
    for key, value in source.get("outputs", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Statistics", ""])
    if source.get("stats"):
        for key, value in source["stats"].items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- No statistics were recorded.")
    lines.extend(["", "## Quality Flags", "", _format_flags(source.get("quality_flags", []))])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return write_manifest(
        task="report",
        output_dir=out_dir,
        inputs={"source_manifest": str(manifest_path)},
        outputs={"report_md": str(report_path)},
        parameters={"title": title},
        stats={"source_task": source.get("task"), "source_job_id": source.get("job_id")},
        quality_flags=source.get("quality_flags", []),
        model={"id": "markdown-report-generator", "backend": "python", "framework": "markdown"},
    )
