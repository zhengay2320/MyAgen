from __future__ import annotations

from pathlib import Path
from typing import Any


def build_report_markdown(manifest: dict[str, Any], output_path: str | Path) -> str:
    """Build a Chinese Markdown report from manifest, statistics, quality flags, and conclusion."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conclusion = manifest.get("conclusion", {})
    if isinstance(conclusion, str):
        conclusion = {"summary": conclusion, "confidence": "medium", "key_findings": [], "limitations": [], "recommended_review_areas": [], "next_steps": []}
    lines = [
        "# 遥感处理分析报告",
        "",
        "## 数据概况",
        f"- Job ID: `{manifest.get('job_id')}`",
        f"- Task: `{manifest.get('task')}`",
        f"- Status: `{manifest.get('status')}`",
        f"- Input files: `{manifest.get('input_files', [])}`",
        "",
        "## 使用模型",
        f"- Model ID: `{manifest.get('model_id')}`",
        f"- Model metadata: `{manifest.get('model', {})}`",
        "",
        "## 参数设置",
        f"`{manifest.get('parameters', {})}`",
        "",
        "## 推理结果",
        _format_outputs(manifest.get("outputs", {})),
        "",
        "## 统计分析",
        _format_dict(manifest.get("statistics", {})),
        "",
        "## 质量检查",
        _format_flags(manifest.get("quality_flags", [])),
        "",
        "## 风险与局限",
        _format_list(conclusion.get("limitations", [])),
        "",
        "## 结论",
        f"- Summary: {conclusion.get('summary', '')}",
        f"- Confidence: `{conclusion.get('confidence', 'medium')}`",
        "- Key findings:",
        _format_list(conclusion.get("key_findings", [])),
        "",
        "## 建议人工复核区域",
        _format_list(conclusion.get("recommended_review_areas", [])),
        "",
        "## 下一步建议",
        _format_list(conclusion.get("next_steps", [])),
        "",
        "## 输出文件清单",
        _format_outputs(manifest.get("outputs", {})),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _format_outputs(outputs: dict[str, Any]) -> str:
    if not outputs:
        return "- 无输出文件记录。"
    return "\n".join(f"- {key}: `{value}`" for key, value in outputs.items())


def _format_flags(flags: list[dict[str, Any]]) -> str:
    if not flags:
        return "- 未发现质量检查标记。"
    return "\n".join(f"- [{item.get('severity', 'warning')}] {item.get('code')}: {item.get('message')}" for item in flags)


def _format_list(items: list[Any]) -> str:
    if not items:
        return "- 无。"
    return "\n".join(f"- {item}" for item in items)


def _format_dict(payload: dict[str, Any]) -> str:
    if not payload:
        return "- 无统计结果。"
    return "\n".join(f"- {key}: `{value}`" for key, value in payload.items())
