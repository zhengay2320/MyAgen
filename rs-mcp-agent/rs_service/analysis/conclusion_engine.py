from __future__ import annotations

from typing import Any


def build_conclusion(manifest: dict[str, Any], statistics: dict[str, Any], quality_flags: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a rule-based conclusion from statistics and quality flags."""
    task = str(manifest.get("task", "unknown"))
    serious = [flag for flag in quality_flags if flag.get("severity") in {"error", "warning"}]
    confidence = "high"
    if any(flag.get("severity") == "error" for flag in quality_flags):
        confidence = "low"
    elif serious:
        confidence = "medium"
    if _has_registration_risk(manifest, quality_flags):
        confidence = "low"

    key_findings = _key_findings(task, statistics)
    limitations = _limitations(task, manifest, quality_flags)
    return {
        "summary": _summary(task, statistics, quality_flags),
        "confidence": confidence,
        "key_findings": key_findings,
        "limitations": limitations,
        "recommended_review_areas": _review_areas(task, statistics, quality_flags),
        "next_steps": _next_steps(task, quality_flags),
    }


def _summary(task: str, statistics: dict[str, Any], quality_flags: list[dict[str, Any]]) -> str:
    """Create a concise Chinese summary grounded in statistics."""
    if task in {"object_detection", "oriented_detection", "instance_segmentation"}:
        count = statistics.get("target_count", statistics.get("instance_count", 0))
        mean_conf = statistics.get("mean_confidence", 0.0)
        return f"本次任务识别到 {count} 个目标，平均置信度为 {mean_conf:.3f}，质量标记 {len(quality_flags)} 项。"
    if task == "semantic_segmentation":
        total_area = statistics.get("total_area_km2", 0.0)
        dominant = statistics.get("dominant_class")
        return f"本次语义分割覆盖约 {total_area:.6f} km2，主导类别为 {dominant}，质量标记 {len(quality_flags)} 项。"
    if task == "change_detection":
        ratio = statistics.get("change_area_ratio", 0.0)
        area = statistics.get("changed_area_km2", 0.0)
        return f"本次变化检测变化面积约 {area:.6f} km2，占比 {ratio:.3%}，质量标记 {len(quality_flags)} 项。"
    if task == "super_resolution":
        return "本次超分辨率结果已完成 transform 和输出尺寸检查；无参考真值时不评价 PSNR、SSIM 或实际清晰度提升。"
    if task == "spectral_indices":
        return f"本次光谱指数统计均值为 {statistics.get('mean', 0.0):.4f}，需结合地物类型解释。"
    return f"任务 {task} 已完成统计和质量检查。"


def _key_findings(task: str, statistics: dict[str, Any]) -> list[str]:
    """Extract key findings without inventing unavailable metrics."""
    findings: list[str] = []
    if "target_count" in statistics:
        findings.append(f"目标数量：{statistics['target_count']}")
        findings.append(f"低置信度比例：{statistics.get('low_confidence_ratio', 0.0):.3f}")
    if "class_ratios" in statistics:
        findings.append(f"类别占比：{statistics['class_ratios']}")
        findings.append(f"连通域数量：{statistics.get('connected_component_count', 0)}")
    if "change_area_ratio" in statistics:
        findings.append(f"变化面积占比：{statistics['change_area_ratio']:.3%}")
        findings.append(f"最大变化斑块面积：{statistics.get('max_patch_area_m2', 0.0):.2f} m2")
    if statistics.get("type") == "super_resolution":
        findings.append(f"输出尺寸是否匹配 scale：{statistics.get('shape_matches_scale')}")
        findings.append(f"transform 是否匹配 scale：{statistics.get('transform_matches_scale')}")
    if statistics.get("type") == "spectral_index":
        findings.append(f"指数范围：{statistics.get('min')} 至 {statistics.get('max')}")
    return findings or ["无额外关键统计发现。"]


def _limitations(task: str, manifest: dict[str, Any], quality_flags: list[dict[str, Any]]) -> list[str]:
    """List limitations grounded in task type and quality flags."""
    limitations = ["未提供人工真值，因此不输出 IoU、mAP、PSNR 等监督评价指标。"]
    if task == "change_detection":
        limitations.append("变化检测结论依赖两期影像配准质量；若 CRS、分辨率或对齐存在问题，变化面积可能失真。")
    if task == "super_resolution":
        limitations.append("无参考超分只能检查尺寸、transform 和基础统计，不能声称结果更清晰或更准确。")
    for item in quality_flags:
        limitations.append(f"{item.get('code')}: {item.get('message')}")
    return limitations


def _review_areas(task: str, statistics: dict[str, Any], quality_flags: list[dict[str, Any]]) -> list[str]:
    """Recommend human review areas based on warnings and statistics."""
    areas = []
    codes = {flag.get("code") for flag in quality_flags}
    if "low_confidence_warning" in codes:
        areas.append("低置信度目标集中区域")
    if "too_many_small_polygons_warning" in codes:
        areas.append("小斑块密集区域")
    if task == "change_detection":
        areas.append("最大变化斑块及其周边")
    if task == "super_resolution":
        areas.append("道路、建筑边缘等高频纹理区域")
    return areas or ["抽样检查主要输出区域"]


def _next_steps(task: str, quality_flags: list[dict[str, Any]]) -> list[str]:
    """Recommend next steps grounded in quality outcomes."""
    steps = ["核对输出文件路径并抽样人工复核。"]
    if quality_flags:
        steps.append("优先处理 quality_flags 中的 warning/error。")
    if task in {"object_detection", "oriented_detection", "instance_segmentation"}:
        steps.append("如目标漏检或误检较多，调整置信度阈值或更换真实模型 adapter。")
    if task == "change_detection":
        steps.append("复核两期影像配准和时相差异。")
    return steps


def _has_registration_risk(manifest: dict[str, Any], quality_flags: list[dict[str, Any]]) -> bool:
    codes = {flag.get("code") for flag in quality_flags}
    return "georeference_mismatch" in codes or "no_crs_warning" in codes
