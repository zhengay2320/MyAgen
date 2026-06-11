from __future__ import annotations

from typing import Any


def run_quality_checks(manifest: dict[str, Any], statistics: dict[str, Any]) -> list[dict[str, Any]]:
    """Run rule-based quality checks from manifest and statistics."""
    flags: list[dict[str, Any]] = []
    flags.extend(manifest.get("quality_flags", []))
    _no_crs_warning(manifest, flags)
    _empty_result_warning(statistics, flags)
    _too_many_small_polygons_warning(statistics, flags)
    _single_class_dominance_warning(statistics, flags)
    _low_confidence_warning(statistics, flags)
    _edge_object_ratio_warning(statistics, flags)
    _change_area_warnings(statistics, flags)
    _sr_transform_inconsistent_warning(statistics, flags)
    _tile_seam_risk_warning(manifest, flags)
    return _deduplicate(flags)


def flag(code: str, message: str, severity: str = "warning", **extra: Any) -> dict[str, Any]:
    """Build a quality flag."""
    payload = {"code": code, "message": message, "severity": severity}
    payload.update(extra)
    return payload


def _no_crs_warning(manifest: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    for value in manifest.get("inputs", {}).values():
        if isinstance(value, dict) and "crs" in value and not value.get("crs"):
            flags.append(flag("no_crs_warning", "输入或输出影像缺少 CRS，面积和空间叠加结论需人工复核。"))


def _empty_result_warning(statistics: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    if statistics.get("target_count") == 0 or statistics.get("instance_count") == 0:
        flags.append(flag("empty_result_warning", "结果为空，可能是阈值过高、模型不适配或输入质量不足。"))
    if statistics.get("changed_pixels") == 0:
        flags.append(flag("empty_result_warning", "变化检测未识别到变化像元。"))


def _too_many_small_polygons_warning(statistics: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    components = int(statistics.get("connected_component_count", 0) or 0)
    small = int(statistics.get("small_patch_count", 0) or 0)
    if components and small / components > 0.5:
        flags.append(flag("too_many_small_polygons_warning", "小斑块占比较高，可能存在噪声或切片拼接碎斑。", ratio=small / components))


def _single_class_dominance_warning(statistics: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    ratios = statistics.get("class_ratios") or {}
    if ratios and max(float(value) for value in ratios.values()) > 0.95:
        flags.append(flag("single_class_dominance_warning", "单一类别占比超过 95%，需检查类别映射、阈值或输入影像。"))


def _low_confidence_warning(statistics: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    ratio = float(statistics.get("low_confidence_ratio", 0.0) or 0.0)
    if ratio > 0.3:
        flags.append(flag("low_confidence_warning", "低置信度目标比例较高，建议人工复核。", ratio=ratio))


def _edge_object_ratio_warning(statistics: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    ratio = float(statistics.get("edge_object_ratio", 0.0) or 0.0)
    if ratio > 0.25:
        flags.append(flag("edge_object_ratio_warning", "边缘目标比例较高，存在切片截断或重复检测风险。", ratio=ratio))


def _change_area_warnings(statistics: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    ratio = statistics.get("change_area_ratio")
    if ratio is None:
        return
    ratio_value = float(ratio)
    if ratio_value > 0.75:
        flags.append(flag("change_area_too_high_warning", "变化面积比例过高，需重点检查配准、时相一致性和阈值。", ratio=ratio_value))
    if 0 <= ratio_value < 0.001:
        flags.append(flag("change_area_too_low_warning", "变化面积比例极低，可能无显著变化，也可能阈值过严。", ratio=ratio_value))


def _sr_transform_inconsistent_warning(statistics: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    if statistics.get("type") == "super_resolution" and not statistics.get("transform_matches_scale", True):
        flags.append(flag("sr_transform_inconsistent_warning", "超分输出 transform 与 scale 不一致，空间定位需复核。"))


def _tile_seam_risk_warning(manifest: dict[str, Any], flags: list[dict[str, Any]]) -> None:
    params = manifest.get("parameters", {})
    tile_size = params.get("tile_size")
    overlap = params.get("overlap")
    if tile_size and overlap is not None and int(overlap) < max(16, int(tile_size) * 0.05):
        flags.append(flag("tile_seam_risk_warning", "overlap 较小，存在切片边界拼接痕迹风险。"))


def _deduplicate(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for item in flags:
        key = (item.get("code"), item.get("message"))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
