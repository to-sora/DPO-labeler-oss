from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .common import DEFECT_TAGS, DECISIONS


@dataclass(frozen=True)
class FilterField:
    field: str
    label: str
    operators: tuple[str, ...]
    value_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "label": self.label,
            "operators": list(self.operators),
            "value_kind": self.value_kind,
        }


FILTER_FIELDS = (
    FilterField("dataset_id", "Dataset", ("eq", "in", "contains"), "string"),
    FilterField("task_key", "Task", ("eq", "in"), "task"),
    FilterField("task_name", "Task Name", ("eq", "in", "contains"), "string"),
    FilterField("task_yaml_name", "Task YAML", ("eq", "in", "contains"), "string"),
    FilterField("reviewer_username", "Reviewer", ("eq", "in", "contains"), "string"),
    FilterField("decision", "Decision", ("eq", "in"), "enum"),
    FilterField("is_labeled", "Is Labeled", ("eq",), "boolean"),
    FilterField("label_created_at", "Label Time", ("before", "after", "between"), "datetime"),
    FilterField("note", "Note", ("contains", "eq"), "string"),
    FilterField("image_a_defects", "Image A Defects", ("any_of", "all_of", "none_of"), "defects"),
    FilterField("image_b_defects", "Image B Defects", ("any_of", "all_of", "none_of"), "defects"),
    FilterField("either_image_defects", "Either Image Defects", ("any_of", "all_of", "none_of"), "defects"),
    FilterField("both_images_defects", "Both Images Defects", ("any_of", "all_of", "none_of"), "defects"),
)
FILTER_FIELDS_BY_NAME = {field.field: field for field in FILTER_FIELDS}


class FilterValidationError(ValueError):
    pass


class FilterEngine:
    def metadata(self) -> dict[str, Any]:
        return {
            "fields": [field.to_dict() for field in FILTER_FIELDS],
            "decisions": list(DECISIONS),
            "defect_tags": list(DEFECT_TAGS),
        }

    def validate(self, ast: Any) -> dict[str, Any]:
        if ast in (None, {}):
            return {"type": "group", "operator": "and", "conditions": []}
        validated = self._validate_node(ast)
        if validated.get("type") != "group":
            raise FilterValidationError("Root filter node must be a group")
        return validated

    def matches(self, ast: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        return self._matches_node(ast, context)

    def _validate_node(self, node: Any) -> dict[str, Any]:
        if not isinstance(node, Mapping):
            raise FilterValidationError("Filter node must be an object")
        node_type = node.get("type")
        if node_type == "group":
            operator = str(node.get("operator", "and")).lower()
            if operator not in {"and", "or"}:
                raise FilterValidationError("Group operator must be 'and' or 'or'")
            raw_conditions = node.get("conditions", [])
            if not isinstance(raw_conditions, list):
                raise FilterValidationError("Group conditions must be a list")
            return {
                "type": "group",
                "operator": operator,
                "conditions": [self._validate_node(condition) for condition in raw_conditions],
            }

        if node_type == "rule":
            field_name = str(node.get("field", "")).strip()
            operator = str(node.get("operator", "")).strip().lower()
            if field_name not in FILTER_FIELDS_BY_NAME:
                raise FilterValidationError(f"Unsupported filter field: {field_name}")
            field = FILTER_FIELDS_BY_NAME[field_name]
            if operator not in field.operators:
                raise FilterValidationError(f"Operator {operator!r} is not allowed for field {field_name!r}")
            value = self._validate_rule_value(field, operator, node.get("value"))
            return {
                "type": "rule",
                "field": field_name,
                "operator": operator,
                "value": value,
            }

        raise FilterValidationError("Filter node type must be 'group' or 'rule'")

    def _validate_rule_value(self, field: FilterField, operator: str, value: Any) -> Any:
        if field.value_kind in {"string", "task", "enum"} and operator in {"eq", "contains"}:
            return str(value or "").strip()
        if field.value_kind in {"string", "task", "enum"} and operator == "in":
            if not isinstance(value, list) or not value:
                raise FilterValidationError("Operator 'in' requires a non-empty list")
            normalized = [str(item).strip() for item in value if str(item).strip()]
            if not normalized:
                raise FilterValidationError("Operator 'in' requires a non-empty list")
            return normalized
        if field.value_kind == "boolean":
            return bool(value)
        if field.value_kind == "datetime":
            if operator in {"before", "after"}:
                return self._parse_datetime(value).isoformat()
            if not isinstance(value, Mapping):
                raise FilterValidationError("Operator 'between' requires an object with start and end")
            start = self._parse_datetime(value.get("start")).isoformat()
            end = self._parse_datetime(value.get("end")).isoformat()
            if self._parse_datetime(start) > self._parse_datetime(end):
                raise FilterValidationError("Datetime 'between' start must be earlier than or equal to end")
            return {"start": start, "end": end}
        if field.value_kind == "defects":
            if not isinstance(value, list) or not value:
                raise FilterValidationError("Defect filter value must be a non-empty list")
            normalized = []
            for item in value:
                defect = str(item).strip()
                if defect not in DEFECT_TAGS:
                    raise FilterValidationError(f"Unknown defect tag: {defect}")
                if defect not in normalized:
                    normalized.append(defect)
            return normalized
        raise FilterValidationError(f"Unsupported value kind for field {field.field}")

    def _matches_node(self, node: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        node_type = node["type"]
        if node_type == "group":
            conditions = node["conditions"]
            if not conditions:
                return True
            results = [self._matches_node(condition, context) for condition in conditions]
            return all(results) if node["operator"] == "and" else any(results)

        field_name = node["field"]
        operator = node["operator"]
        value = node["value"]
        if field_name == "image_a_defects":
            return self._match_defect_operator(operator, context.get("defects_a", []), value)
        if field_name == "image_b_defects":
            return self._match_defect_operator(operator, context.get("defects_b", []), value)
        if field_name == "either_image_defects":
            return self._match_defect_operator(operator, context.get("defects_a", []), value) or self._match_defect_operator(operator, context.get("defects_b", []), value)
        if field_name == "both_images_defects":
            return self._match_defect_operator(operator, context.get("defects_a", []), value) and self._match_defect_operator(operator, context.get("defects_b", []), value)

        candidate = context.get(field_name)
        if operator == "eq":
            return candidate == value
        if operator == "contains":
            return str(value).lower() in str(candidate or "").lower()
        if operator == "in":
            return candidate in value
        if operator == "before":
            return self._context_datetime(candidate) < self._parse_datetime(value)
        if operator == "after":
            return self._context_datetime(candidate) > self._parse_datetime(value)
        if operator == "between":
            candidate_dt = self._context_datetime(candidate)
            return self._parse_datetime(value["start"]) <= candidate_dt <= self._parse_datetime(value["end"])
        raise FilterValidationError(f"Unsupported operator: {operator}")

    @staticmethod
    def _match_defect_operator(operator: str, current: Sequence[str], selected: Sequence[str]) -> bool:
        current_set = set(current)
        selected_set = set(selected)
        if operator == "any_of":
            return bool(current_set & selected_set)
        if operator == "all_of":
            return selected_set.issubset(current_set)
        if operator == "none_of":
            return current_set.isdisjoint(selected_set)
        raise FilterValidationError(f"Unsupported defect operator: {operator}")

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        text = str(value or "").strip()
        if not text:
            raise FilterValidationError("Datetime filter value is required")
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _context_datetime(self, value: Any) -> datetime:
        return self._parse_datetime(value)
