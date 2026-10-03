from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any


def build_native_generation_schema(
    schema: dict[str, Any], context: dict[str, Any], preferences: dict[str, Any]
) -> dict[str, Any]:
    """Encode backend ranges and catalog choices without choosing a routine."""
    definitions = schema.get("$defs", {})

    def inline(value: Any) -> Any:
        if isinstance(value, list):
            return [inline(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            return inline(definitions[value["$ref"].split("/")[-1]])
        result = {
            key: inline(item)
            for key, item in value.items()
            if key not in {"$defs", "title", "default"}
        }
        # Large string bounds fail in the Polo grammar; local validation retains them.
        if "maxLength" in result:
            result["maxLength"] = min(result["maxLength"], 120)
        if "prefixItems" in result:
            result["items"] = result.pop("prefixItems")
            result["additionalItems"] = False
        return result

    native: dict[str, Any] = inline(schema)
    catalog = preferences["allowed_catalog"]
    requirements = preferences.get("muscle_counts_per_day", [])
    focus = {requirement["muscle"] for requirement in requirements}
    extras = [
        index
        for index, exercise in enumerate(catalog)
        if not focus.intersection(exercise.get("primary_muscles", []))
    ]
    required_properties = {}
    for requirement in requirements:
        muscle, count = requirement["muscle"], requirement["count"]
        choices = [
            index
            for index, exercise in enumerate(catalog)
            if muscle in exercise.get("primary_muscles", [])
        ][:8]
        if len(choices) < count:
            raise ValueError("The required muscle has insufficient catalog choices")
        required_properties[muscle] = {
            "enum": [
                [{"exercise": index} for index in selection]
                for selection in combinations(choices, count)
            ]
        }
    day = native["properties"]["days"]["items"]
    day["properties"]["required_exercises"] = {
        "type": "object",
        "properties": required_properties,
        "required": list(required_properties),
        "additionalProperties": False,
    }
    if "required_exercises" not in day["required"]:
        day["required"].append("required_exercises")
    if "exercises" not in day["required"]:
        day["required"].append("exercises")
    day["properties"]["exercises"]["items"]["properties"]["exercise"] = {"enum": extras}
    rules = preferences.get("prescription_constraints", {}).get("purposes", {})
    if not rules:
        return native
    required_count = sum(requirement["count"] for requirement in requirements)
    available_days = context.get("available_days_per_week", 7)
    variants = []
    for purpose, rule in rules.items():
        minimum_days, maximum_days = rule["weekly_frequency"]
        maximum_days = min(maximum_days, available_days)
        minimum_extras = max(0, rule["exercises_per_day"][0] - required_count)
        maximum_extras = min(rule["exercises_per_day"][1] - required_count, len(extras))
        if minimum_days > maximum_days or maximum_extras < minimum_extras:
            continue
        if minimum_extras > 0 and not extras:
            continue
        variant = deepcopy(native)
        properties = variant["properties"]
        properties["routine_type"] = {"const": purpose}
        properties["days"].update({"minItems": minimum_days, "maxItems": maximum_days})
        work_group = deepcopy(properties["prescription"]["items"])
        fields = work_group["properties"]
        fields["n"].update(
            dict(zip(("minimum", "maximum"), rule["work_sets_per_exercise"], strict=True))
        )
        lower, upper = rule["repetitions"]
        fields["reps"] = {
            "enum": [
                [low, high] for low in range(lower, upper + 1) for high in range(low, upper + 1)
            ]
        }
        fields["rest"].update(dict(zip(("minimum", "maximum"), rule["rest_seconds"], strict=True)))
        fields["warmup"] = {"const": False}
        properties["prescription"].update({"items": work_group, "minItems": 1, "maxItems": 1})
        extra_array = properties["days"]["items"]["properties"]["exercises"]
        if maximum_extras == 0:
            extra_array.clear()
            extra_array["enum"] = [[]]
        else:
            extra_array.update({"minItems": minimum_extras, "maxItems": maximum_extras})
            extra = extra_array["items"]
            warmup = deepcopy(native["properties"]["prescription"]["items"])
            warmup["properties"]["warmup"] = {"const": True}
            extra["properties"]["groups"] = {
                "anyOf": [
                    {"type": "null"},
                    {"type": "array", "items": [work_group], "additionalItems": False},
                    {"type": "array", "items": [warmup, work_group], "additionalItems": False},
                ]
            }
        variants.append(variant)
    if not variants:
        raise ValueError("No prescription fits the requested counts and available days")
    return {"anyOf": variants}
