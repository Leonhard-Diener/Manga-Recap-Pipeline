"""Format-only checks; model prose is never rewritten or judged by keywords.

The historical helper names remain for downstream callers. These checks do not
claim to verify visual accuracy, and repeated/short/uncertain prose is retained.
"""


def text_issues(text: str) -> list[str]:
    return [] if isinstance(text, str) and text.strip() else ["empty_text"]


def action_issues(text: str, *, confidence=None, actor=None, target=None) -> list[str]:
    issues = text_issues(text)
    if actor is not None and target is not None and actor == target:
        issues.append("self_target")
    return issues


def action_is_visual(text: str) -> bool:
    """Compatibility helper: checks format, not whether an action is visible."""
    return not action_issues(text)


def character_description_is_useful(text: str) -> bool:
    """Compatibility helper: any nonempty model string is retained."""
    return not text_issues(text)


def annotate_panel_quality(panels: list[dict]) -> tuple[dict[int, list[str]], list[dict]]:
    issues, rejected = {}, []
    for panel in panels:
        reasons = text_issues(panel.get("description"))
        if reasons:
            number = panel["panel_number"]
            issues[number] = reasons
            rejected.append({
                "kind": "panel_description", "panel_number": number,
                "reasons": reasons, "value": panel.get("description"),
            })
    return issues, rejected
