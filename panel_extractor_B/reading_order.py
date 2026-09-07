# Determines manga-style reading order from panel bounding boxes.

from config import READING_ORDER_RTL, ROW_OVERLAP_THRESHOLD


def _vertical_overlap_ratio(
    y1_a: int,
    y2_a: int,
    y1_b: int,
    y2_b: int,
) -> float:
    """Calculate vertical overlap relative to the shorter panel."""
    overlap = min(y2_a, y2_b) - max(y1_a, y1_b)

    if overlap <= 0:
        return 0.0

    shorter_height = min(
        y2_a - y1_a,
        y2_b - y1_b,
    )

    return overlap / shorter_height


def _group_into_rows(items: list, box_key) -> list:
    """Group panels into rows based on vertical overlap."""
    rows = []

    for item in sorted(items, key=lambda i: box_key(i)[1]):
        x1, y1, x2, y2 = box_key(item)

        for row in rows:
            if (
                _vertical_overlap_ratio(
                    y1,
                    y2,
                    row["y1"],
                    row["y2"],
                )
                >= ROW_OVERLAP_THRESHOLD
            ):
                row["items"].append(item)
                row["y1"] = min(row["y1"], y1)
                row["y2"] = max(row["y2"], y2)
                break
        else:
            rows.append(
                {
                    "y1": y1,
                    "y2": y2,
                    "items": [item],
                }
            )

    return rows


def sort_reading_order(
    items: list,
    box_key=lambda item: item,
    right_to_left: bool = READING_ORDER_RTL,
) -> list:
    """Sort panels top-to-bottom and right-to-left."""
    rows = sorted(
        _group_into_rows(items, box_key),
        key=lambda row: row["y1"],
    )

    ordered = []

    for row in rows:
        ordered.extend(
            sorted(
                row["items"],
                key=lambda item: box_key(item)[0],
                reverse=right_to_left,
            )
        )

    return ordered