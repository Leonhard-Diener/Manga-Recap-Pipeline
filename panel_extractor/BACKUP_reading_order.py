# Determines manga-style reading order from a set of panel bounding boxes.

from config import READING_ORDER_RTL, ROW_OVERLAP_THRESHOLD


def _vertical_overlap_ratio(y1_a, y2_a, y1_b, y2_b) -> float:
    # Overlap between two vertical spans, relative to the shorter panel's height.
    overlap = min(y2_a, y2_b) - max(y1_a, y1_b)
    if overlap <= 0:
        return 0.0
    shorter_height = min(y2_a - y1_a, y2_b - y1_b)
    return overlap / shorter_height


def _group_into_rows(items: list, box_key) -> list:
    # Cluster panels into reading rows based on vertical overlap.
    rows = []
    for item in sorted(items, key=lambda i: box_key(i)[1]):
        x1, y1, x2, y2 = box_key(item)
        for row in rows:
            if _vertical_overlap_ratio(y1, y2, row["y1"], row["y2"]) >= ROW_OVERLAP_THRESHOLD:
                row["items"].append(item)
                row["y1"] = min(row["y1"], y1)
                row["y2"] = max(row["y2"], y2)
                break
        else:
            rows.append({"y1": y1, "y2": y2, "items": [item]})
    return rows


def sort_reading_order(items: list, box_key=lambda item: item, right_to_left: bool = READING_ORDER_RTL) -> list:
    # Order items top-to-bottom by row, then right-to-left (or left-to-right) within each row.
    # box_key extracts (x1, y1, x2, y2) from an item, so this works for plain boxes or richer tuples.
    rows = sorted(_group_into_rows(items, box_key), key=lambda row: row["y1"])

    ordered = []
    for row in rows:
        ordered.extend(sorted(row["items"], key=lambda i: box_key(i)[0], reverse=right_to_left))

    return ordered
