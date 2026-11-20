"""Small, line-based progress bars that also survive redirected stage logs."""


def show_progress(label: str, completed: int, total: int) -> None:
    filled = min(24, int(24 * completed / total)) if total else 24
    print(f"[progress] {label} [{'#' * filled}{'-' * (24 - filled)}] {completed}/{total}", flush=True)
