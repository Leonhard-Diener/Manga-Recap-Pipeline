"""Explicit, resumable YouTube uploads; local receipts prevent blind duplicate retries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import config
from common.json_io import write_json

SETTINGS_PATH = config.BASE_DIR / "youtube_upload.json"
SECRETS_DIR = config.BASE_DIR / ".secrets"
SCOPE = "https://www.googleapis.com/auth/youtube.upload"


def upload_metadata(settings_path: Path, title: str) -> tuple[dict, bool]:
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    expected = {"title", "description", "tags", "category_id", "privacy_status",
                "notify_subscribers", "made_for_kids"}
    if not isinstance(settings, dict) or set(settings) - {"contains_synthetic_media"} != expected:
        raise ValueError("Upload settings must contain exactly the keys in youtube_upload.json.")
    for key in ("title", "description", "category_id", "privacy_status"):
        if not isinstance(settings[key], str):
            raise ValueError(f"{key} must be a string.")
    for key in ("notify_subscribers", "made_for_kids", "contains_synthetic_media"):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"{key} must be true or false.")
    rendered_title = settings["title"].replace("{title}", title)
    if not rendered_title.strip() or len(rendered_title) > 100 or any(c in rendered_title for c in "<>"):
        raise ValueError("YouTube title must be 1–100 characters without < or >.")
    description = settings["description"].replace("{title}", title)
    if len(description.encode("utf-8")) > 5000 or any(c in description for c in "<>"):
        raise ValueError("Description must be at most 5000 UTF-8 bytes without < or >.")
    tags = settings["tags"]
    if (not isinstance(tags, list) or any(not isinstance(t, str) or not t.strip() for t in tags)
            or sum(len(t) + (2 if " " in t else 0) for t in tags) + max(0, len(tags) - 1) > 500):
        raise ValueError("Tags must be nonempty strings within YouTube's 500-character total limit.")
    if settings["privacy_status"] not in {"private", "unlisted", "public"}:
        raise ValueError("privacy_status must be private, unlisted, or public.")
    if not re.fullmatch(r"[0-9]+", settings["category_id"]):
        raise ValueError("category_id must be a numeric YouTube category ID.")
    return {
        "snippet": {"title": rendered_title, "description": description,
                    "tags": tags, "categoryId": settings["category_id"]},
        "status": {"privacyStatus": settings["privacy_status"],
                   "selfDeclaredMadeForKids": settings["made_for_kids"],
                   **({"containsSyntheticMedia": settings["contains_synthetic_media"]}
                      if "contains_synthetic_media" in settings else {})},
    }, settings["notify_subscribers"]


def authenticated_service():
    # Imports stay optional for model-free tests and pipelines without --upload.
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    scopes = [SCOPE]
    token_path = SECRETS_DIR / "youtube_token.json"
    client_path = SECRETS_DIR / "youtube_client_secret.json"
    credentials = Credentials.from_authorized_user_file(str(token_path), scopes) if token_path.exists() else None
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials or not credentials.valid:
        if not client_path.is_file():
            raise FileNotFoundError(f"Add a Google Desktop OAuth client JSON at {client_path}; see YOUTUBE_UPLOAD.md.")
        credentials = InstalledAppFlow.from_client_secrets_file(str(client_path), scopes).run_local_server(port=0)
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(token_path, json.loads(credentials.to_json()))
    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def upload_video(video_path: Path, title: str, settings_path: Path = SETTINGS_PATH, *, dry_run=False):
    video_path = video_path.resolve(strict=True)
    if (not video_path.is_file() or video_path.suffix.lower() != ".mp4"
            or video_path.name.lower().endswith(".partial.mp4") or video_path.stat().st_size == 0):
        raise ValueError("Select a completed, nonempty MP4 file.")
    body, notify = upload_metadata(settings_path, title)
    if dry_run:
        print(json.dumps({"file": str(video_path), "body": body, "notifySubscribers": notify}, indent=2))
        return None
    digest = hashlib.sha256()
    with video_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    receipt_path = video_path.with_suffix(".youtube.json")
    fingerprint = digest.hexdigest()
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous.get("sha256") == fingerprint and previous.get("state") == "uploaded":
            print(f"Already uploaded: {previous['url']}", flush=True)
            return previous
        raise RuntimeError(f"Upload receipt needs manual review: {receipt_path}. Check YouTube Studio before retrying to avoid duplicates.")
    service = authenticated_service()
    from googleapiclient.http import MediaFileUpload

    receipt = {"state": "uploading", "file": str(video_path), "sha256": fingerprint,
               "requested_metadata": body, "notify_subscribers": notify}
    # Exclusive creation blocks concurrent sends; uncertain failures deliberately keep this guard.
    with receipt_path.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2)
    request = service.videos().insert(
        part="snippet,status", body=body, notifySubscribers=notify,
        media_body=MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True),
    )
    response = None
    while response is None:
        status, response = request.next_chunk(num_retries=3)
        if status:
            print(f"[progress] YouTube upload {status.progress():.0%}", flush=True)
    video_id = response.get("id")
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise RuntimeError("YouTube did not return a valid video ID. Check Studio before retrying.")
    receipt.update(state="uploaded", video_id=video_id, url=f"https://www.youtube.com/watch?v={video_id}",
                   returned_status=response.get("status"))
    write_json(receipt_path, receipt)
    print(f"Uploaded: {receipt['url']}", flush=True)
    return receipt
