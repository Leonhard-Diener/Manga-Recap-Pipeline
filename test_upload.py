"""Upload one explicitly selected MP4; use --dry-run to preview without network/auth."""
import argparse
from pathlib import Path

from youtube_uploader import SETTINGS_PATH, upload_video


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Exact completed MP4 path (no automatic latest-file selection).")
    parser.add_argument("--title", required=True, help="Replaces {title} in upload settings; include chapter number if needed.")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print metadata; do not authorize or upload.")
    args = parser.parse_args(argv)
    upload_video(args.video, args.title, args.settings, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
