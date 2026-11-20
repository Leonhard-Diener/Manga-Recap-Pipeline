# YouTube upload

Uploading is **opt-in**. Ordinary rendering/pipeline commands remain offline with
respect to YouTube. `youtube_upload.json` defaults to **unlisted**, with subscriber
notifications off. Nothing is uploaded just by installing or running unit tests.

## One-time account setup

1. Install upload dependencies: `python -m pip install google-api-python-client google-auth-oauthlib`.
2. In Google Cloud Console, enable **YouTube Data API v3** for your project.
3. Configure the OAuth consent screen. For an external app in testing mode, add
   your Google account as a test user. Testing-mode refresh tokens may expire;
   Google may require renewed authorization. Use a **Desktop app** OAuth client,
   not an API key or service account.
4. Save the downloaded client JSON as `.secrets/youtube_client_secret.json`.
   The first real upload opens Google's browser authorization; choose the intended
    YouTube channel/account. Uploads request only `youtube.upload`. Later uploads reuse
   `.secrets/youtube_token.json`, including refreshing credentials when possible.
5. Keep `.secrets/` private: it is gitignored, but not encrypted on disk. Do not
   share it or include it in logs/backups. To change accounts, revoke the app's
   access in Google and remove the local token, then authorize again.

Review copyright/source rights and YouTube policies before uploading manga art.
Private uploads are still sent to Google and can be subject to copyright checks.
Account verification may be needed for videos longer than 15 minutes; API quotas,
OAuth project review, and channel restrictions can also prevent uploads.

## Edit upload behavior

Edit `youtube_upload.json` **before** running an upload:

- `privacy_status`: `private` (testing), `unlisted`, or `public`. Public really
  publishes; no extra confirmation is added to an explicitly requested upload.
- `title`: replace `{title}` with the command's title/manga argument. YouTube
  limits titles to 100 characters; use a shorter custom title when necessary.
- `description`, `tags`, `category_id` (default `24`, Entertainment).
- `notify_subscribers` and `made_for_kids`: required booleans. The optional
  `contains_synthetic_media` boolean is omitted by default; set it for an
  individual upload when a disclosure is required.

The default description puts `Name: {title}` below the business email, using
the supplied manga title. The uploader does not post or pin comments.

Use a separate settings JSON with the same keys for individual videos. Changes
apply to **new uploads**, not previously uploaded video metadata/privacy.

## Standalone test/upload

Use the exact completed MP4, including chapter-one videos in `chapter_runs/`:

```powershell
python test_upload.py "path/to/recap.mp4" --title "Sekai - Chapter 1" --dry-run
python test_upload.py "path/to/recap.mp4" --title "Sekai - Chapter 1"
```

The first command validates/prints metadata with **no OAuth or network calls**.
The second really uploads, using the edited settings (unlisted by default).
Add `--settings "my_upload_settings.json"` to select a different settings file.
No real upload has been performed as part of implementing this feature.

## Automatic upload after rendering

```powershell
python main.py "Manga title" --upload
python test_video.py "Manga title" --upload
```

Both accept `--upload-settings "my_upload_settings.json"`. Upload uses the exact
MP4 just rendered, never a guessed newest file. `main.py` still processes the full
selected manga; uploading does **not** add chapter filtering. For the existing
chapter-one render, use the standalone command above, not a full pipeline rerun.
On upload failure, the pipeline fails but the completed local video is retained.
Retry that exact file with `test_upload.py` rather than rendering again.

## Receipts and retry safety

Each video has a sibling `recap.youtube.json` recording the source hash, requested
metadata, returned video ID/status, and watch URL. A successful repeat skips
uploading that file; it does not change the existing video's settings. Progress
uses resumable chunks with up to three SDK retries per chunk.

An interrupted/failed upload leaves an `uploading` receipt. **Check YouTube Studio
first**: the server may have accepted the video even if the response was lost.
Only after confirming there is no upload should you move that receipt aside and
retry. The resumable session itself is not persisted across process restarts.
Do not blindly clear receipts or retry via a copied file/new render: the guard is
local to that exact video path, not global/channel-wide deduplication. Upload
success means YouTube accepted it, not that processing/copyright checks finished.

## Model-free tests

```powershell
python -X utf8 -m unittest discover -s tests -p test_youtube_uploader.py -v
```

These mock Google services and do not authorize, render, or upload.
