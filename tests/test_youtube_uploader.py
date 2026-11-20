"""No network, OAuth, or real uploads; optional Google packages are mocked."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import youtube_uploader as uploader
import test_upload


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / "recap.mp4"
        self.video.write_bytes(b"mock video bytes")
        self.settings = json.loads(uploader.SETTINGS_PATH.read_text(encoding="utf-8"))
        self.path = self.root / "settings.json"
        self.save_settings()

    def save_settings(self):
        self.path.write_text(json.dumps(self.settings), encoding="utf-8")

    def test_defaults_are_unlisted_without_subscriber_notifications(self):
        body, notify = uploader.upload_metadata(self.path, "Example chapter 1")
        self.assertEqual(body["snippet"]["title"], "Example chapter 1 - Manga Recap")
        self.assertEqual(body["status"]["privacyStatus"], "unlisted")
        self.assertNotIn("containsSyntheticMedia", body["status"])
        self.assertIs(notify, False)

    def test_invalid_settings_fail_closed(self):
        cases = [("privacy_status", "privat"), ("notify_subscribers", "false"),
                 ("made_for_kids", 0), ("contains_synthetic_media", None),
                 ("title", ""), ("title", "x" * 101), ("title", "<bad>"),
                 ("description", "é" * 2501), ("tags", "not a list"),
                 ("tags", [""]), ("tags", ["x" * 501]), ("category_id", "invalid")]
        original = dict(self.settings)
        for key, value in cases:
            with self.subTest(key=key, value=str(value)[:20]):
                self.settings = dict(original, **{key: value})
                self.save_settings()
                with self.assertRaises(ValueError):
                    uploader.upload_metadata(self.path, "Example")
        self.settings = dict(original, unknown=True)
        self.save_settings()
        with self.assertRaises(ValueError):
            uploader.upload_metadata(self.path, "Example")

    def test_dry_run_never_authenticates_or_creates_receipt(self):
        with patch.object(uploader, "authenticated_service") as auth, patch("builtins.print"):
            test_upload.main([str(self.video), "--title", "Chapter 1", "--settings", str(self.path), "--dry-run"])
        auth.assert_not_called()
        self.assertFalse(self.video.with_suffix(".youtube.json").exists())

    def test_missing_empty_and_partial_files_are_rejected(self):
        for name, data in (("empty.mp4", b""), ("recap.partial", b"bytes"), ("recap.partial.mp4", b"bytes")):
            path = self.root / name
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                uploader.upload_video(path, "Title", self.path, dry_run=True)
        with self.assertRaises(FileNotFoundError):
            uploader.upload_video(self.root / "missing.mp4", "Title", self.path, dry_run=True)

    def test_oauth_setup_saves_token_and_reuses_valid_credentials(self):
        credentials = MagicMock()
        credentials.valid = True
        credentials.expired = False
        credentials.to_json.return_value = '{"token": "fake-token"}'
        oauth = MagicMock()
        oauth.InstalledAppFlow.from_client_secrets_file.return_value.run_local_server.return_value = credentials
        credential_module = MagicMock()
        credential_module.Credentials.from_authorized_user_file.return_value = credentials
        discovery = MagicMock()
        modules = {"google.auth.transport.requests": MagicMock(),
                   "google.oauth2.credentials": credential_module,
                   "google_auth_oauthlib.flow": oauth, "googleapiclient.discovery": discovery}
        with patch.object(uploader, "SECRETS_DIR", self.root), patch.dict("sys.modules", modules):
            with self.assertRaises(FileNotFoundError):
                uploader.authenticated_service()
            (self.root / "youtube_client_secret.json").write_text("{}")
            uploader.authenticated_service()
            oauth.InstalledAppFlow.from_client_secrets_file.assert_called_once_with(
                str(self.root / "youtube_client_secret.json"), [uploader.SCOPE])
            self.assertTrue((self.root / "youtube_token.json").is_file())
            uploader.authenticated_service()
            oauth.InstalledAppFlow.from_client_secrets_file.assert_called_once()
            discovery.build.assert_called_with("youtube", "v3", credentials=credentials, cache_discovery=False)

    def test_oauth_refresh_failure_is_not_silently_reauthorized(self):
        (self.root / "youtube_token.json").write_text("{}")
        credentials = MagicMock()
        credentials.expired = True
        credentials.refresh_token = "fake-refresh"
        credentials.refresh.side_effect = RuntimeError("refresh rejected")
        credential_module = MagicMock()
        credential_module.Credentials.from_authorized_user_file.return_value = credentials
        modules = {"google.auth.transport.requests": MagicMock(),
                   "google.oauth2.credentials": credential_module,
                   "google_auth_oauthlib.flow": MagicMock(), "googleapiclient.discovery": MagicMock()}
        with patch.object(uploader, "SECRETS_DIR", self.root), patch.dict("sys.modules", modules):
            with self.assertRaisesRegex(RuntimeError, "refresh rejected"):
                uploader.authenticated_service()
        self.assertEqual((self.root / "youtube_token.json").read_text(), "{}")

    def test_upload_receipt_prevents_repeat_and_changed_file_upload(self):
        self.settings["privacy_status"] = "private"
        self.save_settings()
        service = MagicMock()
        request = service.videos.return_value.insert.return_value
        request.next_chunk.side_effect = [(None, None), (None, {"id": "Abcdef123_-", "status": {"privacyStatus": "private"}})]
        media_module = MagicMock()
        with patch.object(uploader, "authenticated_service", return_value=service) as auth, \
             patch.dict("sys.modules", {"googleapiclient.http": media_module}), patch("builtins.print"):
            receipt = uploader.upload_video(self.video, "Title", self.path)
            again = uploader.upload_video(self.video, "Title", self.path)
            auth.assert_called_once()
            self.assertEqual(receipt, again)
            self.assertEqual(receipt["state"], "uploaded")
            service.commentThreads.assert_not_called()
            kwargs = service.videos.return_value.insert.call_args.kwargs
            self.assertFalse(kwargs["notifySubscribers"])
            self.assertEqual(kwargs["body"]["status"]["privacyStatus"], "private")
            media_module.MediaFileUpload.assert_called_once_with(str(self.video.resolve()), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
            request.next_chunk.assert_called_with(num_retries=3)
            self.video.write_bytes(b"changed video")
            with self.assertRaises(RuntimeError):
                uploader.upload_video(self.video, "Title", self.path)

    def test_unlisted_upload_includes_name_in_description_without_comment(self):
        self.settings["privacy_status"] = "unlisted"
        self.save_settings()
        service = MagicMock()
        service.videos.return_value.insert.return_value.next_chunk.return_value = (None, {
            "id": "Abcdef123_-", "snippet": {"channelId": "channel"}, "status": {"privacyStatus": "unlisted"}})
        with patch.object(uploader, "authenticated_service", return_value=service), patch.dict("sys.modules", {"googleapiclient.http": MagicMock()}):
            receipt = uploader.upload_video(self.video, "Series title", self.path)
            uploader.upload_video(self.video, "Series title", self.path)
        self.assertEqual(receipt["requested_metadata"]["snippet"]["description"],
                         "Business Email: empire.for.manga@gmail.com\n\nName: Series title")
        service.commentThreads.assert_not_called()

    def test_uncertain_failure_retains_guard_and_blocks_blind_retry(self):
        service = MagicMock()
        service.videos.return_value.insert.return_value.next_chunk.side_effect = RuntimeError("connection lost")
        with patch.object(uploader, "authenticated_service", return_value=service) as auth, \
             patch.dict("sys.modules", {"googleapiclient.http": MagicMock()}):
            with self.assertRaisesRegex(RuntimeError, "connection lost"):
                uploader.upload_video(self.video, "Title", self.path)
            with self.assertRaisesRegex(RuntimeError, "manual review"):
                uploader.upload_video(self.video, "Title", self.path)
            auth.assert_called_once()
        receipt = json.loads(self.video.with_suffix(".youtube.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["state"], "uploading")

    def test_unknown_response_does_not_mark_uploaded(self):
        service = MagicMock()
        service.videos.return_value.insert.return_value.next_chunk.return_value = (None, {"id": "invalid"})
        with patch.object(uploader, "authenticated_service", return_value=service), \
             patch.dict("sys.modules", {"googleapiclient.http": MagicMock()}):
            with self.assertRaisesRegex(RuntimeError, "valid video ID"):
                uploader.upload_video(self.video, "Title", self.path)
        self.assertEqual(json.loads(self.video.with_suffix(".youtube.json").read_text())["state"], "uploading")
