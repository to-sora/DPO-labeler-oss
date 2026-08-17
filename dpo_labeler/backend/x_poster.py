from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class XPosterError(ValueError):
    pass


@dataclass(frozen=True)
class XCredentials:
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str


class XPoster:
    UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"
    STATUS_UPDATE_URL = "https://api.x.com/1.1/statuses/update.json"

    def __init__(
        self,
        consumer_key: str | None,
        consumer_secret: str | None,
        access_token: str | None,
        access_token_secret: str | None,
    ) -> None:
        self.credentials = self._build_credentials(
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret,
        )

    @property
    def is_configured(self) -> bool:
        return self.credentials is not None

    def post_image_tweet(self, image_bytes: bytes, media_type: str, text: str) -> dict[str, Any]:
        if not self.credentials:
            raise XPosterError("X posting is not configured on this server.")
        media_id = self._upload_image(image_bytes, media_type)
        return self._create_status(text, media_id)

    @staticmethod
    def _build_credentials(
        consumer_key: str | None,
        consumer_secret: str | None,
        access_token: str | None,
        access_token_secret: str | None,
    ) -> XCredentials | None:
        values = [consumer_key, consumer_secret, access_token, access_token_secret]
        normalized = [str(value or "").strip() for value in values]
        if not any(normalized):
            return None
        if not all(normalized):
            raise XPosterError("Incomplete X posting credentials; provide consumer key/secret and access token/secret.")
        return XCredentials(
            consumer_key=normalized[0],
            consumer_secret=normalized[1],
            access_token=normalized[2],
            access_token_secret=normalized[3],
        )

    def _upload_image(self, image_bytes: bytes, media_type: str) -> str:
        boundary = f"----dpo-labeler-{secrets.token_hex(12)}"
        body = self._multipart_body(
            boundary,
            "media",
            "image.jpg",
            media_type or "image/jpeg",
            image_bytes,
        )
        request = self._signed_request(
            "POST",
            self.UPLOAD_URL,
            params={"media_category": "tweet_image"},
            body_params=None,
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        payload = self._read_json(request)
        media_id = str(payload.get("media_id_string") or payload.get("media_id") or "").strip()
        if not media_id:
            raise XPosterError("X media upload did not return a media_id.")
        return media_id

    def _create_status(self, text: str, media_id: str) -> dict[str, Any]:
        body_params = {
            "status": text,
            "media_ids": media_id,
        }
        encoded = urlencode(body_params).encode("utf-8")
        request = self._signed_request(
            "POST",
            self.STATUS_UPDATE_URL,
            params=None,
            body_params=body_params,
            body=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        payload = self._read_json(request)
        tweet_id = str(payload.get("id_str") or payload.get("id") or "").strip()
        user = payload.get("user") if isinstance(payload.get("user"), Mapping) else {}
        screen_name = str(user.get("screen_name", "")).strip()
        result = {
            "tweet_id": tweet_id,
            "screen_name": screen_name,
            "raw": payload,
        }
        if tweet_id and screen_name:
            result["tweet_url"] = f"https://x.com/{screen_name}/status/{tweet_id}"
        return result

    def _signed_request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        body_params: Mapping[str, Any] | None,
        body: bytes | None,
        headers: Mapping[str, str] | None = None,
    ) -> Request:
        assert self.credentials is not None
        oauth_params = {
            "oauth_consumer_key": self.credentials.consumer_key,
            "oauth_nonce": secrets.token_hex(16),
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": str(int(time.time())),
            "oauth_token": self.credentials.access_token,
            "oauth_version": "1.0",
        }
        signature_params: list[tuple[str, str]] = []
        for source in (params or {}, body_params or {}, oauth_params):
            for key, value in source.items():
                signature_params.append((self._percent_encode(str(key)), self._percent_encode(str(value))))
        signature_params.sort()
        normalized_params = "&".join(f"{key}={value}" for key, value in signature_params)
        signature_base = "&".join(
            [
                method.upper(),
                self._percent_encode(url),
                self._percent_encode(normalized_params),
            ]
        )
        signing_key = "&".join(
            [
                self._percent_encode(self.credentials.consumer_secret),
                self._percent_encode(self.credentials.access_token_secret),
            ]
        )
        signature = base64.b64encode(
            hmac.new(signing_key.encode("utf-8"), signature_base.encode("utf-8"), hashlib.sha1).digest()
        ).decode("ascii")
        oauth_params["oauth_signature"] = signature
        authorization = "OAuth " + ", ".join(
            f'{self._percent_encode(key)}="{self._percent_encode(value)}"'
            for key, value in sorted(oauth_params.items())
        )
        request_url = url
        if params:
            request_url = f"{request_url}?{urlencode({key: str(value) for key, value in params.items()})}"
        request = Request(request_url, data=body, method=method.upper())
        request.add_header("Authorization", authorization)
        request.add_header("Accept", "application/json")
        for key, value in (headers or {}).items():
            request.add_header(key, value)
        return request

    @staticmethod
    def _read_json(request: Request) -> dict[str, Any]:
        try:
            with urlopen(request) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise XPosterError(f"X request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise XPosterError("X response was not a JSON object.")
        if payload.get("errors"):
            raise XPosterError(f"X API error: {payload['errors']}")
        return payload

    @staticmethod
    def _multipart_body(boundary: str, field_name: str, filename: str, media_type: str, payload: bytes) -> bytes:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode("utf-8")
        footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        return header + payload + footer

    @staticmethod
    def _percent_encode(value: str) -> str:
        return quote(value, safe="~")
