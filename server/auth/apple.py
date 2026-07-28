"""
Sign in with Apple: identity-token verification, code exchange, and revocation.

Configuration (all required except APPLE_PRIVATE_KEY_PATH vs _PEM, pick one):

    APPLE_TEAM_ID           10-char team id from the developer portal
    APPLE_CLIENT_ID         the app's bundle id, e.g. com.wemendai.app
    APPLE_KEY_ID            10-char Key ID of the Sign in with Apple .p8 key
    APPLE_PRIVATE_KEY_PEM   contents of the .p8  (or)
    APPLE_PRIVATE_KEY_PATH  path to the .p8 on disk

The .p8 is a private key. Keep it out of the repo and out of logs.

Why the code exchange is not optional
-------------------------------------
Verifying the `identityToken` proves who the user is, and that is all it does. Apple's
revocation endpoint requires a **refresh token**, and the only way to obtain one is to
exchange the `authorizationCode` at sign-up. App Store guideline 5.1.1(v) requires
in-app account deletion, and for Sign in with Apple that means revoking. Skip the
exchange and you cannot comply — discovered at App Review, after the code is written.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx
import jwt
from jwt import PyJWKClient

APPLE_ISSUER = "https://appleid.apple.com"
APPLE_JWKS_URL = f"{APPLE_ISSUER}/auth/keys"
APPLE_TOKEN_URL = f"{APPLE_ISSUER}/auth/token"
APPLE_REVOKE_URL = f"{APPLE_ISSUER}/auth/revoke"

# Apple caps client_secret lifetime at 6 months; short is fine since we mint per call.
_CLIENT_SECRET_TTL = 600


class AppleAuthError(Exception):
    """Anything wrong with an Apple credential. Never leaks the token itself."""


@dataclass(frozen=True)
class AppleConfig:
    team_id: str
    client_id: str
    key_id: str
    private_key: str

    @classmethod
    def from_env(cls) -> AppleConfig:
        pem = os.environ.get("APPLE_PRIVATE_KEY_PEM")
        if not pem:
            path = os.environ.get("APPLE_PRIVATE_KEY_PATH")
            if path and os.path.exists(path):
                pem = open(path).read()
        missing = [
            n for n, v in (
                ("APPLE_TEAM_ID", os.environ.get("APPLE_TEAM_ID")),
                ("APPLE_CLIENT_ID", os.environ.get("APPLE_CLIENT_ID")),
                ("APPLE_KEY_ID", os.environ.get("APPLE_KEY_ID")),
                ("APPLE_PRIVATE_KEY_PEM/_PATH", pem),
            ) if not v
        ]
        if missing:
            raise AppleAuthError(f"Apple config incomplete: missing {', '.join(missing)}")
        return cls(
            team_id=os.environ["APPLE_TEAM_ID"],
            client_id=os.environ["APPLE_CLIENT_ID"],
            key_id=os.environ["APPLE_KEY_ID"],
            private_key=pem,          # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class AppleIdentity:
    """What we learned from a verified identity token."""
    sub: str                  # stable id — THE identity key
    email: str | None         # may be a privaterelay address, may be absent
    is_private_email: bool


# ─────────────────────────── identity token ────────────────────────────
# One client for the process. PyJWKClient caches keys and refreshes on unknown kid,
# which is what we want: Apple rotates its signing keys, so a hardcoded or
# indefinitely-cached key is a time bomb that fires at 3am months from now.
_jwk_client: PyJWKClient | None = None


def _jwks() -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = PyJWKClient(APPLE_JWKS_URL, cache_keys=True, lifespan=3600)
    return _jwk_client


def verify_identity_token(identity_token: str, *, expected_nonce_sha256: str | None,
                          cfg: AppleConfig) -> AppleIdentity:
    """Verify Apple's identity token and return the identity it asserts.

    `expected_nonce_sha256` is the SHA-256 hex of the raw nonce the client generated.
    Apple echoes the hash it was given, so comparing it is what stops a token captured
    from one sign-in being replayed into another session.
    """
    try:
        key = _jwks().get_signing_key_from_jwt(identity_token).key
    except Exception as e:                                    # network, unknown kid…
        raise AppleAuthError(f"could not resolve Apple signing key: {type(e).__name__}") from e

    try:
        claims = jwt.decode(
            identity_token,
            key,
            algorithms=["RS256"],
            audience=cfg.client_id,     # must be OUR app, not any Apple client
            issuer=APPLE_ISSUER,
            options={"require": ["sub", "aud", "iss", "exp"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise AppleAuthError("identity token expired") from e
    except jwt.InvalidAudienceError as e:
        raise AppleAuthError("identity token was not issued for this app") from e
    except jwt.PyJWTError as e:
        raise AppleAuthError(f"identity token invalid: {type(e).__name__}") from e

    if expected_nonce_sha256 is not None:
        got = claims.get("nonce")
        if not got or got != expected_nonce_sha256:
            # Deliberately not logging either value: both are single-use secrets.
            raise AppleAuthError("nonce mismatch")

    sub = claims.get("sub")
    if not sub:
        raise AppleAuthError("identity token has no sub")

    return AppleIdentity(
        sub=sub,
        email=claims.get("email"),
        # Apple sends this as a bool or the strings "true"/"false" depending on flow.
        is_private_email=str(claims.get("is_private_email", "false")).lower() == "true",
    )


# ──────────────────────── client secret & token calls ───────────────────────
def _client_secret(cfg: AppleConfig) -> str:
    """ES256 JWT proving we are this app. Minted per call rather than cached."""
    now = int(time.time())
    return jwt.encode(
        {
            "iss": cfg.team_id,
            "iat": now,
            "exp": now + _CLIENT_SECRET_TTL,
            "aud": APPLE_ISSUER,
            "sub": cfg.client_id,
        },
        cfg.private_key,
        algorithm="ES256",
        headers={"kid": cfg.key_id},
    )


async def exchange_code(authorization_code: str, *, cfg: AppleConfig) -> str:
    """Exchange the one-time authorization code for a refresh token.

    Must happen at sign-up: the code is single-use and short-lived, and the refresh
    token it yields is the only thing that can later revoke this user.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(APPLE_TOKEN_URL, data={
            "client_id": cfg.client_id,
            "client_secret": _client_secret(cfg),
            "code": authorization_code,
            "grant_type": "authorization_code",
        })
    if r.status_code != 200:
        raise AppleAuthError(f"code exchange failed ({r.status_code}): {r.text[:200]}")
    refresh = r.json().get("refresh_token")
    if not refresh:
        raise AppleAuthError("code exchange returned no refresh_token")
    return refresh


async def revoke(refresh_token: str, *, cfg: AppleConfig) -> None:
    """Revoke the user's tokens with Apple. Required before account deletion."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(APPLE_REVOKE_URL, data={
            "client_id": cfg.client_id,
            "client_secret": _client_secret(cfg),
            "token": refresh_token,
            "token_type_hint": "refresh_token",
        })
    # Apple returns 200 with an empty body on success.
    if r.status_code != 200:
        raise AppleAuthError(f"revoke failed ({r.status_code}): {r.text[:200]}")
