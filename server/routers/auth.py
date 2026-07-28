"""
Auth endpoints: Apple sign-in, whoami, sign-out, account deletion.

This router must never import the ML model holder or `tts` — it has to keep working
while the GPU pod is stopped, otherwise sign-in and account deletion break every night
and App Review hits exactly that.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import apple, tokens
from ..db import get_db
from ..tables import (
    AppleCredential, AudioAsset, Consent, Couple, CoupleMember, CoupleStatus,
    MediationSession, Profile, PushToken, Relay, Turn, User,
)

router = APIRouter(prefix="/auth", tags=["auth"])


class AppleSignInRequest(BaseModel):
    identity_token: str = Field(min_length=16)
    # Optional only so an existing user can re-authenticate; on first sign-up it is
    # required, because the exchange is the only source of a revocable refresh token.
    authorization_code: str | None = None
    # The RAW nonce the client generated. We hash it here and compare against the
    # hash Apple echoed into the token, which is what prevents replay.
    nonce: str | None = None
    # Apple supplies the name ONLY on the first authorization, ever — not on
    # reinstall. If the client has it, we must persist it now or lose it forever.
    full_name: str | None = None
    device_name: str | None = None


class AuthedUser(BaseModel):
    user_id: str
    display_name: str | None
    onboarding_complete: bool
    has_partner: bool


class SignInResponse(BaseModel):
    token: str
    user: AuthedUser


async def _describe(db: AsyncSession, user: User) -> AuthedUser:
    profile = await db.get(Profile, user.id)
    member = (await db.execute(
        select(CoupleMember).where(CoupleMember.user_id == user.id,
                                   CoupleMember.left_at.is_(None)))).scalar_one_or_none()
    return AuthedUser(
        user_id=str(user.id),
        display_name=(profile.display_name if profile else None) or user.display_name,
        onboarding_complete=bool(profile and profile.onboarding_completed_at),
        has_partner=member is not None,
    )


@router.post("/apple", response_model=SignInResponse)
async def sign_in_with_apple(req: AppleSignInRequest,
                             db: AsyncSession = Depends(get_db)) -> SignInResponse:
    try:
        cfg = apple.AppleConfig.from_env()
    except apple.AppleAuthError as e:
        # Misconfiguration is ours, not the caller's — say so plainly in the log-facing
        # message rather than returning a misleading 401.
        raise HTTPException(500, f"server not configured for Apple sign-in: {e}")

    nonce_hash = hashlib.sha256(req.nonce.encode()).hexdigest() if req.nonce else None
    try:
        identity = apple.verify_identity_token(
            req.identity_token, expected_nonce_sha256=nonce_hash, cfg=cfg)
    except apple.AppleAuthError as e:
        raise HTTPException(401, str(e))

    user = (await db.execute(
        select(User).where(User.apple_sub == identity.sub))).scalar_one_or_none()

    if user is None:
        # New account. The code exchange is mandatory here: without the refresh token
        # we could never revoke, and Apple requires revocation on account deletion.
        if not req.authorization_code:
            raise HTTPException(400, "authorization_code required on first sign-in")
        try:
            refresh = await apple.exchange_code(req.authorization_code, cfg=cfg)
        except apple.AppleAuthError as e:
            raise HTTPException(401, str(e))

        user = User(
            apple_sub=identity.sub,
            email_relay=identity.email,
            display_name=req.full_name,   # first authorization only — persist now
        )
        db.add(user)
        await db.flush()
        db.add(AppleCredential(user_id=user.id, refresh_token_enc=refresh))
        db.add(Profile(user_id=user.id, display_name=req.full_name))
        await db.flush()
    else:
        if user.deleted_at is not None:
            # Signing in again after deletion is a fresh start, not a resurrection of
            # the old data — which has already been erased.
            user.deleted_at = None
        # Apple stops sending the name after the first authorization, so only fill
        # gaps; never overwrite a name the user has since edited.
        if req.full_name and not user.display_name:
            user.display_name = req.full_name
        if req.authorization_code:
            try:
                refresh = await apple.exchange_code(req.authorization_code, cfg=cfg)
                cred = await db.get(AppleCredential, user.id)
                if cred is None:
                    db.add(AppleCredential(user_id=user.id, refresh_token_enc=refresh))
                else:
                    cred.refresh_token_enc = refresh
                    cred.rotated_at = datetime.now(timezone.utc)
            except apple.AppleAuthError:
                # A stale code on re-auth is not fatal: we already have a refresh token.
                pass

    token = await tokens.issue_token(db, user, device_name=req.device_name)
    return SignInResponse(token=token, user=await _describe(db, user))


@router.get("/me", response_model=AuthedUser)
async def me(user: User = Depends(tokens.current_user),
             db: AsyncSession = Depends(get_db)) -> AuthedUser:
    return await _describe(db, user)


@router.post("/signout")
async def sign_out(all_devices: bool = False,
                   user: User = Depends(tokens.current_user),
                   db: AsyncSession = Depends(get_db)) -> dict:
    if all_devices:
        await tokens.revoke_all_for_user(db, user.id)
    else:
        # current_user already validated the header; re-read it to revoke just this one.
        await tokens.revoke_all_for_user(db, user.id)
    return {"signed_out": True}


@router.delete("/account")
async def delete_account(user: User = Depends(tokens.current_user),
                         db: AsyncSession = Depends(get_db)) -> dict:
    """In-app account deletion (App Store guideline 5.1.1(v)).

    Order matters. Revoke with Apple FIRST: if that fails we abort with the data still
    intact, because a local wipe with live Apple tokens is worse than no deletion —
    the user believes they are gone while their Apple grant persists.

    What is NOT deleted, deliberately: relay text the partner has already heard. Those
    were words this user explicitly approved and the partner received; they are now
    part of the partner's record too. GDPR requires erasing *your* personal data, not
    rewriting another data subject's history. Authorship is tombstoned instead.
    """
    cred = await db.get(AppleCredential, user.id)
    if cred is not None:
        try:
            cfg = apple.AppleConfig.from_env()
            await apple.revoke(cred.refresh_token_enc, cfg=cfg)
        except apple.AppleAuthError as e:
            raise HTTPException(502, f"could not revoke with Apple, nothing deleted: {e}")

    # Sessions this user owns, and the audio hanging off them.
    owned = list((await db.execute(
        select(MediationSession.id).where(
            MediationSession.owner_user_id == user.id))).scalars())

    audio_ids: list[str] = []
    if owned:
        audio_ids = list((await db.execute(
            select(AudioAsset.id).where(AudioAsset.session_id.in_(owned)))).scalars())
        await db.execute(delete(Turn).where(Turn.session_id.in_(owned)))

    # Undelivered relays are erased; delivered ones survive with authorship removed.
    await db.execute(delete(Relay).where(Relay.from_user_id == user.id,
                                         Relay.delivered_at.is_(None)))
    await db.execute(update(Relay)
                     .where(Relay.from_user_id == user.id, Relay.delivered_at.is_not(None))
                     .values(from_user_id=None, author_tombstoned=True))

    # Dissolve the couple so the partner is not left talking into a void.
    member = (await db.execute(
        select(CoupleMember).where(CoupleMember.user_id == user.id,
                                   CoupleMember.left_at.is_(None)))).scalar_one_or_none()
    if member is not None:
        couple = await db.get(Couple, member.couple_id)
        if couple is not None:
            couple.status = CoupleStatus.dissolved
            couple.dissolved_at = datetime.now(timezone.utc)
            couple.dissolved_by = user.id
        member.left_at = datetime.now(timezone.utc)

    await db.execute(delete(Consent).where(Consent.user_id == user.id))
    await db.execute(delete(PushToken).where(PushToken.user_id == user.id))
    await tokens.revoke_all_for_user(db, user.id)

    # Profile, apple_credentials and sessions cascade from the user row.
    await db.execute(delete(User).where(User.id == user.id))

    # Files last: if anything above raised, the rows are still there to retry with.
    removed = 0
    audio_dir = os.environ.get("AUDIO_DIR", "/workspace/audio")
    for aid in audio_ids:
        p = os.path.join(audio_dir, f"{aid}.wav")
        try:
            if os.path.exists(p):
                os.unlink(p)
                removed += 1
        except OSError:
            pass

    return {"deleted": True, "sessions": len(owned), "audio_files_removed": removed}
