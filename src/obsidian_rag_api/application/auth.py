from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..infrastructure.config import settings
from ..infrastructure.db import get_db
from ..domain.models import McpToken, OAuthState, Session as DbSession, User


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def _code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("utf-8")


def build_google_auth_redirect(return_to: str, db: Session) -> str:
    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = _code_challenge(code_verifier)

    db.add(OAuthState(state=state, code_verifier=code_verifier, redirect_uri=return_to))
    db.commit()

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


async def exchange_code_for_user(code: str, state: str, db: Session) -> tuple[User, str]:
    record = db.query(OAuthState).filter(OAuthState.state == state).first()
    if record is None:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    data = {
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.google_redirect_uri,
        "code_verifier": record.code_verifier,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        token_resp = await client.post(GOOGLE_TOKEN_URL, data=data)
        token_resp.raise_for_status()
        token_json = token_resp.json()
        access_token = token_json.get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="Failed to fetch access token")

        userinfo_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        userinfo_resp.raise_for_status()
        userinfo = userinfo_resp.json()

    db.delete(record)
    db.commit()

    google_sub = userinfo.get("sub")
    if not google_sub:
        raise HTTPException(status_code=400, detail="Google user info missing sub")

    user = db.query(User).filter(User.google_sub == google_sub).first()
    if user is None:
        has_admin = (
            db.query(User)
            .filter(User.is_demo.is_(False))
            .count()
            > 0
        )
        user = User(
            google_sub=google_sub,
            email=userinfo.get("email"),
            name=userinfo.get("name"),
            avatar_url=userinfo.get("picture"),
            is_demo=False,
            is_admin=not has_admin,
        )
        db.add(user)
    else:
        user.email = userinfo.get("email")
        user.name = userinfo.get("name")
        user.avatar_url = userinfo.get("picture")

    db.commit()
    return user, record.redirect_uri


def ensure_demo_user(db: Session) -> User:
    user = db.query(User).filter(User.is_demo.is_(True)).first()
    if user is None:
        user = User(
            email=settings.demo_user_email,
            name="Demo Workspace",
            is_demo=True,
            is_admin=False,
        )
        db.add(user)
        db.commit()
    return user


def _generate_mcp_token() -> str:
    return f"mcp_{secrets.token_urlsafe(32)}"


def get_or_create_mcp_token(user: User, db: Session) -> McpToken:
    record = db.query(McpToken).filter(McpToken.user_id == user.id).first()
    if record:
        return record
    record = McpToken(user_id=user.id, token=_generate_mcp_token())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def rotate_mcp_token(user: User, db: Session) -> McpToken:
    record = db.query(McpToken).filter(McpToken.user_id == user.id).first()
    if record is None:
        record = McpToken(user_id=user.id, token=_generate_mcp_token())
        db.add(record)
    else:
        record.token = _generate_mcp_token()
        record.created_at = datetime.utcnow()
        record.last_used_at = None
    db.commit()
    db.refresh(record)
    return record


def get_user_by_mcp_token(token: str, db: Session) -> Optional[User]:
    if not token:
        return None
    record = db.query(McpToken).filter(McpToken.token == token).first()
    if record is None:
        return None
    record.last_used_at = datetime.utcnow()
    db.commit()
    return record.user


def create_session(user: User, db: Session) -> DbSession:
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.session_ttl_days)
    session = DbSession(user_id=user.id, token=token, expires_at=expires_at)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _get_active_session(token: str | None, db: Session) -> DbSession | None:
    if not token:
        return None
    now = datetime.now(timezone.utc)
    return (
        db.query(DbSession)
        .filter(DbSession.token == token, DbSession.expires_at > now)
        .first()
    )


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get(settings.session_cookie_name)
    session = _get_active_session(token, db)
    if session:
        return session.user
    return ensure_demo_user(db)


def get_optional_user(request: Request, db: Session) -> Optional[User]:
    token = request.cookies.get(settings.session_cookie_name)
    session = _get_active_session(token, db)
    return session.user if session else None
