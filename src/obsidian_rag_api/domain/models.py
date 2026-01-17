from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..infrastructure.db import Base


def _uuid_str() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True)
    email: Mapped[str | None] = mapped_column(String(320), unique=True)
    name: Mapped[str | None] = mapped_column(String(255))
    avatar_url: Mapped[str | None] = mapped_column(Text)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    token: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(User)


class OAuthState(Base):
    __tablename__ = "oauth_states"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    state: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    code_verifier: Mapped[str] = mapped_column(String(256))
    redirect_uri: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Vault(Base):
    __tablename__ = "vaults"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    include_paths: Mapped[list[str]] = mapped_column(JSONB, default=list)
    exclude_paths: Mapped[list[str]] = mapped_column(JSONB, default=list)
    chunk_size: Mapped[int] = mapped_column(Integer, default=500)
    notes_count: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped[User] = relationship(User)


class VaultFile(Base):
    __tablename__ = "vault_files"
    __table_args__ = (UniqueConstraint("vault_id", "path", name="uq_vault_files"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    vault_id: Mapped[str] = mapped_column(String(36), ForeignKey("vaults.id"), index=True)
    path: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    vault: Mapped[Vault] = relationship(Vault)


class McpToken(Base):
    __tablename__ = "mcp_tokens"
    __table_args__ = (UniqueConstraint("user_id", name="uq_mcp_tokens_user_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    token: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(User)


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    vault_id: Mapped[str] = mapped_column(String(36), ForeignKey("vaults.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="Новый чат")
    model_name: Mapped[str | None] = mapped_column(String(255))
    thread_id: Mapped[str] = mapped_column(String(64), default=_uuid_str)
    is_shared: Mapped[bool] = mapped_column(Boolean, default=False)
    share_token: Mapped[str | None] = mapped_column(String(128), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    user: Mapped[User] = relationship(User)
    vault: Mapped[Vault] = relationship(Vault)
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="chat",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    chat_id: Mapped[str] = mapped_column(String(36), ForeignKey("chats.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    chat: Mapped[Chat] = relationship(Chat, back_populates="messages")


class LlmModel(Base):
    __tablename__ = "llm_models"
    __table_args__ = (UniqueConstraint("system_name", name="uq_llm_models_system_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid_str)
    display_name: Mapped[str] = mapped_column(String(255))
    system_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    avatar_url: Mapped[str | None] = mapped_column(Text)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
