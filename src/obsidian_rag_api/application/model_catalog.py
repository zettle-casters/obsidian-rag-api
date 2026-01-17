from __future__ import annotations

from datetime import datetime
from typing import Iterable

from sqlalchemy.orm import Session

from ..domain.models import LlmModel
from ..infrastructure.config import settings


DEFAULT_MODEL_DISPLAY = "GPT OSS 120B"
DEFAULT_MODEL_SYSTEM = "openai/gpt-oss-120b"
DEFAULT_MODEL_DESCRIPTION = "Базовая модель для ассистента."
DEFAULT_MODEL_AVATAR = settings.default_model_avatar_url


def ensure_default_models(db: Session) -> None:
    if db.query(LlmModel).count() > 0:
        return
    model = LlmModel(
        display_name=DEFAULT_MODEL_DISPLAY,
        system_name=DEFAULT_MODEL_SYSTEM,
        description=DEFAULT_MODEL_DESCRIPTION,
        avatar_url=DEFAULT_MODEL_AVATAR,
        is_enabled=True,
    )
    db.add(model)
    db.commit()


def list_models(db: Session, include_disabled: bool = False) -> Iterable[LlmModel]:
    query = db.query(LlmModel)
    if not include_disabled:
        query = query.filter(LlmModel.is_enabled.is_(True))
    return query.order_by(LlmModel.display_name.asc()).all()


def create_model(
    db: Session,
    display_name: str,
    system_name: str,
    description: str | None,
    avatar_url: str | None,
    is_enabled: bool,
) -> LlmModel:
    model = LlmModel(
        display_name=display_name,
        system_name=system_name,
        description=description,
        avatar_url=avatar_url,
        is_enabled=is_enabled,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def update_model(
    db: Session,
    model: LlmModel,
    display_name: str | None = None,
    system_name: str | None = None,
    description: str | None = None,
    avatar_url: str | None = None,
    is_enabled: bool | None = None,
) -> LlmModel:
    if display_name is not None:
        model.display_name = display_name
    if system_name is not None:
        model.system_name = system_name
    if description is not None:
        model.description = description
    if avatar_url is not None:
        model.avatar_url = avatar_url
    if is_enabled is not None:
        model.is_enabled = is_enabled
    model.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(model)
    return model
