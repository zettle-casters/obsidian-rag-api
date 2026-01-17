from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    from ..domain import models  # noqa: F401

    alembic_ini = Path(__file__).resolve().parents[3] / "alembic.ini"
    if not alembic_ini.exists():
        Base.metadata.create_all(bind=engine)
        return

    from alembic import command
    from alembic.config import Config

    config = Config(str(alembic_ini))
    config.set_main_option("script_location", str(alembic_ini.parent / "alembic"))
    config.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(config, "head")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
