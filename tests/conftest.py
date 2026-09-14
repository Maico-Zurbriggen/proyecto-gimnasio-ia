from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from gym_engine.persistence.models import Base


@pytest.fixture
def sqlite_session_factory() -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture
def session(sqlite_session_factory: sessionmaker[Session]) -> Iterator[Session]:
    db_session = sqlite_session_factory()
    yield db_session
    db_session.close()
