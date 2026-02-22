import os
import tempfile

import pytest

from cluster_api.db import Base, init_db, get_session, _engine


@pytest.fixture(autouse=True)
def test_db(tmp_path):
    """Initialize a fresh test database for each test."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    yield
    # Clean up global state
    import cluster_api.db as db_module
    if db_module._engine:
        db_module._engine.dispose()
    db_module._engine = None
    db_module._SessionLocal = None


@pytest.fixture
def db_session(test_db):
    """Provide a DB session for tests."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()
