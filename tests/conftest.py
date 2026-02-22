import pytest
from fastapi.testclient import TestClient

from cluster_api.db import get_session, init_db


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


@pytest.fixture
def client(test_db):
    """Provide a TestClient that uses the test database (already initialized by test_db)."""
    from cluster_api.app import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
