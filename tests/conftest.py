import json
from unittest.mock import MagicMock

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


def add_idea(client, text, user_id="u1"):
    """Helper to add an idea via the API and return the idea_id."""
    resp = client.post("/ideas", json={"text": text, "user_id": user_id})
    assert resp.status_code == 201
    return resp.json()["idea_id"]


def mock_openai_response(cluster_name: str, is_new: bool):
    """Create a mock OpenAI chat completion response."""
    mock_message = MagicMock()
    mock_message.content = json.dumps(
        {"cluster_name": cluster_name, "is_new": is_new}
    )
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def mock_openai_chat_response(text: str):
    """Create a mock OpenAI chat completion response with plain text content."""
    mock_message = MagicMock()
    mock_message.content = text
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


def make_mock_embedding_response(embedding):
    """Create a mock OpenAI embeddings API response."""
    mock_data = MagicMock()
    mock_data.embedding = list(embedding)
    mock_response = MagicMock()
    mock_response.data = [mock_data]
    return mock_response
