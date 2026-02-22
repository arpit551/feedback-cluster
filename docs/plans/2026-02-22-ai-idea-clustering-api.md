# AI-Powered Idea Clustering API

## Overview

Build a FastAPI REST API that stores plain-English user ideas in a SQLite database and clusters them into topics using two independent methods: BERTopic (embedding-based) and LLM-based (structured JSON output). Ideas are added incrementally - each new idea is checked against existing clusters and either assigned to one or triggers a new cluster.

## Context

- Files involved: All new (greenfield project)
- Related patterns: None (fresh repo)
- Dependencies: fastapi, uvicorn, bertopic, sentence-transformers, openai, pydantic, sqlalchemy, aiosqlite, scikit-learn, pytest, httpx

## Development Approach

- **Testing approach**: Regular (code first, then tests)
- Tests are integration tests that hit the actual FastAPI endpoints via TestClient - no isolated unit test files
- Complete each task fully before moving to the next
- **CRITICAL: every task MUST include new/updated tests**
- **CRITICAL: all tests must pass before starting next task**

## Implementation Steps

### Task 1: Project setup, database, and data models

**Files:**
- Create: `pyproject.toml`
- Create: `src/cluster_api/__init__.py`
- Create: `src/cluster_api/models.py` (Pydantic request/response schemas)
- Create: `src/cluster_api/db.py` (SQLAlchemy models and DB setup)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py` (shared TestClient fixture)

- [x] Initialize pyproject.toml with dependencies (fastapi, uvicorn, bertopic, sentence-transformers, openai, pydantic, sqlalchemy, aiosqlite, pytest, httpx)
- [x] Define SQLAlchemy models: Idea table (id, text, user_id, created_at), Cluster table (id, name, method), IdeaCluster join table (idea_id, cluster_id)
- [x] Define Pydantic schemas: AddIdeaRequest (text, user_id), AddIdeaResponse (idea_id), ClusterResponse (cluster_id, name, ideas list)
- [x] Create DB initialization (SQLite file, create tables on startup)
- [x] Write conftest.py with a TestClient fixture using a test SQLite DB
- [x] Run test suite - must pass before task 2

### Task 2: Add Ideas endpoint with DB storage

**Files:**
- Create: `src/cluster_api/app.py`
- Create: `src/cluster_api/config.py`
- Create: `tests/test_ideas.py`

- [x] Create config module (OpenAI API key from env, DB path, model defaults)
- [x] Create FastAPI app with lifespan that initializes DB on startup
- [x] Implement POST /ideas endpoint - accepts idea text and user_id, stores raw idea in DB, returns idea_id
- [x] Implement GET /ideas endpoint - returns all stored ideas
- [x] Add GET /health endpoint
- [x] Write integration tests via TestClient: POST an idea, verify it is stored, GET ideas and verify the list
- [x] Run test suite - must pass before task 3

### Task 3: BERTopic clustering engine with incremental assignment

**Files:**
- Create: `src/cluster_api/engines/__init__.py`
- Create: `src/cluster_api/engines/bertopic_engine.py`
- Create: `tests/test_bertopic_clustering.py`

- [x] Implement cluster_idea function: given a new idea text, compute its embedding using sentence-transformers
- [x] Load existing BERTopic clusters from DB (method=bertopic). Compare the new idea embedding against existing cluster centroids
- [x] If the idea is close enough to an existing cluster (cosine similarity above threshold), assign it to that cluster
- [x] If the idea does not fit any existing cluster, create a new cluster using BERTopic topic representation as the name, save to DB
- [x] Store the idea-cluster assignment in the IdeaCluster table
- [x] Implement POST /cluster/bertopic endpoint that triggers BERTopic clustering for a given idea_id (idea must already exist in DB)
- [x] Implement GET /clusters/bertopic endpoint that returns all BERTopic clusters with their assigned ideas
- [x] Write integration tests via TestClient: POST several ideas, cluster each via the endpoint, verify clusters are created and ideas are assigned correctly
- [x] Run test suite - must pass before task 4

### Task 4: LLM-based clustering engine with incremental assignment

**Files:**
- Create: `src/cluster_api/engines/llm_engine.py`
- Create: `tests/test_llm_clustering.py`

- [ ] Implement cluster_idea function: given a new idea text, load existing LLM clusters (method=llm) from DB
- [ ] Build a prompt that provides existing cluster names and asks the LLM whether the new idea fits an existing cluster or needs a new one
- [ ] Use OpenAI API with structured JSON schema (response_format) to get a decision: {cluster_name: str, is_new: bool}
- [ ] If the LLM says the idea fits an existing cluster, assign it. If new, create the cluster with the LLM-suggested name
- [ ] Store the idea-cluster assignment in the IdeaCluster table
- [ ] Implement POST /cluster/llm endpoint that triggers LLM clustering for a given idea_id
- [ ] Implement GET /clusters/llm endpoint that returns all LLM clusters with their assigned ideas
- [ ] Write integration tests via TestClient: POST ideas, cluster via endpoint, verify correct assignment (mock OpenAI API responses for CI reliability)
- [ ] Run test suite - must pass before task 5

### Task 5: Verify acceptance criteria

- [ ] Manual test: POST 10+ ideas via /ideas, cluster each with both /cluster/bertopic and /cluster/llm, verify clusters form sensibly
- [ ] Manual test: add a new idea that should fit an existing cluster, verify it gets assigned rather than creating a new cluster
- [ ] Run full test suite (pytest)
- [ ] Run linter (ruff check)
- [ ] Verify test coverage meets 80%+

### Task 6: Update documentation

- [ ] Update README.md with project description, setup instructions, API usage examples, and environment variable configuration
- [ ] Move this plan to `docs/plans/completed/`
