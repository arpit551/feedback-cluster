# Unified Clustering Engine Redesign

## Overview
- Replace the dual-algorithm bertopic engine (single-idea cosine similarity + batch UMAP/HDBSCAN) with a single unified approach: OpenAI embeddings + cosine similarity
- Removes `sentence-transformers`, `umap-learn`, `hdbscan`, and `scikit-learn` dependencies
- Adds embedding storage in DB for zero-recomputation, incremental centroid updates, LLM-generated cluster names, and confidence scores
- Improves LLM engine by including representative idea texts in the classification prompt

## Context (from discovery)
- Files involved: `bertopic_engine.py`, `llm_engine.py`, `db.py`, `config.py`, `models.py`, `app.py`, `pyproject.toml`, all test files
- Current bertopic engine has two separate algorithms (single-idea vs batch) that produce inconsistent results
- LLM engine only sends cluster names to GPT, not example ideas -- limits classification accuracy
- Embeddings are recomputed from scratch every time (no caching)
- Heavy ML dependencies (umap, hdbscan, sentence-transformers) are only used in batch mode

## Development Approach
- **testing approach**: Regular (code first, then tests)
- complete each task fully before moving to the next
- make small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task
- **CRITICAL: all tests must pass before starting next task**
- **CRITICAL: update this plan file when scope changes during implementation**
- run tests after each change
- maintain backward compatibility for API endpoints (same routes, same response shapes + new `confidence` field)

## Testing Strategy
- **unit tests**: required for every task
- Mock OpenAI API calls in all tests (both embeddings and chat completions)
- Existing test patterns: `conftest.py` has `test_db`, `db_session`, `client` fixtures and `mock_openai_response` helper
- Test command: `.venv/bin/pytest`
- Lint command: `.venv/bin/ruff check src/ tests/`

## Progress Tracking
- mark completed items with `[x]` immediately when done
- add newly discovered tasks with + prefix
- document issues/blockers with warning prefix
- update plan if implementation deviates from original scope

## Implementation Steps

### Task 1: Update DB schema

**Files:**
- Modify: `src/cluster_api/db.py`

- [ ] Add `embedding` column to `Idea` table (`Column(LargeBinary, nullable=True)`)
- [ ] Add `centroid` column to `Cluster` table (`Column(LargeBinary, nullable=True)`)
- [ ] Add `size` column to `Cluster` table (`Column(Integer, default=0)`)
- [ ] Write tests: verify new columns exist and accept data (serialize/deserialize numpy arrays as bytes)
- [ ] Run tests - must pass before task 2

### Task 2: Update config and remove old parameters

**Files:**
- Modify: `src/cluster_api/config.py`

- [ ] Remove UMAP config: `umap_n_neighbors`, `umap_n_components`, `umap_min_dist`, `umap_metric`
- [ ] Remove HDBSCAN config: `hdbscan_min_cluster_size`, `hdbscan_min_samples`
- [ ] Remove `embedding_model` (no longer using sentence-transformers)
- [ ] Add `openai_embedding_model` (default `text-embedding-3-small`, env var `OPENAI_EMBEDDING_MODEL`)
- [ ] Write tests: verify new config defaults and env var override
- [ ] Run tests - must pass before task 3

### Task 3: Rewrite bertopic engine with OpenAI embeddings + cosine similarity

**Files:**
- Modify: `src/cluster_api/engines/bertopic_engine.py`

- [ ] Remove all imports: `sentence_transformers`, `sklearn`, `numpy` cosine_similarity usage, `umap`, `hdbscan`
- [ ] Add OpenAI client helper (reuse pattern from `llm_engine.py`)
- [ ] Implement `_compute_embedding(text: str) -> np.ndarray` using OpenAI embeddings API
- [ ] Implement `_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float` (simple numpy dot product, 3 lines)
- [ ] Implement `_generate_cluster_name(texts: list[str]) -> str` using OpenAI chat API to generate a 2-4 word label
- [ ] Rewrite `cluster_idea()`:
  - Compute embedding for the idea, store in DB
  - Load all bertopic cluster centroids
  - Compute cosine similarity to each centroid
  - If best >= threshold: assign to cluster, update centroid incrementally (`new_centroid = (centroid * size + embedding) / (size + 1)`), increment size
  - Else: create new cluster, set centroid = embedding, size = 1, generate name via LLM
  - Return result with `confidence` field (the similarity score, or 1.0 for new clusters)
- [ ] Rewrite `recluster_all()`:
  - Clear all bertopic cluster assignments and clusters
  - Load all ideas with stored embeddings (compute + store any missing ones)
  - Re-run `cluster_idea` logic sequentially for each idea (reuse same assign-or-create code)
  - Support `nr_topics` by merging the two closest clusters (by centroid similarity) until target count reached
- [ ] Remove `_get_model()`, `_get_cluster_centroid()`, `_generate_topic_label()` and all UMAP/HDBSCAN code
- [ ] Write tests for `_compute_embedding` (mock OpenAI, verify storage)
- [ ] Write tests for `_cosine_similarity` (known vectors, edge cases)
- [ ] Write tests for `cluster_idea` (new cluster, existing cluster match, threshold boundary)
- [ ] Write tests for `recluster_all` (empty DB, single idea, multiple ideas, nr_topics merging)
- [ ] Run tests - must pass before task 4

### Task 4: Add confidence score to API response

**Files:**
- Modify: `src/cluster_api/models.py`
- Modify: `src/cluster_api/app.py`

- [ ] Add `confidence: float` field to `ClusterIdeaResponse` model
- [ ] Update `/cluster/bertopic` endpoint to pass through confidence from engine result
- [ ] Update `/cluster/llm` endpoint to return `confidence: 1.0` (LLM doesn't produce a numeric score)
- [ ] Write tests: verify confidence field present in bertopic and llm responses
- [ ] Run tests - must pass before task 5

### Task 5: Improve LLM engine with representative ideas in prompt

**Files:**
- Modify: `src/cluster_api/engines/llm_engine.py`

- [ ] When building the prompt, for each existing cluster query the 2-3 ideas closest to the centroid (requires stored embeddings + centroids from bertopic clusters, or just pick first 2-3 ideas if no embeddings available)
- [ ] Update prompt format to include example idea texts under each cluster name
- [ ] Fallback: if no bertopic centroids exist for a cluster, just include the first 2-3 idea texts (by ID order)
- [ ] Write tests: verify prompt includes example ideas when clusters have members
- [ ] Write tests: verify fallback when no embeddings are available
- [ ] Run tests - must pass before task 6

### Task 6: Update dependencies in pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] Remove `sentence-transformers>=3.0.0`
- [ ] Remove `umap-learn>=0.5.0`
- [ ] Remove `hdbscan>=0.8.0`
- [ ] Remove `scikit-learn>=1.5.0`
- [ ] Add `numpy>=1.24.0` (needed for embedding array operations, was previously a transitive dep)
- [ ] Run `pip install -e .` to verify dependencies resolve
- [ ] Run full test suite to verify nothing breaks
- [ ] Run tests - must pass before task 7

### Task 7: Update existing tests for new engine behavior

**Files:**
- Modify: `tests/test_bertopic_clustering.py`
- Modify: `tests/test_batch_clustering.py`
- Modify: `tests/conftest.py`

- [ ] Remove `reset_bertopic_model` fixture (no more cached sentence-transformer model)
- [ ] Add mock fixtures for OpenAI embeddings API calls in bertopic tests
- [ ] Add `mock_openai_embedding_response` helper to `conftest.py`
- [ ] Update `test_bertopic_clustering.py`: mock both OpenAI embeddings and chat completions (for name generation)
- [ ] Update `test_batch_clustering.py`: adapt recluster tests to new unified algorithm, mock OpenAI calls
- [ ] Verify `test_llm_clustering.py` still passes (should need minimal changes)
- [ ] Verify `test_acceptance.py` still passes
- [ ] Run full test suite: `.venv/bin/pytest`
- [ ] Run linter: `.venv/bin/ruff check src/ tests/`

### Task 8: Verify acceptance criteria

- [ ] Verify all requirements from Overview are implemented
- [ ] Verify edge cases: empty DB, single idea, idea with no similar clusters, threshold boundary
- [ ] Run full test suite: `.venv/bin/pytest`
- [ ] Run linter: `.venv/bin/ruff check src/ tests/`
- [ ] Verify test coverage is adequate

### Task 9: [Final] Update documentation

- [ ] Update `.env.example` with new config vars (`OPENAI_EMBEDDING_MODEL`), remove old UMAP/HDBSCAN vars
- [ ] Update CLAUDE.md / MEMORY.md if new patterns discovered
- [ ] Move this plan to `docs/plans/completed/`

## Technical Details

### Embedding storage format
- numpy array serialized via `array.tobytes()`, deserialized via `np.frombuffer(blob, dtype=np.float32)`
- `text-embedding-3-small` produces 1536-dim float32 vectors = 6KB per embedding

### Incremental centroid update
```
new_centroid = (old_centroid * old_size + new_embedding) / (old_size + 1)
new_size = old_size + 1
```

### Cosine similarity (replacing scikit-learn)
```python
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    return float(dot / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### OpenAI embedding API call
```python
response = client.embeddings.create(model=settings.openai_embedding_model, input=[text])
embedding = np.array(response.data[0].embedding, dtype=np.float32)
```

### Cluster name generation prompt
```
Given these user-submitted ideas, generate a 2-4 word topic label:
- "Add dark mode to reduce eye strain"
- "Toggle for light/dark theme"
Respond with just the label, nothing else.
```

## Post-Completion

**Manual verification:**
- Test with real idea data to validate clustering quality
- Tune `SIMILARITY_THRESHOLD` based on real-world results (start at 0.7, may need adjustment)
- Compare clustering results between old and new engine on same dataset

**External system updates:**
- Any consuming services need to handle the new `confidence` field in responses
- Environment variables changed: removed UMAP_*/HDBSCAN_*, added OPENAI_EMBEDDING_MODEL
