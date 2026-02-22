# Idea Clustering API

A FastAPI REST API that stores plain-English user ideas in a SQLite database and clusters them into topics using two independent methods:

- **BERTopic** - Embedding-based clustering using sentence-transformers and cosine similarity
- **LLM** - OpenAI-powered clustering using structured JSON output

Ideas are added incrementally. Each new idea is compared against existing clusters and either assigned to a matching one or triggers the creation of a new cluster.

## Setup

### Requirements

- Python 3.9+

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required for LLM clustering) | `""` |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4o-mini` |
| `CLUSTER_API_DB_PATH` | Path to SQLite database file | `ideas.db` |
| `EMBEDDING_MODEL` | Sentence-transformers model for BERTopic | `all-MiniLM-L6-v2` |
| `SIMILARITY_THRESHOLD` | Cosine similarity threshold for BERTopic cluster assignment | `0.7` |

### Running the Server

```bash
uvicorn cluster_api.app:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

### Health Check

```
GET /health
```

Returns `{"status": "ok"}`.

### Ideas

#### Add an Idea

```
POST /ideas
Content-Type: application/json

{
  "text": "Add dark mode to the mobile app",
  "user_id": "user-42"
}
```

Response (201):
```json
{
  "idea_id": 1
}
```

#### List All Ideas

```
GET /ideas
```

Response:
```json
[
  {
    "id": 1,
    "text": "Add dark mode to the mobile app",
    "user_id": "user-42"
  }
]
```

### Clustering

#### Cluster with BERTopic

Assigns an existing idea to a cluster using embedding similarity:

```
POST /cluster/bertopic
Content-Type: application/json

{
  "idea_id": 1
}
```

Response:
```json
{
  "idea_id": 1,
  "cluster_id": 1,
  "cluster_name": "Add dark mode to the mobile app",
  "is_new": true
}
```

#### Cluster with LLM

Assigns an existing idea to a cluster using OpenAI (requires `OPENAI_API_KEY`):

```
POST /cluster/llm
Content-Type: application/json

{
  "idea_id": 1
}
```

Response:
```json
{
  "idea_id": 1,
  "cluster_id": 1,
  "cluster_name": "UI/UX Improvements",
  "is_new": true
}
```

#### List BERTopic Clusters

```
GET /clusters/bertopic
```

Response:
```json
[
  {
    "cluster_id": 1,
    "name": "Add dark mode to the mobile app",
    "ideas": [
      {
        "id": 1,
        "text": "Add dark mode to the mobile app",
        "user_id": "user-42"
      }
    ]
  }
]
```

#### List LLM Clusters

```
GET /clusters/llm
```

Response format is the same as BERTopic clusters.

## Development

### Running Tests

```bash
pytest
```

### Running Tests with Coverage

```bash
pytest --cov=cluster_api --cov-report=term-missing
```

### Linting

```bash
ruff check src/ tests/
```
