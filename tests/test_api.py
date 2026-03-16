from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import create_app


def test_health_endpoint() -> None:
    app = create_app(":memory:")
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_write_and_search_memory_endpoint() -> None:
    app = create_app(":memory:")
    client = TestClient(app)

    write_response = client.post(
        "/memory/write",
        json={
            "memory_type": "episodic",
            "content": "User likes concise Korean explanations.",
            "importance_score": 0.8,
            "semantic_tags": ["user_preference"],
        },
    )
    assert write_response.status_code == 200

    search_response = client.get("/memory/search", params={"query": "Korean explanations"})
    assert search_response.status_code == 200
    body = search_response.json()
    assert body["results"]
    assert "Korean" in body["results"][0]["content"]


def test_document_ingest_and_chat_endpoints() -> None:
    app = create_app(":memory:")
    client = TestClient(app)

    ingest_response = client.post(
        "/document/ingest",
        json={
            "session_id": "s-api",
            "title": "Architecture",
            "text": "Memory hierarchy includes working episodic semantic archival layers. " * 30,
        },
    )
    assert ingest_response.status_code == 200
    ingest_body = ingest_response.json()
    assert ingest_body["total_chunks"] > 0

    chat_response = client.post(
        "/chat",
        json={"session_id": "s-api", "message": "What does memory hierarchy include?"},
    )
    assert chat_response.status_code == 200
    chat_body = chat_response.json()
    assert chat_body["response"]
    assert "memory" in chat_body["response"].lower()
