from fastapi.testclient import TestClient

from rag_eval.main import RAGEvalAPI


def test_health_endpoint_responds():
    api = RAGEvalAPI()
    with TestClient(api.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "qdrant" in body
    assert "fuelix_configured" in body


def test_pipelines_listing():
    api = RAGEvalAPI()
    with TestClient(api.app) as client:
        response = client.get("/experiments")
    assert response.status_code == 200
    pipelines = response.json()
    assert len(pipelines) == 9
    ids = {p["pipeline_id"] for p in pipelines}
    assert ids == {f"P{i}" for i in range(1, 10)}
