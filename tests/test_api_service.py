"""Service-level endpoints (§18 Phase 6, Phase 7).

The dashboard is served from the same origin as the API it reads, so it is a
client of the endpoints rather than a second implementation of them: whatever
it shows, some endpoint returned. These tests pin the two service routes and
the generated schema - a schema that fails to build is a broken contract even
when every route works.
"""


class TestService:
    def test_health(self, api_client):
        assert api_client.get("/health").json() == {"status": "ok"}

    def test_dashboard_is_served_from_the_api_origin(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "Pull Planner" in response.text

    def test_openapi_documents_the_design_document_endpoints(self, api_client):
        paths = api_client.get("/openapi.json").json()["paths"]
        for path in [
            "/accounts",
            "/accounts/{account_id}",
            "/accounts/{account_id}/goals",
            "/accounts/{account_id}/preferences",
            "/accounts/{account_id}/banners",
            "/accounts/{account_id}/income",
            "/accounts/{account_id}/probability/character",
            "/accounts/{account_id}/probability/weapon",
            "/accounts/{account_id}/probability/wishes-needed",
            "/accounts/{account_id}/planner/recommendation",
            "/accounts/{account_id}/planner/safe-spend",
            "/accounts/{account_id}/planner/spend-table",
            "/accounts/{account_id}/planner/stop-conditions",
            "/accounts/{account_id}/simulation/run",
            "/accounts/{account_id}/simulation/{job_id}",
        ]:
            assert path in paths, path
