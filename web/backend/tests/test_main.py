from pathlib import Path

import httpx
import pytest

from solar_web.main import create_app


@pytest.mark.asyncio
async def test_serves_built_frontend_when_static_directory_is_provided(tmp_path: Path) -> None:
    static_dir = tmp_path / "dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<html><body>SOLAR Web</body></html>")
    app = create_app(runtime_root=tmp_path / "runs", solar_root=Path(__file__).resolve().parents[3], static_dir=static_dir)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "SOLAR Web" in response.text
