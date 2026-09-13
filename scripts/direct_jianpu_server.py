"""Run the PDF score branch with its existing local review job directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import create_app
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount
import uvicorn

review_root = ROOT / "artifacts" / "review" / "direct-jianpu"
review_root.mkdir(parents=True, exist_ok=True)
app = create_app(jobs_root=review_root / "jobs")
# Keep saved PDF jobs available across later restarts of this local workspace.
app.state.jobs.cleanup_expired = lambda **kwargs: []
app.router.routes.insert(0, Mount("/review", app=StaticFiles(directory=review_root)))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8012, workers=1)
