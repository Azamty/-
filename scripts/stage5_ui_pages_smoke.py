"""Use Chromium to browse and download the completed long score pages."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import requests
import websockets

from scripts.stage4_ui_smoke import CDP, free_port, wait_for_target


ROOT = Path(__file__).resolve().parents[1]
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
RUN_ROOT = ROOT / "artifacts" / "review" / "stage5-final" / "run-20260904T120124Z"
JOBS_ROOT = RUN_ROOT / "jobs"
JOB_ID = "2621a320-3d83-4729-a6df-a4d6e641df9b"
EVIDENCE_DIR = ROOT / "artifacts" / "review" / "stage5-final"
SCREENSHOT = EVIDENCE_DIR / "stage5-pages-ui.png"
SUMMARY = EVIDENCE_DIR / "stage5-pages-ui.json"


async def main_async() -> int:
    if not CHROME.is_file() or not (JOBS_ROOT / JOB_ID / "job.json").is_file():
        raise FileNotFoundError("stage5 full run artifacts or Chrome is missing")
    server_port = free_port()
    server = subprocess.Popen(
        [
            os.fspath(ROOT / ".venv" / "Scripts" / "python.exe"),
            "-m",
            "scripts.stage5_review_server",
            os.fspath(JOBS_ROOT),
            "--port",
            str(server_port),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    chrome: subprocess.Popen[bytes] | None = None
    try:
        health_deadline = time.monotonic() + 20
        while time.monotonic() < health_deadline:
            try:
                if requests.get(f"http://127.0.0.1:{server_port}/api/health", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError("stage5 review server did not start")

        chrome_port = free_port()
        profile = EVIDENCE_DIR / "stage5-pages-chrome-profile"
        profile.mkdir(parents=True, exist_ok=True)
        chrome = subprocess.Popen(
            [
                os.fspath(CHROME),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--no-first-run",
                "--no-default-browser-check",
                f"--remote-debugging-port={chrome_port}",
                f"--user-data-dir={profile}",
                "--window-size=1440,1200",
                f"http://127.0.0.1:{server_port}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        target = await wait_for_target(chrome_port)
        async with websockets.connect(str(target["webSocketDebuggerUrl"]), max_size=None) as websocket:
            cdp = CDP(websocket)
            await cdp.call("Page.enable")
            await cdp.call("DOM.enable")
            await asyncio.sleep(1.0)
            await cdp.evaluate(
                f"localStorage.setItem('jianpu-job-id',{json.dumps(JOB_ID)}); location.reload();"
            )
            deadline = time.monotonic() + 20
            body = ""
            while time.monotonic() < deadline:
                body = str(await cdp.evaluate("document.body.innerText"))
                if "谱页已经展开" in body and "器乐 分谱" in body:
                    break
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError("long score was not rendered in the browser")

            await cdp.evaluate(
                """[...document.querySelectorAll('.score-tabs button')]
                .find((button) => button.textContent.includes('器乐 分谱'))?.click()"""
            )
            await asyncio.sleep(0.5)
            first_page = await cdp.evaluate(
                """({tab:document.querySelector('.score-tabs button.active')?.textContent,
                   image:document.querySelector('.score-viewer img')?.src,
                   page:document.querySelector('.page-nav span')?.textContent})"""
            )
            await cdp.evaluate(
                "[...document.querySelectorAll('.page-nav button')].find((button) => button.textContent.includes('下一页'))?.click()"
            )
            await asyncio.sleep(0.5)
            second_page = await cdp.evaluate(
                """({tab:document.querySelector('.score-tabs button.active')?.textContent,
                   image:document.querySelector('.score-viewer img')?.src,
                   page:document.querySelector('.page-nav span')?.textContent})"""
            )
            links = await cdp.evaluate(
                "[...document.querySelectorAll('a.download-action')].map((a)=>({text:a.textContent.trim(),href:a.href}))"
            )
            if not isinstance(second_page, dict) or "第 2 / 2 页" not in str(second_page.get("page")):
                raise AssertionError(f"next-page control did not show page 2: {second_page}")
            if not isinstance(second_page.get("image"), str) or "stem-other-svg-2" not in second_page["image"]:
                raise AssertionError(f"browser did not switch to other stem page 2: {second_page}")

            download_checks: list[dict[str, Any]] = []
            for link in links if isinstance(links, list) else []:
                href = str(link.get("href", ""))
                response = requests.get(href, timeout=20)
                download_checks.append(
                    {
                        "text": link.get("text"),
                        "url": href,
                        "status": response.status_code,
                        "bytes": len(response.content),
                    }
                )
                if response.status_code != 200 or not response.content:
                    raise AssertionError(f"browser download link failed: {download_checks[-1]}")

            layout = await cdp.call("Page.getLayoutMetrics")
            content_size = layout.get("cssContentSize", {})
            screenshot = await cdp.call(
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "captureBeyondViewport": True,
                    "fromSurface": True,
                    "clip": {
                        "x": 0,
                        "y": 0,
                        "width": max(1440, float(content_size.get("width", 1440))),
                        "height": min(2200, max(1200, float(content_size.get("height", 1200)))),
                        "scale": 1,
                    },
                },
            )
            SCREENSHOT.write_bytes(base64.b64decode(str(screenshot["data"])))
            summary = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "server": f"http://127.0.0.1:{server_port}",
                "job_id": JOB_ID,
                "score_tab": second_page.get("tab") if isinstance(second_page, dict) else None,
                "first_page": first_page,
                "second_page": second_page,
                "download_checks": download_checks,
                "screenshot": str(SCREENSHOT),
                "body_contains_completed": "谱页已经展开" in body,
            }
            SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            # Keep the evidence file UTF-8; PowerShell consoles on this host
            # may still use a legacy code page for stdout.
            print(json.dumps(summary, ensure_ascii=True, indent=2))
    finally:
        if chrome is not None:
            try:
                chrome.terminate()
                chrome.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                chrome.kill()
        try:
            server.terminate()
            server.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            server.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
