"""Exercise the shipped UI in a real Chromium page and save a screenshot.

The CUA browser surface is not present on this host, so this evidence runner
uses the preinstalled Chrome executable through its local DevTools endpoint.
It still selects the real WAV in the page's file input, clicks the real submit
button, waits for the browser's polling loop, and captures the rendered result.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import requests
import websockets


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts" / "review" / "scale_reference.wav"
EVIDENCE_DIR = ROOT / "artifacts" / "review"
SCREENSHOT = EVIDENCE_DIR / "stage4-ui.png"
SUMMARY = EVIDENCE_DIR / "stage4-ui.json"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def wait_for_target(port: int) -> dict[str, Any]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/json", timeout=1)
            pages = [item for item in response.json() if item.get("type") == "page"]
            if pages and pages[0].get("webSocketDebuggerUrl"):
                return pages[0]
        except (OSError, requests.RequestException, ValueError):
            pass
        await asyncio.sleep(0.25)
    raise TimeoutError("Chrome DevTools target did not start")


class CDP:
    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self.sequence = 0

    async def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.sequence += 1
        identifier = self.sequence
        await self.websocket.send(json.dumps({"id": identifier, "method": method, "params": params or {}}))
        while True:
            message = json.loads(await self.websocket.recv())
            if message.get("id") == identifier:
                if "error" in message:
                    raise RuntimeError(f"CDP {method}: {message['error']}")
                return message.get("result", {})

    async def evaluate(self, expression: str, *, await_promise: bool = False) -> Any:
        result = await self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": await_promise},
        )
        remote = result.get("result", {})
        if remote.get("subtype") == "error" or remote.get("type") == "object" and "value" not in remote:
            return None
        return remote.get("value")


async def run() -> int:
    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)
    if not CHROME.is_file():
        raise FileNotFoundError(CHROME)
    profile = EVIDENCE_DIR / "stage4-ui-chrome-profile"
    profile.mkdir(parents=True, exist_ok=True)
    port = free_port()
    process = subprocess.Popen(
        [
            os.fspath(CHROME),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--window-size=1440,1200",
            "http://127.0.0.1:8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        target = await wait_for_target(port)
        websocket_url = str(target["webSocketDebuggerUrl"])
        async with websockets.connect(websocket_url, max_size=None) as websocket:
            cdp = CDP(websocket)
            await cdp.call("Page.enable")
            await cdp.call("DOM.enable")
            await asyncio.sleep(1.5)
            document = await cdp.call("DOM.getDocument", {"depth": -1})
            root_node = int(document["root"]["nodeId"])
            input_node = await cdp.call(
                "DOM.querySelector",
                {"nodeId": root_node, "selector": "input[type=file]"},
            )
            input_id = int(input_node.get("nodeId", 0))
            if not input_id:
                raise RuntimeError("the shipped UI has no audio file input")
            await cdp.call("DOM.setFileInputFiles", {"nodeId": input_id, "files": [os.fspath(REFERENCE)]})
            await asyncio.sleep(0.5)
            previous_job_id = await cdp.evaluate("localStorage.getItem('jianpu-job-id')")
            await cdp.evaluate("document.querySelector('button[type=submit]')?.click()")

            # Do not mistake a completed job remembered from an earlier
            # browser session for the newly submitted file.
            new_job_id = None
            submit_deadline = time.monotonic() + 30
            while time.monotonic() < submit_deadline:
                candidate = await cdp.evaluate("localStorage.getItem('jianpu-job-id')")
                if candidate and candidate != previous_job_id:
                    new_job_id = str(candidate)
                    break
                await asyncio.sleep(0.25)
            if new_job_id is None:
                raise RuntimeError("the UI did not persist a new job id after upload")

            deadline = time.monotonic() + 180
            final: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                final = await cdp.evaluate(
                    "(async()=>{const id=localStorage.getItem('jianpu-job-id');"
                    "if(!id)return {status:'waiting-for-submit'};"
                    "return await fetch('/api/jobs/'+id,{cache:'no-store'}).then(r=>r.json())})()",
                    await_promise=True,
                )
                if isinstance(final, dict) and final.get("status") in {"completed", "failed", "interrupted"}:
                    break
                await asyncio.sleep(1)
            if not isinstance(final, dict) or final.get("status") != "completed":
                raise RuntimeError(f"UI job did not complete: {final}")
            # The browser's own 1.4 second polling loop may have observed the
            # last running phase just before the API became completed.  Wait
            # for React to render the completed score instead of capturing a
            # technically finished job while the panel still says processing.
            ui_text = ""
            ui_deadline = time.monotonic() + 15
            while time.monotonic() < ui_deadline:
                ui_text = str(await cdp.evaluate("document.body.innerText"))
                if "谱页已经展开" in ui_text and "MIDI" in ui_text:
                    break
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError("API job completed but the browser did not render the score panel")
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
            body_text = ui_text
            job_id = await cdp.evaluate("localStorage.getItem('jianpu-job-id')")
            evidence = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "browser": str(CHROME),
                "url": "http://127.0.0.1:8000",
                "reference": str(REFERENCE),
                "job_id": job_id,
                "status": final.get("status"),
                "phase": final.get("phase"),
                "note_count": final.get("summary", {}).get("note_count"),
                "screenshot": str(SCREENSHOT),
                "body_excerpt": str(body_text)[-2500:],
            }
            SUMMARY.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            # The native PowerShell console may use a legacy code page; the
            # UTF-8 evidence file above remains the canonical readable record.
            print(json.dumps(evidence, ensure_ascii=True, indent=2))
    finally:
        try:
            process.terminate()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
