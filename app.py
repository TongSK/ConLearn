"""
Prompt injection detector API and web interface.

Run:
  python app.py --artifact detector_artifact.pt --host 0.0.0.0 --port 8000

Then open:
  http://localhost:8000
"""

import argparse

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from inference import PromptInjectionDetector


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt Injection Detector</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #18202a;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 28px;
    }
    main {
      width: min(980px, 100%);
      display: grid;
      gap: 18px;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      border-bottom: 1px solid #d9dde4;
      padding-bottom: 16px;
    }
    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: 0;
    }
    .status {
      font-size: 14px;
      color: #52606f;
      text-align: right;
    }
    textarea {
      width: 100%;
      box-sizing: border-box;
      min-height: 220px;
      resize: vertical;
      border: 1px solid #c8ced8;
      border-radius: 8px;
      padding: 16px;
      font: inherit;
      line-height: 1.5;
      background: #ffffff;
      color: #18202a;
      outline: none;
    }
    textarea:focus {
      border-color: #3267d6;
      box-shadow: 0 0 0 3px rgba(50, 103, 214, 0.14);
    }
    .toolbar {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    button {
      border: 0;
      border-radius: 8px;
      background: #1f5fbf;
      color: white;
      font-weight: 700;
      padding: 12px 18px;
      cursor: pointer;
    }
    button:disabled {
      background: #9aa6b5;
      cursor: wait;
    }
    .result {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .tile {
      background: #ffffff;
      border: 1px solid #d9dde4;
      border-radius: 8px;
      padding: 14px;
      min-height: 74px;
    }
    .tile span {
      display: block;
      color: #52606f;
      font-size: 13px;
      margin-bottom: 8px;
    }
    .tile strong {
      font-size: 18px;
      overflow-wrap: anywhere;
    }
    .danger strong { color: #b42318; }
    .safe strong { color: #067647; }
    .review strong { color: #b54708; }
    .error {
      color: #b42318;
      min-height: 22px;
    }
    @media (max-width: 760px) {
      header { align-items: start; flex-direction: column; }
      .status { text-align: left; }
      .result { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Prompt Injection Detector</h1>
      <div class="status">Contrastive RoBERTa prototype</div>
    </header>

    <textarea id="prompt" placeholder="Paste a user prompt here..."></textarea>

    <div class="toolbar">
      <button id="detect">Detect</button>
      <div id="error" class="error"></div>
    </div>

    <section class="result">
      <div id="labelTile" class="tile">
        <span>Prediction</span>
        <strong id="label">Waiting</strong>
      </div>
      <div id="actionTile" class="tile">
        <span>Recommended action</span>
        <strong id="action">-</strong>
      </div>
      <div class="tile">
        <span>Risk probability</span>
        <strong id="risk">-</strong>
      </div>
      <div class="tile">
        <span>Raw score</span>
        <strong id="score">-</strong>
      </div>
    </section>
  </main>

  <script>
    const button = document.getElementById("detect");
    const promptBox = document.getElementById("prompt");
    const errorBox = document.getElementById("error");
    const label = document.getElementById("label");
    const action = document.getElementById("action");
    const risk = document.getElementById("risk");
    const score = document.getElementById("score");
    const labelTile = document.getElementById("labelTile");
    const actionTile = document.getElementById("actionTile");

    function setTileClasses(result) {
      labelTile.className = "tile " + (result.is_prompt_injection ? "danger" : "safe");
      actionTile.className = "tile " + (result.action === "review" ? "review" : result.action === "block" ? "danger" : "safe");
    }

    button.addEventListener("click", async () => {
      const text = promptBox.value.trim();
      errorBox.textContent = "";
      if (!text) {
        errorBox.textContent = "Enter a prompt first.";
        return;
      }

      button.disabled = true;
      button.textContent = "Checking...";

      try {
        const response = await fetch("/detect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const result = await response.json();
        setTileClasses(result);
        label.textContent = result.label.replace("_", " ");
        action.textContent = result.action;
        risk.textContent = (result.risk_probability * 100).toFixed(1) + "%";
        score.textContent = result.score.toFixed(4);
      } catch (error) {
        errorBox.textContent = error.message || "Detection failed.";
      } finally {
        button.disabled = false;
        button.textContent = "Detect";
      }
    });
  </script>
</body>
</html>
"""


class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=1)


def create_app(artifact_path):
    detector = PromptInjectionDetector(artifact_path)
    app = FastAPI(title="Prompt Injection Detector")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTML_PAGE

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model_name": detector.model_name,
            "device": str(detector.device),
            "threshold": detector.threshold,
        }

    @app.post("/detect")
    def detect(request: DetectionRequest):
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text must not be empty.")
        return detector.predict(text)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="detector_artifact.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.artifact)
    uvicorn.run(app, host=args.host, port=args.port)
