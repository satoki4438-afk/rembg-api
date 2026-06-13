import os
import io
import json
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image
import google.generativeai as genai
import replicate

app = FastAPI()

session = new_session("isnet-general-use")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


@app.get("/")
def health():
    return {"ok": True}


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    input_bytes = await file.read()
    try:
        output_bytes = remove(input_bytes, session=session)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return Response(content=output_bytes, media_type="image/png")


CHECKER_TILE = 16


def make_checkerboard_composite(png_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size
    checker = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    dark_tile = Image.new("RGBA", (CHECKER_TILE, CHECKER_TILE), (204, 204, 204, 255))
    for y in range(0, h, CHECKER_TILE):
        for x in range(0, w, CHECKER_TILE):
            if ((x // CHECKER_TILE) + (y // CHECKER_TILE)) % 2 == 1:
                checker.paste(dark_tile, (x, y))
    composite = Image.alpha_composite(checker, img)
    out = io.BytesIO()
    composite.convert("RGB").save(out, format="PNG")
    return out.getvalue()


AUDIT_PROMPT = """あなたはプラモデル画像の背景除去監査AIです。

入力：
- 元画像（1枚目）
- rembg背景除去後の透過PNGを市松模様に重ねた確認画像（2枚目）

タスク：
1. 1枚目と2枚目を比較し、1枚目では本体（プラモデル）だった領域が
   2枚目で市松模様（透明）になっている部分を全て検出する
   - アンテナ・武器のような細い突起だけでなく、
     腕・脚・足のようなパーツ全体が丸ごと透明化しているケースも
     必ず検出すること（特に白背景×白パーツで起きやすい）
2. 2枚目で市松模様にならず、1枚目の背景がそのまま残っている領域を検出する
3. 各領域に対して以下を返す：
   - bbox座標（元画像のピクセル座標）
   - 何の部分か（アンテナ/肩/脚/シールド/背景残り等）
   - 確信度（0.0〜1.0）

注意：
- 細い突起（アンテナ・武器）も、腕や脚など大きなパーツ全体の消失も
  どちらも見逃さないこと
- 背景と同色のパーツは特に慎重に判定する
- 影や反射は前景として扱う
- 確信度が低い場合は省略してよい
"""

AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "missing_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                        },
                        "required": ["x", "y", "w", "h"],
                    },
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["bbox", "description", "confidence"],
            },
        },
        "excess_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                        },
                        "required": ["x", "y", "w", "h"],
                    },
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["bbox", "description", "confidence"],
            },
        },
    },
    "required": ["missing_regions", "excess_regions"],
}


@app.post("/audit-mask")
async def audit_mask(original: UploadFile = File(...), transparent: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    original_bytes = await original.read()
    transparent_bytes = await transparent.read()
    checkerboard_bytes = make_checkerboard_composite(transparent_bytes)

    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    try:
        response = model.generate_content(
            [
                AUDIT_PROMPT,
                {"mime_type": "image/png", "data": original_bytes},
                {"mime_type": "image/png", "data": checkerboard_bytes},
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=AUDIT_SCHEMA,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return JSONResponse(content=json.loads(response.text))


def serialize_replicate_output(value):
    if isinstance(value, dict):
        return {k: serialize_replicate_output(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_replicate_output(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@app.post("/correct-mask")
async def correct_mask(file: UploadFile = File(...)):
    if not REPLICATE_API_TOKEN:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not configured")

    input_bytes = await file.read()
    composite_uri = "data:image/png;base64," + base64.b64encode(input_bytes).decode("utf-8")

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    try:
        output = client.run(
            "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
            input={"image": composite_uri},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return JSONResponse(content=serialize_replicate_output(output))
