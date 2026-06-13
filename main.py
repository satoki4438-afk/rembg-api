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
    output_bytes = remove(input_bytes, session=session)
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
- 元画像
- rembg背景除去後の透過PNGを市松模様に重ねた確認画像

タスク：
1. rembgが欠落させた前景部分を検出する
   （特に白背景×白パーツで起きやすい）
2. rembgが残してしまった背景部分を検出する
3. 各領域に対して以下を返す：
   - bbox座標（元画像のピクセル座標）
   - 何の部分か（アンテナ/肩/脚/シールド/背景残り等）
   - 確信度（0.0〜1.0）

注意：
- 細い突起（アンテナ・武器）は見逃しやすいので注意する
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

    return JSONResponse(content=json.loads(response.text))


@app.post("/correct-mask")
async def correct_mask(file: UploadFile = File(...)):
    if not REPLICATE_API_TOKEN:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not configured")

    input_bytes = await file.read()
    composite_uri = "data:image/png;base64," + base64.b64encode(input_bytes).decode("utf-8")

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    output = client.run(
        "meta/sam-2",
        input={"image": composite_uri},
    )

    return JSONResponse(content=output)
