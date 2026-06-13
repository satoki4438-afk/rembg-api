import os
import io
import json
import base64
import colorsys
import numpy as np
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image, ImageDraw, ImageFont
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


SELECT_PROMPT = """あなたはプラモデル画像のセグメント選別AIです。

入力：
- 元画像（1枚目）
- 元画像にSAM2で生成したセグメントを半透明色＋番号でオーバーレイした画像（2枚目）

タスク：
2枚目に表示されている番号付きセグメントそれぞれについて、
プラモデル本体（パーツ・武器・台座を含む可動部）に属するか、
背景（机・壁・床・影・反射等）に属するかを判定する。

本体に属すると判断した番号を全てリストで返す。

注意：
- 透明・半透明なクリアパーツも本体として扱う
- 影や反射は背景として扱う
- 番号が読み取れない・対象が小さすぎて判断できない場合は除外してよい
"""

SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "body_segment_ids": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["body_segment_ids"],
}

OVERLAY_MIN_AREA_RATIO = 0.005
OVERLAY_MAX_AREA_RATIO = 0.95
OVERLAY_MAX_SEGMENTS = 40
OVERLAY_ALPHA = 115


def build_segment_overlay(original_bytes: bytes, mask_urls: list):
    original = Image.open(io.BytesIO(original_bytes)).convert("RGBA")
    w, h = original.size
    total = w * h

    candidates = []
    for url in mask_urls:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        mask_img = Image.open(io.BytesIO(resp.content)).convert("L").resize((w, h))
        mask_np = np.array(mask_img) > 127
        area = int(mask_np.sum())
        if area == 0:
            continue
        ratio = area / total
        if ratio < OVERLAY_MIN_AREA_RATIO or ratio > OVERLAY_MAX_AREA_RATIO:
            continue
        ys, xs = np.where(mask_np)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        candidates.append((area, url, mask_np, bbox))

    candidates.sort(key=lambda c: c[0], reverse=True)
    candidates = candidates[:OVERLAY_MAX_SEGMENTS]

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
    except Exception:
        try:
            font = ImageFont.load_default(size=32)
        except TypeError:
            font = ImageFont.load_default()

    overlay = original.copy()
    filtered_urls = []
    for i, (area, url, mask_np, bbox) in enumerate(candidates, start=1):
        hue = ((i - 1) * 0.37) % 1.0
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.85, 0.95)]
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        color_rgba = Image.new("RGBA", (w, h), (r, g, b, OVERLAY_ALPHA))
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        layer.paste(color_rgba, (0, 0), mask_img)
        overlay = Image.alpha_composite(overlay, layer)
        filtered_urls.append(url)

    draw = ImageDraw.Draw(overlay, "RGBA")
    for i, (area, url, mask_np, bbox) in enumerate(candidates, start=1):
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        draw.text((cx, cy), str(i), font=font, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255), anchor="mm")

    out = io.BytesIO()
    overlay.convert("RGB").save(out, format="PNG")
    return out.getvalue(), filtered_urls


@app.post("/segment-select")
async def segment_select(file: UploadFile = File(...)):
    if not REPLICATE_API_TOKEN:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not configured")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    original_bytes = await file.read()
    composite_uri = "data:image/png;base64," + base64.b64encode(original_bytes).decode("utf-8")

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    try:
        sam_output = client.run(
            "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
            input={"image": composite_uri},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    sam_output = serialize_replicate_output(sam_output)
    mask_urls = sam_output.get("individual_masks", [])
    if not mask_urls:
        return JSONResponse(content={"masks": [], "selected_indices": []})

    try:
        overlay_bytes, filtered_urls = build_segment_overlay(original_bytes, mask_urls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    try:
        response = model.generate_content(
            [
                SELECT_PROMPT,
                {"mime_type": "image/png", "data": original_bytes},
                {"mime_type": "image/png", "data": overlay_bytes},
            ],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=SELECT_SCHEMA,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    result = json.loads(response.text)
    return JSONResponse(content={"masks": filtered_urls, "selected_indices": result.get("body_segment_ids", [])})
