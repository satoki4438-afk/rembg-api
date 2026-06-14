import os
import io
import json
import base64
import tempfile
import numpy as np
import requests
import imageio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import replicate
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SAM2_VIDEO_MODEL = "meta/sam-2-video:33432afdfc06a10da6b4018932893d39b0159f838b6d11dd1236dff85cc5ec1d"

AUDIT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "partial", "failed"]},
        "foreground_coverage": {"type": "number"},
        "message": {"type": "string"},
    },
    "required": ["status", "foreground_coverage", "message"],
}

AUDIT_PROMPT = """1枚目はプラモデル/フィギュアの元画像、2枚目はAIで背景を切り抜いた結果(透過PNG)です。

切り抜きが成功しているか判定してください。
- ok: モデル全体がほぼ完全に切り抜けている
- partial: 一部のパーツ(足だけ、上半身だけなど)しか切り抜けていない
- failed: ほとんど何も切り抜けていない、または元画像と無関係なものが切り抜かれている

foreground_coverageは、元画像に対して切り抜けたモデルの割合(0.0〜1.0)です。
messageには判定理由を日本語で簡潔に書いてください。
JSON形式で返してください。"""


@app.get("/")
def health():
    return {"ok": True}


def image_to_single_frame_video(img: Image.Image) -> bytes:
    rgb = img.convert("RGB")
    w, h = rgb.size
    new_w = w - (w % 2)
    new_h = h - (h % 2)
    if (new_w, new_h) != (w, h):
        rgb = rgb.crop((0, 0, new_w, new_h))

    frame = np.array(rgb)

    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        imageio.mimwrite(path, [frame], fps=1, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
        with open(path, "rb") as f:
            return f.read(), new_w, new_h
    finally:
        os.unlink(path)


def serialize_replicate_output(value):
    if isinstance(value, dict):
        return {k: serialize_replicate_output(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_replicate_output(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def find_image_urls(value, found):
    if isinstance(value, str) and value.startswith("http") and any(
        value.lower().split("?")[0].endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif")
    ):
        found.append(value)
    elif isinstance(value, dict):
        for v in value.values():
            find_image_urls(v, found)
    elif isinstance(value, list):
        for v in value:
            find_image_urls(v, found)


@app.post("/sam2-click")
async def sam2_click(file: UploadFile = File(...), points: str = Form(...)):
    if not REPLICATE_API_TOKEN:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not configured")

    try:
        point_list = json.loads(points)
        if not point_list:
            raise ValueError("points is empty")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid points: {e}")

    input_bytes = await file.read()
    img = Image.open(io.BytesIO(input_bytes))

    try:
        video_bytes, w, h = image_to_single_frame_video(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"video encode failed: {type(e).__name__}: {e}")

    video_uri = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode("utf-8")

    coords_str = ",".join(f"[{int(x)},{int(y)}]" for x, y in point_list)
    labels_str = ",".join("1" for _ in point_list)
    frames_str = ",".join("0" for _ in point_list)
    object_ids_str = ",".join("model" for _ in point_list)

    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    try:
        output = client.run(
            SAM2_VIDEO_MODEL,
            input={
                "input_video": video_uri,
                "click_coordinates": coords_str,
                "click_labels": labels_str,
                "click_frames": frames_str,
                "click_object_ids": object_ids_str,
                "mask_type": "binary",
                "annotation_type": "mask",
                "output_video": False,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    if not isinstance(output, (str, bytes, dict, list)) and hasattr(output, "__iter__"):
        output = list(output)

    serialized = serialize_replicate_output(output)

    urls = []
    find_image_urls(serialized, urls)
    if not urls:
        return JSONResponse(content={"error": "no mask output", "raw": serialized}, status_code=500)

    mask_resp = requests.get(urls[0], timeout=30)
    mask_resp.raise_for_status()
    mask_img = Image.open(io.BytesIO(mask_resp.content)).convert("L").resize((w, h))
    mask_np = np.array(mask_img) > 127

    rgb = img.convert("RGB")
    new_w_h = (w, h)
    if rgb.size != new_w_h:
        rgb = rgb.crop((0, 0, w, h))
    result_np = np.dstack([np.array(rgb), np.where(mask_np, 255, 0).astype(np.uint8)])
    out_img = Image.fromarray(result_np, "RGBA")
    out_buf = io.BytesIO()
    out_img.save(out_buf, format="PNG")

    return JSONResponse(content={
        "png_base64": base64.b64encode(out_buf.getvalue()).decode("utf-8"),
        "width": w,
        "height": h,
        "debug": {"mask_url": urls[0], "raw_type": type(serialized).__name__, "all_urls": urls, "raw": serialized},
    })


@app.post("/audit-cutout")
async def audit_cutout(file: UploadFile = File(...), cutout_base64: str = Form(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    original_bytes = await file.read()
    original_img = Image.open(io.BytesIO(original_bytes)).convert("RGB")

    try:
        cutout_bytes = base64.b64decode(cutout_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid cutout_base64: {e}")
    cutout_img = Image.open(io.BytesIO(cutout_bytes))

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": AUDIT_RESPONSE_SCHEMA,
        },
    )

    try:
        response = model.generate_content([AUDIT_PROMPT, original_img, cutout_img])
        result = json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return JSONResponse(content=result)
