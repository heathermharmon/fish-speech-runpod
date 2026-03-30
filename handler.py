#!/usr/bin/env python3
"""
RunPod Serverless Handler for Fish Speech 1.5 Voice Cloning
Starts Fish Speech's built-in API server, then proxies requests to it.

Input:
    {
        "input": {
            "text": "Text to synthesize (supports [happy] [sad] [angry] [whisper] tags)",
            "voice_reference": "<base64_encoded_audio>",
            "reference_text": "Optional transcript of reference audio",
            "format": "mp3"  // optional, default mp3
        }
    }

Output:
    {
        "success": true,
        "audio_base64": "<base64_encoded_audio>",
        "duration": 5.23,
        "generation_time": 2.15,
        "model": "fish-speech-1.5"
    }
"""

import os
import sys
import json
import base64
import time
import subprocess
import tempfile
import wave
import struct
import urllib.request
import urllib.error

import runpod

CHECKPOINT_DIR = "/app/checkpoints/fish-speech-1.5"
FISH_SPEECH_DIR = "/app/fish-speech"
API_PORT = 8080
API_BASE = f"http://127.0.0.1:{API_PORT}"

_server_process = None
_server_ready = False


def get_audio_duration(audio_bytes, fmt="mp3"):
    """Estimate audio duration from byte size (fallback)."""
    if fmt == "mp3":
        # ~192kbps = 24000 bytes/sec
        return round(len(audio_bytes) / 24000, 1)
    elif fmt == "wav":
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_bytes)
                tmp = f.name
            with wave.open(tmp, "r") as w:
                dur = w.getnframes() / float(w.getframerate())
            os.unlink(tmp)
            return round(dur, 1)
        except Exception:
            return round(len(audio_bytes) / 88200, 1)
    return 0.0


def wait_for_server(timeout=180):
    """Poll /v1/health until Fish Speech API server is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{API_BASE}/v1/health",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=b"{}",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                body = json.loads(resp.read())
                if body.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_server():
    """Start Fish Speech API server as a background subprocess."""
    global _server_process, _server_ready

    if _server_ready:
        return True

    print("🚀 [HANDLER] Starting Fish Speech 1.5 API server...")

    # Verify checkpoint exists
    if not os.path.isdir(CHECKPOINT_DIR):
        print(f"❌ [HANDLER] Checkpoint not found at {CHECKPOINT_DIR}")
        return False

    # Locate decoder .pth file
    decoder_pth = None
    for fname in os.listdir(CHECKPOINT_DIR):
        if fname.endswith(".pth") and "firefly" in fname.lower():
            decoder_pth = os.path.join(CHECKPOINT_DIR, fname)
            break
    if not decoder_pth:
        # Fall back: let api_server use its default path
        decoder_pth = os.path.join(CHECKPOINT_DIR, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")

    cmd = [
        sys.executable,
        "tools/api_server.py",
        "--mode", "tts",
        "--listen", f"0.0.0.0:{API_PORT}",
        "--llama-checkpoint-path", CHECKPOINT_DIR,
        "--decoder-checkpoint-path", decoder_pth,
        "--device", "cuda" if _has_cuda() else "cpu",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = FISH_SPEECH_DIR + ":" + env.get("PYTHONPATH", "")

    _server_process = subprocess.Popen(
        cmd,
        cwd=FISH_SPEECH_DIR,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    print(f"🔄 [HANDLER] Server PID {_server_process.pid}, waiting for ready...")
    if wait_for_server(timeout=180):
        _server_ready = True
        print("✅ [HANDLER] Fish Speech API server is ready!")
        return True
    else:
        print("❌ [HANDLER] Fish Speech API server failed to start within 180s")
        return False


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def call_tts(text, voice_ref_b64, reference_text="", fmt="mp3"):
    """Call the local Fish Speech API server and return raw audio bytes."""
    payload = {
        "text": text,
        "format": fmt,
        "references": [
            {
                "audio": voice_ref_b64,   # base64 string — schema auto-decodes it
                "text": reference_text or "",
            }
        ],
        "streaming": False,
        "normalize": True,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/v1/tts",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def handler(event):
    global _server_ready

    # Start server on first request (cold start)
    if not _server_ready:
        if not start_server():
            return {"error": "Fish Speech server failed to start", "success": False}

    input_data = event.get("input", {})
    text = input_data.get("text", "").strip()
    voice_ref_b64 = input_data.get("voice_reference", "").strip()
    reference_text = input_data.get("reference_text", "")
    fmt = input_data.get("format", "mp3")

    if not text:
        return {"error": "Missing text parameter", "success": False}
    if not voice_ref_b64:
        return {"error": "Missing voice_reference parameter", "success": False}

    print(f"🎤 [HANDLER] Synthesizing {len(text)} chars | format={fmt}")
    if reference_text:
        print(f"📝 [HANDLER] Reference text: {len(reference_text)} chars")

    gen_start = time.time()

    try:
        audio_bytes = call_tts(text, voice_ref_b64, reference_text, fmt)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"❌ [HANDLER] API error {e.code}: {err_body[:300]}")
        return {"error": f"Fish Speech API error {e.code}: {err_body[:200]}", "success": False}
    except Exception as e:
        print(f"❌ [HANDLER] TTS call failed: {e}")
        return {"error": str(e), "success": False}

    gen_time = round(time.time() - gen_start, 2)
    duration = get_audio_duration(audio_bytes, fmt)

    print(f"✅ [HANDLER] Generated {duration}s audio in {gen_time}s ({len(audio_bytes)//1024}KB)")

    return {
        "success": True,
        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "duration": duration,
        "generation_time": gen_time,
        "text_length": len(text),
        "model": "fish-speech-1.5",
    }


if __name__ == "__main__":
    print("🚀 [HANDLER] RunPod Serverless Handler (Fish Speech 1.5) starting...")
    runpod.serverless.start({"handler": handler})
