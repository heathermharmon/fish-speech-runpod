#!/usr/bin/env python3
"""
RunPod Serverless Handler for Fish Speech 1.5 Voice Cloning
Starts Fish Speech's API server, then proxies requests to it.

Input:
    {
        "input": {
            "text": "Text (supports [happy] [sad] [angry] [whisper] [cry] tags)",
            "voice_reference": "<base64_encoded_audio>",
            "reference_text": "Optional transcript of reference audio",
            "format": "mp3"
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
import urllib.request
import urllib.error

import runpod

CHECKPOINT_DIR = "/app/checkpoints/fish-speech-1.5"
FISH_SPEECH_DIR = "/app/fish-speech"
API_PORT = 8080
API_BASE = f"http://127.0.0.1:{API_PORT}"

_server_process = None
_server_ready = False


def download_model():
    """Download Fish Speech 1.5 model weights from HuggingFace if not present."""
    if os.path.isdir(CHECKPOINT_DIR) and len(os.listdir(CHECKPOINT_DIR)) > 3:
        print(f"[STARTUP] Model already present at {CHECKPOINT_DIR}")
        return True

    print("[STARTUP] Downloading Fish Speech 1.5 model (~5GB)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "fishaudio/fish-speech-1.5",
            local_dir=CHECKPOINT_DIR,
            ignore_patterns=["*.git*", "*.gitattributes"],
        )
        print("[STARTUP] Model downloaded successfully")
        return True
    except Exception as e:
        print(f"[STARTUP] Model download failed: {e}")
        return False


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def wait_for_server(timeout=480):
    """Poll /v1/health until Fish Speech API server is ready."""
    deadline = time.time() + timeout
    last_log = time.time()
    while time.time() < deadline:
        # Check if server process died
        if _server_process is not None:
            ret = _server_process.poll()
            if ret is not None:
                print(f"[SERVER] Server process exited with code {ret}!")
                return False

        try:
            req = urllib.request.Request(
                f"{API_BASE}/v1/health",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=b"{}",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                if body.get("status") == "ok":
                    return True
        except Exception:
            pass

        # Log progress every 30 seconds
        now = time.time()
        if now - last_log >= 30:
            elapsed = int(now - (deadline - timeout))
            remaining = int(deadline - now)
            print(f"[SERVER] Still waiting for server... {elapsed}s elapsed, {remaining}s remaining")
            last_log = now

        time.sleep(3)
    return False


def start_server():
    """Start Fish Speech API server as a background subprocess."""
    global _server_process, _server_ready

    if _server_ready:
        return True

    # Locate decoder checkpoint file
    decoder_pth = os.path.join(
        CHECKPOINT_DIR, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    )
    if not os.path.exists(decoder_pth):
        # Try to find any .pth file
        for fname in os.listdir(CHECKPOINT_DIR):
            if fname.endswith(".pth"):
                decoder_pth = os.path.join(CHECKPOINT_DIR, fname)
                break

    if not os.path.exists(decoder_pth):
        print(f"[SERVER] No decoder checkpoint found in {CHECKPOINT_DIR}!")
        print(f"[SERVER] Files present: {os.listdir(CHECKPOINT_DIR)}")
        return False

    device = "cuda" if _has_cuda() else "cpu"
    print(f"[SERVER] Starting Fish Speech API server on {device}...")
    print(f"[SERVER] Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"[SERVER] Decoder path: {decoder_pth}")
    print(f"[SERVER] Files in checkpoint dir: {os.listdir(CHECKPOINT_DIR)}")

    cmd = [
        sys.executable,
        "tools/api_server.py",
        "--mode", "tts",
        "--listen", f"0.0.0.0:{API_PORT}",
        "--llama-checkpoint-path", CHECKPOINT_DIR,
        "--decoder-checkpoint-path", decoder_pth,
        "--device", device,
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

    print(f"[SERVER] PID {_server_process.pid}, waiting for ready (up to 8 min)...")

    if wait_for_server(timeout=480):
        _server_ready = True
        print("[SERVER] Fish Speech API server is ready!")
        return True
    else:
        ret = _server_process.poll()
        print(f"[SERVER] Fish Speech API server failed to start (process exit code: {ret})")
        return False


def get_audio_duration_estimate(audio_bytes, fmt="mp3"):
    """Estimate audio duration from file size."""
    if fmt == "mp3":
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


def call_tts(text, voice_ref_b64, reference_text="", fmt="mp3"):
    """POST to local Fish Speech API server, return raw audio bytes."""
    payload = {
        "text": text,
        "format": fmt,
        "references": [
            {
                "audio": voice_ref_b64,
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

    print(f"[HANDLER] Synthesizing {len(text)} chars | format={fmt}")

    gen_start = time.time()

    try:
        audio_bytes = call_tts(text, voice_ref_b64, reference_text, fmt)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"[HANDLER] API error {e.code}: {err_body[:300]}")
        return {"error": f"Fish Speech error {e.code}: {err_body[:200]}", "success": False}
    except Exception as e:
        print(f"[HANDLER] TTS call failed: {e}")
        return {"error": str(e), "success": False}

    gen_time = round(time.time() - gen_start, 2)
    duration = get_audio_duration_estimate(audio_bytes, fmt)
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    print(f"[HANDLER] Generated {duration}s in {gen_time}s ({len(audio_bytes)//1024}KB)")

    return {
        "success": True,
        "audio_base64": audio_b64,
        "duration": duration,
        "generation_time": gen_time,
        "text_length": len(text),
        "model": "fish-speech-1.5",
    }


if __name__ == "__main__":
    print("[STARTUP] Fish Speech 1.5 RunPod Serverless Handler")

    # Download model at startup (before accepting jobs)
    if not download_model():
        print("[STARTUP] Cannot proceed without model. Exiting.")
        sys.exit(1)

    print("[STARTUP] Starting RunPod serverless worker loop...")
    runpod.serverless.start({"handler": handler})
