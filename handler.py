#!/usr/bin/env python3
"""
RunPod Serverless Handler for Fish Speech 1.5 Voice Cloning
Starts Fish Speech's API server, then proxies requests to it.
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
_server_log_file = None


def download_model():
    """Download Fish Speech 1.5 model weights from HuggingFace if not present."""
    if os.path.isdir(CHECKPOINT_DIR) and len(os.listdir(CHECKPOINT_DIR)) > 3:
        print(f"[STARTUP] Model already present at {CHECKPOINT_DIR}", flush=True)
        print(f"[STARTUP] Files: {os.listdir(CHECKPOINT_DIR)}", flush=True)
        return True

    print("[STARTUP] Downloading Fish Speech 1.5 model (~5GB)...", flush=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "fishaudio/fish-speech-1.5",
            local_dir=CHECKPOINT_DIR,
            ignore_patterns=["*.git*", "*.gitattributes"],
        )
        print("[STARTUP] Model downloaded successfully", flush=True)
        print(f"[STARTUP] Files: {os.listdir(CHECKPOINT_DIR)}", flush=True)
        return True
    except Exception as e:
        print(f"[STARTUP] Model download failed: {e}", flush=True)
        return False


def _has_cuda():
    try:
        import torch
        avail = torch.cuda.is_available()
        if avail:
            print(f"[CUDA] CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
        else:
            print("[CUDA] CUDA not available, using CPU", flush=True)
        return avail
    except Exception as e:
        print(f"[CUDA] torch check failed: {e}", flush=True)
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
                print(f"[SERVER] Server process exited with code {ret}!", flush=True)
                # Read captured log
                if _server_log_file:
                    try:
                        with open(_server_log_file, "r") as f:
                            log = f.read()
                        print(f"[SERVER LOG]\n{log[-3000:]}", flush=True)
                    except Exception:
                        pass
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

        now = time.time()
        if now - last_log >= 30:
            elapsed = int(now - (deadline - timeout))
            remaining = int(deadline - now)
            print(f"[SERVER] Waiting... {elapsed}s elapsed, {remaining}s remaining", flush=True)
            last_log = now

        time.sleep(3)
    # Timeout — read log
    if _server_log_file:
        try:
            with open(_server_log_file, "r") as f:
                log = f.read()
            print(f"[SERVER TIMEOUT LOG]\n{log[-3000:]}", flush=True)
        except Exception:
            pass
    return False


def start_server():
    """Start Fish Speech API server as a background subprocess."""
    global _server_process, _server_ready, _server_log_file

    if _server_ready:
        return True

    # Locate decoder checkpoint file
    decoder_pth = os.path.join(
        CHECKPOINT_DIR, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    )
    if not os.path.exists(decoder_pth):
        for fname in os.listdir(CHECKPOINT_DIR):
            if fname.endswith(".pth"):
                decoder_pth = os.path.join(CHECKPOINT_DIR, fname)
                break

    if not os.path.exists(decoder_pth):
        print(f"[SERVER] No .pth decoder file found in {CHECKPOINT_DIR}!", flush=True)
        print(f"[SERVER] Files present: {os.listdir(CHECKPOINT_DIR)}", flush=True)
        return False

    device = "cuda" if _has_cuda() else "cpu"
    print(f"[SERVER] Starting Fish Speech on {device} | decoder: {decoder_pth}", flush=True)

    # Capture server output to a log file so we can read it on crash
    _server_log_file = "/tmp/fish_speech_server.log"
    log_fh = open(_server_log_file, "w")

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
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[SERVER] Command: {' '.join(cmd)}", flush=True)

    _server_process = subprocess.Popen(
        cmd,
        cwd=FISH_SPEECH_DIR,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    print(f"[SERVER] PID {_server_process.pid}, waiting (up to 8 min)...", flush=True)

    if wait_for_server(timeout=480):
        _server_ready = True
        print("[SERVER] Fish Speech API server is ready!", flush=True)
        return True
    else:
        ret = _server_process.poll()
        print(f"[SERVER] Failed to start (exit code: {ret})", flush=True)
        return False


def get_audio_duration_estimate(audio_bytes, fmt="mp3"):
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
            # Include log snippet in error
            log_snippet = ""
            if _server_log_file:
                try:
                    with open(_server_log_file, "r") as f:
                        log_snippet = f.read()[-500:]
                except Exception:
                    pass
            return {
                "error": "Fish Speech server failed to start",
                "log": log_snippet,
                "success": False,
            }

    input_data = event.get("input", {})
    text = input_data.get("text", "").strip()
    voice_ref_b64 = input_data.get("voice_reference", "").strip()
    reference_text = input_data.get("reference_text", "")
    fmt = input_data.get("format", "mp3")

    if not text:
        return {"error": "Missing text parameter", "success": False}
    if not voice_ref_b64:
        return {"error": "Missing voice_reference parameter", "success": False}

    print(f"[HANDLER] Synthesizing {len(text)} chars | format={fmt}", flush=True)
    gen_start = time.time()

    try:
        audio_bytes = call_tts(text, voice_ref_b64, reference_text, fmt)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"[HANDLER] API error {e.code}: {err_body[:300]}", flush=True)
        return {"error": f"Fish Speech error {e.code}: {err_body[:200]}", "success": False}
    except Exception as e:
        print(f"[HANDLER] TTS call failed: {e}", flush=True)
        return {"error": str(e), "success": False}

    gen_time = round(time.time() - gen_start, 2)
    duration = get_audio_duration_estimate(audio_bytes, fmt)
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    print(f"[HANDLER] Generated {duration}s in {gen_time}s ({len(audio_bytes)//1024}KB)", flush=True)

    return {
        "success": True,
        "audio_base64": audio_b64,
        "duration": duration,
        "generation_time": gen_time,
        "text_length": len(text),
        "model": "fish-speech-1.5",
    }


if __name__ == "__main__":
    print("[STARTUP] Fish Speech 1.5 RunPod Serverless Handler", flush=True)

    if not download_model():
        print("[STARTUP] Cannot proceed without model. Exiting.", flush=True)
        sys.exit(1)

    print("[STARTUP] Starting RunPod serverless worker loop...", flush=True)
    runpod.serverless.start({"handler": handler})
