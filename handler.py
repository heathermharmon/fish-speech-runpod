#!/usr/bin/env python3
"""
RunPod Serverless Handler for Fish Speech 1.5 Voice Cloning
Mirrors the input/output format of the existing XTTS handler exactly.

Input:
    {
        "input": {
            "text": "Text to synthesize",
            "voice_reference": "<base64_encoded_wav>",
            "reference_text": "Optional transcript of the reference audio",
            "language": "en"
        }
    }

Output:
    {
        "success": true,
        "audio_base64": "<base64_encoded_wav>",
        "duration": 5.23,
        "generation_time": 2.15,
        "text_length": 123,
        "device": "cuda",
        "model": "fish-speech-1.5"
    }
"""

import os
import sys
import runpod
import tempfile
import base64
import time
import subprocess
import wave
import torch

# ── Device detection ─────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎮 [HANDLER] Using device: {device}")

CHECKPOINT_DIR = "/app/checkpoints/fish-speech-1.5"
FISH_SPEECH_DIR = "/app/fish-speech"

# ── Verify checkpoint exists ──────────────────────────────────────────────────
print("🚀 [HANDLER] Verifying Fish Speech 1.5 model...")
if not os.path.isdir(CHECKPOINT_DIR):
    print(f"❌ [HANDLER] Checkpoint not found at {CHECKPOINT_DIR}")
    sys.exit(1)
print(f"✅ [HANDLER] Model found at {CHECKPOINT_DIR}")


def get_wav_duration(wav_path):
    """Return duration in seconds for a WAV file."""
    try:
        with wave.open(wav_path, 'r') as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def handler(event):
    """
    RunPod Serverless handler — mirrors XTTS handler signature exactly.
    """
    try:
        input_data = event.get('input', {})

        text           = input_data.get('text', '').strip()
        voice_ref_b64  = input_data.get('voice_reference', '').strip()
        reference_text = input_data.get('reference_text', '').strip()
        language       = input_data.get('language', 'en')

        # ── Validation ────────────────────────────────────────────────────────
        if not text:
            return {'error': 'Missing text parameter', 'success': False}
        if not voice_ref_b64:
            return {'error': 'Missing voice_reference parameter', 'success': False}

        print(f"🎤 [HANDLER] Synthesizing {len(text)} chars, language={language}")
        if reference_text:
            print(f"📝 [HANDLER] Reference text provided ({len(reference_text)} chars)")

        # ── Decode reference audio ────────────────────────────────────────────
        voice_ref_data = base64.b64decode(voice_ref_b64)

        ref_file   = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        ref_path   = ref_file.name
        ref_file.write(voice_ref_data)
        ref_file.close()

        output_path = tempfile.mktemp(suffix='.wav')

        # ── Fish Speech inference (subprocess via tools/inference.py) ─────────
        # tools/inference.py is Fish Speech 1.5's official end-to-end script.
        # It handles: encode reference → generate tokens → decode audio.
        gen_start = time.time()

        cmd = [
            sys.executable,
            os.path.join(FISH_SPEECH_DIR, 'tools', 'inference.py'),
            '--text',             text,
            '--reference-audio',  ref_path,
            '--output-path',      output_path,
            '--checkpoint-path',  CHECKPOINT_DIR,
            '--device',           device,
        ]

        # reference_text improves clone accuracy significantly when available
        if reference_text:
            cmd += ['--reference-text', reference_text]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=FISH_SPEECH_DIR
        )

        gen_time = time.time() - gen_start

        if result.returncode != 0:
            print(f"❌ [HANDLER] Fish Speech stderr:\n{result.stderr}")
            return {
                'error': f'Fish Speech inference failed: {result.stderr[-500:]}',
                'success': False
            }

        if not os.path.exists(output_path):
            return {'error': 'Output file not created', 'success': False}

        # ── Read output and encode ────────────────────────────────────────────
        duration = get_wav_duration(output_path)
        print(f"✅ [HANDLER] Generated {duration:.2f}s audio in {gen_time:.2f}s")

        with open(output_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')

        # ── Cleanup ───────────────────────────────────────────────────────────
        os.unlink(ref_path)
        os.unlink(output_path)

        return {
            'success':         True,
            'audio_base64':    audio_b64,
            'duration':        duration,
            'generation_time': gen_time,
            'text_length':     len(text),
            'device':          device,
            'model':           'fish-speech-1.5'
        }

    except Exception as e:
        print(f"❌ [HANDLER] Unhandled exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e), 'success': False}


if __name__ == '__main__':
    print("🚀 [HANDLER] Starting RunPod Serverless Handler (Fish Speech 1.5)...")
    runpod.serverless.start({"handler": handler})
