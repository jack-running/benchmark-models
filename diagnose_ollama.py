#!/usr/bin/env python3
"""
Quick diagnostic: dumps raw Ollama chunk fields for a failing model.
Run this on your Windows machine:
    python diagnose_ollama.py
    python diagnose_ollama.py --model qwen3.5:27b --prompt "What is 17*23?"
"""
import argparse
import json
import requests

OLLAMA_HOST = "http://192.168.0.149:11434"

def diagnose(host, model, prompt, max_tokens=15, think=None):
    print(f"\n{'='*60}")
    print(f"Model:      {model}")
    print(f"Prompt:     {prompt!r}")
    print(f"max_tokens: {max_tokens}  think param: {think}")
    print(f"{'='*60}")

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0.05},
    }
    if think is not None:
        payload["think"]            = think      # top-level (newer Ollama)
        payload["options"]["think"] = think      # inside options (some builds)

    all_keys     = set()
    full_resp    = ""
    full_think   = ""
    chunk_count  = 0
    shown        = 0

    try:
        with requests.post(f"{host}/api/generate", json=payload,
                           stream=True, timeout=90) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw: continue
                chunk = json.loads(raw)
                chunk_count += 1
                all_keys.update(chunk.keys())

                resp_tok  = chunk.get("response",  "")
                think_tok = chunk.get("thinking",  "")
                full_resp  += resp_tok
                full_think += think_tok

                if shown < 20 and (resp_tok or think_tok or chunk.get("done")):
                    shown += 1
                    nz = {k: v for k, v in chunk.items()
                          if v not in (None, "", False, 0, {}, [])}
                    field_tag = ("[THINK]" if think_tok and not resp_tok
                                 else "[RESP]"  if resp_tok
                                 else "[META]"  if chunk.get("done")
                                 else "[OTHER]")
                    print(f"  chunk {chunk_count:3d} {field_tag}  {nz}")

                if chunk.get("done"):
                    ev  = chunk.get("eval_count", 0)
                    ed  = chunk.get("eval_duration", 1)
                    tps = ev / (ed / 1e9) if ed else 0
                    print(f"\n  ── done: eval_count={ev}  TPS={tps:.1f}")
                    break

    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print(f"\n  All chunk keys seen : {sorted(all_keys)}")
    print(f"  full_response       : {repr(full_resp[:200])}")
    print(f"  full_thinking       : {repr(full_think[:300])}")

    if full_think and not full_resp:
        print("\n  ⚠️  CONFIRMED: model is using thinking mode.")
        print("     All tokens went to 'thinking' field, 'response' is empty.")
        print("     Fix: pass think=False in the API payload.")
    elif full_resp:
        print(f"\n  ✅ Response present: {full_resp!r}")


def get_ollama_version(host):
    try:
        r = requests.get(f"{host}/api/version", timeout=5)
        return r.json().get("version", "unknown")
    except:
        return "could not fetch"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default=OLLAMA_HOST)
    parser.add_argument("--model",  default="gpt-oss:20b")
    parser.add_argument("--prompt", default="What is 17 multiplied by 23? Reply with the number only.")
    args = parser.parse_args()

    host = args.host
    print(f"Ollama version : {get_ollama_version(host)}")

    # Test 1: Default (thinking may be on)
    diagnose(host, args.model, args.prompt, max_tokens=15, think=None)

    # Test 2: Explicitly disable thinking
    diagnose(host, args.model, args.prompt, max_tokens=15, think=False)

    # Test 3: Disable thinking + generous token budget (safety net)
    diagnose(host, args.model, args.prompt, max_tokens=200, think=False)
