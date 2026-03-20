"""Quick test to verify Ollama + Qwen3 works for the Pokemon project."""

import json
import time

print("=" * 50)
print("  Ollama + Qwen3 8B Test")
print("=" * 50)

# 1. Test basic connectivity
print("\n1. Testing Ollama connection...")
try:
    from ollama_client import _ollama_chat, _parse_json_response, MODEL
    print(f"   Model: {MODEL}")
except ImportError as e:
    print(f"   FAIL: {e}")
    exit(1)

# 2. Test simple text response
print("\n2. Testing basic response...")
start = time.time()
try:
    resp = _ollama_chat(
        [{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_tokens=20,
    )
    elapsed = time.time() - start
    print(f"   Response: {resp}")
    print(f"   Time: {elapsed:.1f}s")
except Exception as e:
    print(f"   FAIL: {e}")
    print("   Is Ollama running? Try: ollama serve")
    exit(1)

# 3. Test JSON output (critical for gameplay)
print("\n3. Testing JSON output (game action format)...")
start = time.time()
try:
    resp = _ollama_chat(
        [{"role": "user", "content": (
            "You are playing Pokemon FireRed. The player is in Pallet Town at position (5,6). "
            "There is a door to the south. Respond with ONLY valid JSON:\n"
            '{"action": "<button>", "reason": "<why>", "display": "<summary>"}\n'
            "Valid buttons: A, B, Up, Down, Left, Right, Start, Select\n"
            "No other text. Just the JSON."
        )}],
        max_tokens=150,
    )
    elapsed = time.time() - start
    print(f"   Raw: {resp[:200]}")
    parsed = _parse_json_response(resp)
    if parsed and "action" in parsed:
        print(f"   Parsed: {json.dumps(parsed, indent=2)}")
        print(f"   JSON parsing: OK")
    else:
        print(f"   WARNING: Could not parse JSON from response")
        print(f"   This may need prompt tuning")
    print(f"   Time: {elapsed:.1f}s")
except Exception as e:
    print(f"   FAIL: {e}")

# 4. Test navigation intent (used by navigation system)
print("\n4. Testing navigation intent classification...")
start = time.time()
try:
    resp = _ollama_chat(
        [{"role": "user", "content": (
            "You are helping an AI agent play Pokemon FireRed.\n"
            "Choose the SINGLE best immediate overworld navigation intent.\n"
            "Allowed intents: go_to_building, go_to_route_exit, talk_to_npc, interact_with_object, leave_building, train, none.\n\n"
            "Current map: PALLET_TOWN\nMap type: outdoor\n"
            "Current objective: Get a starter Pokemon from Professor Oak's lab\n"
            "Visible doors/exits on map: 3\n\n"
            'Reply ONLY with JSON like {"intent":"go_to_building","reason":"Need to find Oak\'s Lab."}\n'
            "No other text. Just the JSON."
        )}],
        max_tokens=120,
    )
    elapsed = time.time() - start
    parsed = _parse_json_response(resp)
    if parsed and "intent" in parsed:
        print(f"   Intent: {parsed['intent']}")
        print(f"   Reason: {parsed.get('reason', '')}")
        print(f"   Intent classification: OK")
    else:
        print(f"   Raw: {resp[:200]}")
        print(f"   WARNING: Could not parse intent")
    print(f"   Time: {elapsed:.1f}s")
except Exception as e:
    print(f"   FAIL: {e}")

print("\n" + "=" * 50)
print("  Tests complete!")
print("  If all tests passed, run: python ollama_agent.py")
print("=" * 50)
