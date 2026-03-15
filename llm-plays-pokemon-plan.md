# LLM Plays Pokemon — Project Plan

## Project Overview
An autonomous AI agent that plays Pokemon FireRed by reading the game screen and memory, deciding inputs via Claude API, and optionally streaming on Twitch. Uses a hybrid vision + memory approach with tiered model routing (Haiku for navigation, Sonnet for battles) to balance cost and performance.

---

## Tech Stack
- **Emulator:** BizHawk (GBA) — exposes Lua socket bridge for Python control
- **Game:** Pokemon FireRed (GBA ROM)
- **Vision:** PIL / mss for screenshots, sent as base64 to Claude
- **LLM:** Anthropic API — `claude-haiku-4-5-20251001` for navigation, `claude-sonnet-4-6` for battles
- **Input injection:** BizHawk socket bridge (preferred over pyautogui)
- **Overlay:** Local HTML page updated via websockets, captured by OBS browser source
- **Logging:** SQLite via Python `sqlite3`
- **Environment:** Python 3.11+, venv, `.env` for API key

---

## File Structure
```
llm-plays-pokemon/
├── emulator.py        # BizHawk socket wrapper (screenshot, press_button, read_memory)
├── memory.py          # FireRed memory address map (HP, position, battle flag, etc.)
├── agent.py           # Main game loop
├── claude_client.py   # Anthropic API calls, prompt caching, model routing
├── anti_stuck.py      # Loop detection, position watchdog, recovery logic
├── progress.py        # Milestone tracking, rolling summaries, progress JSON
├── overlay/
│   └── index.html     # OBS browser source overlay (action, reasoning, cost)
└── logs/
    └── runs.db        # SQLite log of every action, token count, cost
```

---

## Phase 1 — Environment Setup (Day 1–2)
**Goal:** Emulator running and fully controllable from Python.

### Tasks
- [ ] Install BizHawk, load FireRed ROM
- [ ] Enable BizHawk Lua HTTP socket bridge
- [ ] Write `emulator.py` with three core methods:
  - `screenshot()` → returns PIL image
  - `press_button(btn)` → sends keypress via socket
  - `read_memory(addr)` → returns value at memory address
- [ ] Verify end-to-end: Python captures frame → sends "A" press → screen changes

### Memory Addresses to Map (`memory.py`)
| Field | Address | Notes |
|-------|---------|-------|
| Player X position | 0x02036E2C | |
| Player Y position | 0x02036E28 | |
| Current map ID | 0x02036DFC | |
| Battle state flag | 0x02022B50 | 0 = overworld, 1 = in battle |
| Player active Pokemon HP | 0x02024284 | |
| Player active Pokemon level | 0x02024285 | |
| Enemy Pokemon species | 0x0202402C | |
| Enemy Pokemon HP | 0x02024050 | |

---

## Phase 2 — Core Game Loop (Day 3–5)
**Goal:** LLM makes real decisions and the game advances.

### Tasks
- [ ] Build `claude_client.py`:
  - Encode screenshot as base64
  - Build prompt with game state + history
  - Cache static system prompt using Anthropic prompt caching
  - Parse JSON response for action + reason
- [ ] Build `agent.py` main loop:
  - Read memory state
  - Capture screenshot
  - Route to Haiku (overworld) or Sonnet (battle) based on battle flag
  - Call Claude, extract action, press button
  - Log action + reasoning + tokens to SQLite
  - Sleep 15 seconds (4 calls/min)
- [ ] Write and tune system prompt (see below)

### System Prompt Structure
```
You are an AI playing Pokemon FireRed. Your goal is to complete the game — earn all 8 badges and defeat the Elite Four.

CONTROLS: A, B, Up, Down, Left, Right, Start, Select

CURRENT PROGRESS: {progress_summary}
CURRENT STATE: {memory_state_json}
RECENT ACTIONS: {last_10_actions}

ANTI-LOOP RULE: If you have repeated the same action 5+ times, you MUST try a different button.

Respond ONLY with valid JSON: {"action": "A", "reason": "brief explanation"}
```

### Model Routing Logic
```python
model = "claude-haiku-4-5-20251001" if not state["in_battle"] else "claude-sonnet-4-6"
```

---

## Phase 3 — Anti-Stuck Systems (Day 6–7)
**Goal:** Detect and recover from loops and stuck states automatically.

### Tasks
- [ ] **Repetition detector** (`anti_stuck.py`):
  - If same action repeats 10+ times consecutively → inject warning into next prompt
  - Temporarily force Sonnet for next 5 calls when triggered
- [ ] **Position watchdog**:
  - If X/Y coordinates unchanged for 2+ minutes → flag as stuck
  - Inject "you appear to be physically stuck, try a different direction" into prompt
- [ ] **Progress tracker** (`progress.py`):
  - JSON file tracking: badges earned, maps visited, key story flags
  - One-line summary fed into every prompt: e.g. "Progress: 1 badge, currently in Mt. Moon"
- [ ] **Rolling summary**:
  - Every 50 actions, ask Claude to write a 2-sentence summary of what happened
  - Store summary and use instead of raw action history to keep context small

---

## Phase 4 — Streaming Layer (Day 8–10)
**Goal:** Make it watchable and stream to Twitch.

### Tasks
- [ ] Build `overlay/index.html`:
  - Connects to local websocket served by `agent.py`
  - Displays: last action, Claude's reasoning, model used (Haiku/Sonnet), running cost, progress summary
  - Styled to look good as an OBS browser source overlay
- [ ] OBS setup:
  - Capture BizHawk window as game source
  - Add browser source pointing to `localhost:8765` for overlay
  - Configure Twitch stream key
- [ ] Add replay buffer in OBS for auto-clipping level-ups and gym battles

---

## Phase 5 — Hardening & Cost Monitoring (Ongoing)
**Goal:** Keep costs visible and the system reliable for long runs.

### Tasks
- [ ] Log every API call to SQLite: timestamp, model, input tokens, output tokens, cost, action, reason
- [ ] Print hourly cost report to terminal
- [ ] Build frame diffing: skip API call if current frame is >95% similar to last (saves ~30% of calls during animations/walking)
- [ ] Tune Haiku/Sonnet routing — test if Haiku alone can handle early-game battles
- [ ] Add graceful shutdown: on Ctrl+C, save full state and progress to disk

---

## Cost Model
| Scenario | Model split | Est. cost per hour | Est. full run (100–200 hrs) |
|----------|-------------|-------------------|----------------------------|
| Conservative | 100% Haiku | ~$0.15/hr | $15–30 |
| Recommended | 70% Haiku / 30% Sonnet | ~$0.45/hr | $45–90 |
| Quality-first | 100% Sonnet | ~$0.90/hr | $90–180 |

Prompt caching on the static system prompt reduces input costs by ~60% on every call.

---

## Key Decisions & Rationale
- **FireRed over Red/Blue** — GBA has better-documented memory addresses and BizHawk support
- **BizHawk over mGBA** — exposes a socket API for clean Python integration without window focus hacks
- **Hybrid vision + memory** — vision handles menus/dialogue/maps; memory gives precise battle stats without relying on OCR
- **Haiku/Sonnet routing** — battles need more reasoning; navigation is mostly pattern matching Haiku handles well
- **15-second cadence** — gives game time to process animations; slow enough to be cost-efficient, fast enough to be watchable
- **Prompt caching** — system prompt is static across all calls, making it a perfect cache candidate

---

## Setup Commands
```bash
# 1. Scaffold project
mkdir llm-plays-pokemon && cd llm-plays-pokemon
python3 -m venv venv && source venv/bin/activate
pip install anthropic pillow mss pyautogui requests websockets

# 2. Create file structure
mkdir -p overlay logs
touch emulator.py memory.py agent.py claude_client.py anti_stuck.py progress.py overlay/index.html

# 3. Environment
echo "ANTHROPIC_API_KEY=your_key_here" > .env
echo "venv/\n.env\nlogs/" > .gitignore
git init

# 4. Test BizHawk connection (after BizHawk is installed and ROM loaded)
python3 -c "import socket; s = socket.socket(); s.connect(('localhost', 9999)); print('BizHawk connected')"
```

---

## Build Order (Recommended)
1. `emulator.py` — nothing works without this
2. `memory.py` — define all address constants
3. `claude_client.py` — API wrapper with caching
4. `agent.py` — wire everything together
5. `anti_stuck.py` — add after basic loop is working
6. `progress.py` — add after anti-stuck is stable
7. `overlay/index.html` — last, once core loop is solid
