# LLM Plays Pokemon - Quick Start

## Startup Order (any order works now)

1. **Start the Python agent:**
   Double-click `start_agent.bat` — it will wait for BizHawk

2. **Start BizHawk:**
   Double-click `start_bizhawk.bat` — load your ROM

3. **Load the Lua script:**
   In BizHawk: Tools > Lua Console > Open `bizhawk_bridge.lua`
   You should see "Bridge ready (file-based)." in the Lua console

## How it works

Communication is file-based (no sockets). The `bridge/` folder contains:
- `command.txt` — Python writes commands here
- `response.txt` — Lua writes responses here
- `ready.txt` — Lua creates this on startup so Python knows it's alive

## Restarting

- If you change `agent.py`: close the agent window, double-click `start_agent.bat` again
- If you change `bizhawk_bridge.lua`: in Lua Console, click the refresh/reload button

## Troubleshooting

- **Agent says "Waiting for BizHawk Lua script":** Load the Lua script in BizHawk
- **No output in agent window:** Make sure you're using `start_agent.bat` (runs with `-u` for unbuffered output)
