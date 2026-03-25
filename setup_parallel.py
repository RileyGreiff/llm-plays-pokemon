"""Generate per-instance Lua scripts and batch files for parallel RL training.

Creates:
  - bizhawk_bridge_N.lua    (Lua script with hardcoded bridge_N/ directory)
  - bridge_N/               (bridge directory for each instance)
  - start_bizhawk_N.bat     (launches BizHawk instance N)
  - start_training.bat      (launches all workers + trainer)

Usage:
    python setup_parallel.py --instances 2
    python setup_parallel.py --instances 4
"""

import argparse
import os
import re


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BIZHAWK_DIR = os.path.join(PROJECT_DIR, "BizHawk")
ROM_NAME = "Pokemon - FireRed Version (USA, Europe).gba"
LUA_SOURCE = os.path.join(PROJECT_DIR, "bizhawk_bridge.lua")


def generate_lua_script(instance_id: int) -> str:
    """Read the base Lua script and replace the bridge directory."""
    with open(LUA_SOURCE, "r") as f:
        lua_code = f.read()

    # Replace the BRIDGE_DIR line with absolute path so it works
    # regardless of BizHawk's working directory
    abs_bridge = os.path.join(PROJECT_DIR, f"bridge_{instance_id}").replace("\\", "/")
    lua_code = re.sub(
        r'local BRIDGE_DIR = os\.getenv\("BIZHAWK_BRIDGE_DIR"\) or "bridge"',
        f'local BRIDGE_DIR = "{abs_bridge}"',
        lua_code,
    )

    return lua_code


def generate_bizhawk_bat(instance_id: int) -> str:
    """Generate a .bat file that launches BizHawk and auto-loads the Lua script."""
    lua_script = f"bizhawk_bridge_{instance_id}.lua"
    # BizHawk loads Lua from its own Lua/ directory, so we copy there
    lua_dest = os.path.join("BizHawk", "Lua", lua_script)
    return f"""@echo off
title BizHawk Instance {instance_id}
echo Starting BizHawk instance {instance_id} (bridge_{instance_id}/)
echo.
echo IMPORTANT: After BizHawk opens:
echo   1. Load ROM: {ROM_NAME}
echo   2. Tools ^> Lua Console
echo   3. Open Script: Lua\\{lua_script}
echo.
start "" "%~dp0BizHawk\\EmuHawk.exe"
"""


def generate_training_bat(num_instances: int, threshold: int = 2048) -> str:
    """Generate a .bat file that launches all workers + trainer."""
    lines = [
        "@echo off",
        f"title RL Training ({num_instances} workers)",
        "cd /d \"%~dp0\"",
        "call venv\\Scripts\\activate.bat",
        "",
        f"echo Starting RL training with {num_instances} workers...",
        "echo.",
        "",
        ":: Start the central trainer",
        f'start "RL Trainer" cmd /k "venv\\Scripts\\activate.bat && python -u rl_trainer.py --update-threshold {threshold}"',
        "timeout /t 2 /nobreak >NUL",
        "",
    ]

    for i in range(num_instances):
        lines.extend([
            f":: Start worker {i}",
            f'start "RL Worker {i}" cmd /k "venv\\Scripts\\activate.bat && python -u rl_worker.py --worker-id {i}"',
            "timeout /t 1 /nobreak >NUL",
            "",
        ])

    lines.extend([
        f"echo All {num_instances} workers + trainer started.",
        "echo Close this window or press Ctrl+C in individual windows to stop.",
        "pause",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Setup parallel RL training")
    parser.add_argument("--instances", type=int, default=2, help="Number of parallel instances")
    parser.add_argument("--threshold", type=int, default=2048, help="PPO update step threshold")
    args = parser.parse_args()

    n = args.instances
    print(f"Setting up {n} parallel training instances...\n")

    # Create bridge directories
    for i in range(n):
        bridge_dir = os.path.join(PROJECT_DIR, f"bridge_{i}")
        os.makedirs(bridge_dir, exist_ok=True)
        print(f"  Created: bridge_{i}/")

    # Generate per-instance Lua scripts
    lua_dir = os.path.join(BIZHAWK_DIR, "Lua")
    for i in range(n):
        lua_code = generate_lua_script(i)
        lua_path = os.path.join(lua_dir, f"bizhawk_bridge_{i}.lua")
        with open(lua_path, "w") as f:
            f.write(lua_code)
        print(f"  Created: BizHawk/Lua/bizhawk_bridge_{i}.lua")

    # Generate per-instance BizHawk batch files
    for i in range(n):
        bat_content = generate_bizhawk_bat(i)
        bat_path = os.path.join(PROJECT_DIR, f"start_bizhawk_{i}.bat")
        with open(bat_path, "w") as f:
            f.write(bat_content)
        print(f"  Created: start_bizhawk_{i}.bat")

    # Generate training launch script
    training_bat = generate_training_bat(n, args.threshold)
    bat_path = os.path.join(PROJECT_DIR, "start_training.bat")
    with open(bat_path, "w") as f:
        f.write(training_bat)
    print(f"  Created: start_training.bat")

    # Ensure checkpoint dir exists
    os.makedirs(os.path.join(PROJECT_DIR, "rl_checkpoints"), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Setup complete! To start training:")
    print(f"{'='*60}")
    print(f"")
    print(f"  Step 1: Launch BizHawk instances")
    for i in range(n):
        print(f"    - Double-click start_bizhawk_{i}.bat")
        print(f"      Load ROM, then load Lua/bizhawk_bridge_{i}.lua")
    print(f"")
    print(f"  Step 2: Start training")
    print(f"    - Double-click start_training.bat")
    print(f"    - Or: python launch_training.py --workers {n}")
    print(f"")
    print(f"  Each instance uses ~1.5 GB RAM + 3 CPU cores.")
    print(f"  {n} instances ~ {n * 1.5:.1f} GB RAM, {n * 3} CPU cores.")


if __name__ == "__main__":
    main()
