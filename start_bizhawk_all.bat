@echo off
title Launch 4 BizHawk Instances
cd /d "%~dp0"

set BIZHAWK=BizHawk\EmuHawk.exe
set ROM=Pokemon - FireRed Version (USA, Europe).gba
set LUA_DIR=BizHawk\Lua

echo Starting 4 BizHawk instances...

start "" "%BIZHAWK%" "%ROM%" --lua="%LUA_DIR%\bizhawk_bridge_0.lua"
timeout /t 3 /nobreak >NUL

start "" "%BIZHAWK%" "%ROM%" --lua="%LUA_DIR%\bizhawk_bridge_1.lua"
timeout /t 3 /nobreak >NUL

start "" "%BIZHAWK%" "%ROM%" --lua="%LUA_DIR%\bizhawk_bridge_2.lua"
timeout /t 3 /nobreak >NUL

start "" "%BIZHAWK%" "%ROM%" --lua="%LUA_DIR%\bizhawk_bridge_3.lua"

echo All 4 BizHawk instances launched.
echo Load the savestate in each instance, then run: python launch_training.py
pause
