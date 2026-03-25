@echo off
title BizHawk Instance 0
echo Starting BizHawk instance 0 (bridge_0/)
echo.
echo IMPORTANT: After BizHawk opens:
echo   1. Load ROM: Pokemon - FireRed Version (USA, Europe).gba
echo   2. Tools ^> Lua Console
echo   3. Open Script: Lua\bizhawk_bridge_0.lua
echo.
start "" "%~dp0BizHawk\EmuHawk.exe"
