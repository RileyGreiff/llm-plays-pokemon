"""Snapshot and diff FireRed save memory around Pokedex acquisition."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from emulator import _send_command
from memory import SAVEBLOCK1_PTR, SAVEBLOCK2_PTR

SNAPSHOT_DIR = os.path.join("logs", "pokedex_diff")
SB1_LENGTH = 0x1200
SB2_LENGTH = 0x0200
READ_CHUNK = 256


def _read_u32(address: int) -> int:
    return int(_send_command(f"READ {address} 4").strip())


def _read_bytes(start: int, length: int) -> list[int]:
    data: list[int] = []
    offset = 0
    while offset < length:
        chunk_len = min(READ_CHUNK, length - offset)
        addrs = " ".join(str(start + offset + i) for i in range(chunk_len))
        raw = _send_command(f"READMULTI {addrs}", timeout=5.0)
        data.extend(int(x) for x in raw.split(",") if x)
        offset += chunk_len
    return data


def _snapshot_payload() -> dict:
    sb1 = _read_u32(SAVEBLOCK1_PTR)
    sb2 = _read_u32(SAVEBLOCK2_PTR)
    return {
        "created_at": datetime.now().isoformat(),
        "sb1_ptr": sb1,
        "sb2_ptr": sb2,
        "sb1_length": SB1_LENGTH,
        "sb2_length": SB2_LENGTH,
        "sb1": _read_bytes(sb1, SB1_LENGTH),
        "sb2": _read_bytes(sb2, SB2_LENGTH),
    }


def _snapshot_path(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return os.path.join(SNAPSHOT_DIR, f"{safe}.json")


def snapshot(name: str) -> None:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    payload = _snapshot_payload()
    path = _snapshot_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved snapshot: {path}")
    print(f"SB1=0x{payload['sb1_ptr']:08X} SB2=0x{payload['sb2_ptr']:08X}")
    print("Notable region: SB2 + 0x018 is struct Pokedex.")


def _load_snapshot(name: str) -> dict:
    path = _snapshot_path(name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_diffs(label: str, before: list[int], after: list[int], max_lines: int = 200) -> list[str]:
    lines: list[str] = []
    for idx, (old, new) in enumerate(zip(before, after)):
        if old == new:
            continue
        marker = ""
        if label == "SB2" and 0x18 <= idx < 0x18 + 0x80:
            marker = "  <-- struct Pokedex region"
        lines.append(f"{label}+0x{idx:04X}: {old:02X} -> {new:02X}{marker}")
        if len(lines) >= max_lines:
            break
    return lines


def diff(before_name: str, after_name: str) -> None:
    before = _load_snapshot(before_name)
    after = _load_snapshot(after_name)

    print(f"Before: {_snapshot_path(before_name)}")
    print(f"After:  {_snapshot_path(after_name)}")
    print(f"SB1 ptrs: 0x{before['sb1_ptr']:08X} -> 0x{after['sb1_ptr']:08X}")
    print(f"SB2 ptrs: 0x{before['sb2_ptr']:08X} -> 0x{after['sb2_ptr']:08X}")

    sb1_lines = _format_diffs("SB1", before["sb1"], after["sb1"])
    sb2_lines = _format_diffs("SB2", before["sb2"], after["sb2"])

    print("\nSB2 changes:")
    if sb2_lines:
        for line in sb2_lines:
            print(line)
    else:
        print("(none)")

    print("\nSB1 changes:")
    if sb1_lines:
        for line in sb1_lines:
            print(line)
    else:
        print("(none)")


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in {"snapshot", "diff"}:
        print("Usage:")
        print(r"  .\venv\Scripts\python.exe pokedex_diff.py snapshot before_pokedex")
        print(r"  .\venv\Scripts\python.exe pokedex_diff.py diff before_pokedex after_pokedex")
        return 1

    cmd = argv[1]
    if cmd == "snapshot":
        snapshot(argv[2])
        return 0

    if len(argv) < 4:
        print("diff requires two snapshot names")
        return 1
    diff(argv[2], argv[3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
