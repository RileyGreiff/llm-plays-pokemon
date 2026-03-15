-- Move Cursor Scanner for Vanilla FireRed
-- F5 = Take snapshot, F6 = Compare to snapshot
--
-- Steps:
-- 1. Be in battle at MOVE SELECT with cursor on top-left move
-- 2. Load this script, press F5
-- 3. Press RIGHT on d-pad (cursor moves to top-right)
-- 4. Press F6 to see what changed

local snapshot = {}
local SCAN_START = 0x02023000
local SCAN_END   = 0x02024FFF

console.log("=== Move Cursor Scanner ===")
console.log("Press F5 to snapshot, then move cursor, then press F6 to scan")

-- Also just continuously print the known addresses every 60 frames
local print_counter = 0

local function take_snapshot()
    snapshot = {}
    for addr = SCAN_START, SCAN_END do
        local ok, val = pcall(memory.read_u8, addr)
        if ok then
            snapshot[addr] = val
        end
    end
    local v1 = memory.read_u8(0x02023FFA)
    local v2 = memory.read_u8(0x02023FFC)
    console.log(string.format("SNAPSHOT TAKEN. 0x02023FFA=%d  0x02023FFC=%d", v1, v2))
end

local function scan_changes()
    if next(snapshot) == nil then
        console.log("ERROR: Press F5 first!")
        return
    end

    local v1 = memory.read_u8(0x02023FFA)
    local v2 = memory.read_u8(0x02023FFC)
    console.log(string.format("SCANNING NOW. 0x02023FFA=%d  0x02023FFC=%d", v1, v2))

    local exact = {}
    local changed = {}

    for addr = SCAN_START, SCAN_END do
        local ok, current = pcall(memory.read_u8, addr)
        if ok and snapshot[addr] then
            local old = snapshot[addr]
            if old ~= current then
                table.insert(changed, {addr = addr, old = old, new = current})
                if old == 0 and current == 1 then
                    table.insert(exact, {addr = addr, old = old, new = current})
                end
            end
        end
    end

    console.log(string.format("Changed: %d bytes. Exact 0->1: %d bytes", #changed, #exact))

    console.log("--- BEST (0 -> 1) ---")
    for _, r in ipairs(exact) do
        console.log(string.format("  0x%08X: %d -> %d", r.addr, r.old, r.new))
    end

    console.log("--- ALL CHANGES ---")
    for _, r in ipairs(changed) do
        console.log(string.format("  0x%08X: %d -> %d", r.addr, r.old, r.new))
    end

    local f = io.open("bridge/scan_results.txt", "w")
    if f then
        f:write("BEST (0 -> 1):\n")
        for _, r in ipairs(exact) do
            f:write(string.format("  0x%08X: %d -> %d\n", r.addr, r.old, r.new))
        end
        f:write("\nALL CHANGES:\n")
        for _, r in ipairs(changed) do
            f:write(string.format("  0x%08X: %d -> %d\n", r.addr, r.old, r.new))
        end
        f:close()
        console.log("Saved to bridge/scan_results.txt")
    end
end

while true do
    local keys = input.get()

    if keys["F5"] then
        take_snapshot()
        -- wait for release
        while input.get()["F5"] do emu.frameadvance() end
    end

    if keys["F6"] then
        scan_changes()
        while input.get()["F6"] do emu.frameadvance() end
    end

    -- Print known addresses periodically
    print_counter = print_counter + 1
    if print_counter % 120 == 0 then
        local v1 = memory.read_u8(0x02023FFA)
        local v2 = memory.read_u8(0x02023FFC)
        console.log(string.format("[live] 0x02023FFA=%d  0x02023FFC=%d", v1, v2))
    end

    emu.frameadvance()
end
