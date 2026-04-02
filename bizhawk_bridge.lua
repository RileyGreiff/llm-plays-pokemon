-- BizHawk Lua file-based bridge for Python control (Pokemon FireRed / GBA)
-- Communication via files instead of sockets (eliminates lag from blocking socket calls)
-- 1. Run Python server first (python agent.py)
-- 2. Launch BizHawk (no --socket flags needed)
-- 3. Load ROM, then load this script in Tools > Lua Console

local POLL_INTERVAL = 30  -- check every ~0.5 seconds at 60fps
local frame_count = 0
local overlay_text = ""
local overlay_action = ""

-- File-based communication paths
local BRIDGE_DIR = "bridge"
local CMD_FILE = BRIDGE_DIR .. "/command.txt"
local RESP_FILE = BRIDGE_DIR .. "/response.txt"
local READY_FILE = BRIDGE_DIR .. "/ready.txt"

-- Create bridge directory
os.execute("mkdir " .. BRIDGE_DIR .. " 2>NUL")

-- Non-blocking button press state
local pending_button = nil
local pending_frames = 0

-- Write ready marker so Python knows we're alive
local function write_ready()
    local f = io.open(READY_FILE, "w")
    if f then
        f:write("ready")
        f:close()
    end
end

-- Read and delete command file (returns nil if no command)
local function read_command()
    local f = io.open(CMD_FILE, "r")
    if not f then return nil end
    local cmd = f:read("*a")
    f:close()
    os.remove(CMD_FILE)
    if cmd and cmd ~= "" then
        return cmd:gsub("%s+$", "")
    end
    return nil
end

-- Write response file
local function write_response(response)
    local f = io.open(RESP_FILE, "w")
    if f then
        f:write(response)
        f:close()
    end
end

local function screenshot_base64()
    local path = "screenshot_tmp.png"
    client.screenshot(path)
    local f = io.open(path, "rb")
    if not f then return "ERROR:no_screenshot" end
    local data = f:read("*a")
    f:close()
    os.remove(path)

    local b64chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    local result = {}
    local pad = (3 - #data % 3) % 3
    data = data .. string.rep('\0', pad)
    for i = 1, #data, 3 do
        local b1, b2, b3 = data:byte(i, i + 2)
        local n = b1 * 65536 + b2 * 256 + b3
        table.insert(result, b64chars:sub(math.floor(n / 262144) % 64 + 1, math.floor(n / 262144) % 64 + 1))
        table.insert(result, b64chars:sub(math.floor(n / 4096) % 64 + 1, math.floor(n / 4096) % 64 + 1))
        table.insert(result, b64chars:sub(math.floor(n / 64) % 64 + 1, math.floor(n / 64) % 64 + 1))
        table.insert(result, b64chars:sub(n % 64 + 1, n % 64 + 1))
    end
    local encoded = table.concat(result)
    if pad == 1 then
        encoded = encoded:sub(1, -2) .. '='
    elseif pad == 2 then
        encoded = encoded:sub(1, -3) .. '=='
    end
    return encoded
end

local function handle_command(cmd)
    cmd = cmd:gsub("%s+$", "")

    if cmd == "PING" then
        return "PONG"

    elseif cmd == "DOMAINS" then
        local domains = memory.getmemorydomainlist()
        return table.concat(domains, ",")

    elseif cmd:sub(1, 7) == "OVERLAY" then
        local payload = cmd:sub(9)
        local sep = payload:find("|")
        if sep then
            overlay_action = payload:sub(1, sep - 1)
            overlay_text = payload:sub(sep + 1)
        end
        return "OK"

    elseif cmd == "SCREENSHOT" then
        return screenshot_base64()

    elseif cmd:sub(1, 5) == "PRESS" then
        local parts = {}
        for part in cmd:gmatch("%S+") do
            table.insert(parts, part)
        end
        pending_button = parts[2]
        pending_frames = tonumber(parts[3]) or 15
        return nil  -- respond when press finishes

    elseif cmd == "GAMESTATE_FR" then
        local function ru8(addr)
            local ok, v = pcall(memory.read_u8, addr)
            return ok and v or 0
        end
        local function ru16(addr)
            local ok, v = pcall(memory.read_u16_le, addr)
            return ok and v or 0
        end
        local function ru32(addr)
            local ok, v = pcall(memory.read_u32_le, addr)
            return ok and v or 0
        end

        local sb1 = ru32(0x03005008)
        local sb2 = ru32(0x0300500C)
        local px = ru16(sb1 + 0x000)
        local py = ru16(sb1 + 0x002)
        local mapbank = ru8(sb1 + 0x004)
        local mapnum = ru8(sb1 + 0x005)

        local badge_byte = ru8(sb1 + 0xFE4)
        local badges = 0
        local b = badge_byte
        while b > 0 do
            badges = badges + (b & 1)
            b = b >> 1
        end

        local cb2 = ru32(0x030030F4)
        local game_state_map = {
            [0x080565B5] = "overworld",
            [0x08011101] = "battle",
            [0x08107EE1] = "bag",
            [0x0811EBA1] = "pokemon",
            [0x08135C35] = "transition",
            [0x08137EE9] = "summary",
            [0x08010509] = "transition",
            [0x0809D9E1] = "nickname_prompt",
            [0x0809FB71] = "naming",
        }
        local game_state_str = game_state_map[cb2] or "unknown"
        if game_state_str == "unknown" then
            console.log(string.format("  UNKNOWN cb2=0x%08X", cb2))
        end
        local battlers = 0
        if game_state_str == "battle" then
            local battleTypeFlags = ru32(0x02022B4C)  -- gBattleTypeFlags
            if (battleTypeFlags & 0x08) > 0 then      -- BATTLE_TYPE_TRAINER
                battlers = 2
            else
                battlers = 1  -- wild battle
            end
        end

        local partycount = ru8(0x02024029)

        local msgbox = ru8(0x0203709C)
        local scriptState = ru8(0x03000EB0)
        local textFlags = ru8(0x02021064)
        local lockField = ru8(0x0203707E)  -- sLockFieldControls: 1 when player controls locked (trainer approach, scripts)
        local dialogue = 0
        if msgbox > 0 or scriptState > 0 or (textFlags & 1) == 1 or lockField > 0 then
            dialogue = 1
        end

        local actionCursor = ru8(0x02023FF8)
        local moveCursor = ru8(0x02023FFC)  -- gMoveSelectionCursor[0]
        local battleMenuState = ru8(0x02023E82)  -- vanilla FireRed: 1=action menu, 2=move select, 4=executing/animations

        local results = {px, py, mapbank, mapnum, battlers, partycount, badges, dialogue, actionCursor, moveCursor, battleMenuState}

        local sub_order = {
            {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,3,1,2},{0,2,3,1},{0,3,2,1},
            {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,3,0,2},{1,2,3,0},{1,3,2,0},
            {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
            {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0}
        }

        local function decrypt_sub(base, sub_id)
            -- Decrypt a substructure (0=Growth, 1=Attacks, 2=EVs, 3=Misc)
            -- Returns offset to decrypted data start and the XOR key
            local pid = ru32(base + 0x00)
            local otid = ru32(base + 0x04)
            local key = pid ~ otid
            local order_idx = (pid % 24) + 1
            local order = sub_order[order_idx]
            local slot = -1
            for i = 1, 4 do
                if order[i] == sub_id then
                    slot = i - 1
                    break
                end
            end
            return base + 0x20 + (slot * 12), key
        end

        local function get_species(base)
            local off, key = decrypt_sub(base, 0)  -- Growth substructure
            local dec_word = ru32(off) ~ key
            return dec_word & 0xFFFF
        end

        local function get_moves(base)
            -- Attacks substructure: 4 move IDs (u16) + 4 PP (u8) = 12 bytes
            local off, key = decrypt_sub(base, 1)
            local w0 = ru32(off) ~ key       -- bytes 0-3: move1 (u16) + move2 (u16)
            local w1 = ru32(off + 4) ~ key   -- bytes 4-7: move3 (u16) + move4 (u16)
            local w2 = ru32(off + 8) ~ key   -- bytes 8-11: pp1, pp2, pp3, pp4 (u8 each)
            local move1 = w0 & 0xFFFF
            local move2 = (w0 >> 16) & 0xFFFF
            local move3 = w1 & 0xFFFF
            local move4 = (w1 >> 16) & 0xFFFF
            return move1, move2, move3, move4
        end

        local party_base = 0x02024284
        for i = 0, 5 do
            local base = party_base + (i * 0x64)
            if i < partycount then
                local species = get_species(base)
                local level = ru8(base + 0x54)
                local hp = ru16(base + 0x56)
                local maxhp = ru16(base + 0x58)
                local m1, m2, m3, m4 = get_moves(base)
                table.insert(results, species)
                table.insert(results, level)
                table.insert(results, hp)
                table.insert(results, maxhp)
                table.insert(results, m1)
                table.insert(results, m2)
                table.insert(results, m3)
                table.insert(results, m4)
            else
                table.insert(results, 0)
                table.insert(results, 0)
                table.insert(results, 0)
                table.insert(results, 0)
                table.insert(results, 0)
                table.insert(results, 0)
                table.insert(results, 0)
                table.insert(results, 0)
            end
        end

        local function has_item_in_pocket(offset, slots, target_item_id)
            for s = 0, slots - 1 do
                local slot_addr = sb1 + offset + (s * 4)
                local item_id = ru16(slot_addr)
                if item_id == target_item_id then
                    return 1
                end
            end
            return 0
        end

        local function has_any_item_in_pocket(offset, slots, target_ids)
            for s = 0, slots - 1 do
                local slot_addr = sb1 + offset + (s * 4)
                local item_id = ru16(slot_addr)
                for _, target_id in ipairs(target_ids) do
                    if item_id == target_id then
                        return 1
                    end
                end
            end
            return 0
        end

        -- Key items and HMs needed for structured progression planning.
        -- Base Kanto Pokedex candidate from SaveBlock2 diffing.
        table.insert(results, (sb2 ~= 0 and ru8(sb2 + 0x00A8) ~= 0) and 1 or 0) -- Pokedex
        table.insert(results, has_item_in_pocket(0x3B8, 30, 349)) -- Oaks Parcel
        table.insert(results, has_any_item_in_pocket(0x3B8, 30, {264, 360})) -- S.S. Ticket
        table.insert(results, has_item_in_pocket(0x3B8, 30, 359)) -- Silph Scope
        table.insert(results, has_item_in_pocket(0x3B8, 30, 350)) -- Poke Flute
        table.insert(results, has_item_in_pocket(0x3B8, 30, 351)) -- Secret Key
        table.insert(results, has_item_in_pocket(0x3B8, 30, 355)) -- Card Key
        table.insert(results, has_item_in_pocket(0x3B8, 30, 356)) -- Lift Key
        table.insert(results, has_item_in_pocket(0x3B8, 30, 377)) -- Tea
        table.insert(results, has_item_in_pocket(0x3B8, 30, 259)) -- Bicycle
        table.insert(results, has_item_in_pocket(0x3B8, 30, 352)) -- Bike Voucher
        table.insert(results, has_item_in_pocket(0x3B8, 30, 353)) -- Gold Teeth
        table.insert(results, has_item_in_pocket(0x3B8, 30, 375)) -- Tri-Pass
        table.insert(results, has_item_in_pocket(0x3B8, 30, 376)) -- Rainbow Pass
        table.insert(results, has_item_in_pocket(0x464, 64, 339)) -- HM01 Cut
        table.insert(results, has_item_in_pocket(0x464, 64, 340)) -- HM02 Fly
        table.insert(results, has_item_in_pocket(0x464, 64, 341)) -- HM03 Surf
        table.insert(results, has_item_in_pocket(0x464, 64, 342)) -- HM04 Strength
        table.insert(results, has_item_in_pocket(0x464, 64, 343)) -- HM05 Flash
        table.insert(results, has_item_in_pocket(0x464, 64, 344)) -- HM06 Rock Smash
        table.insert(results, has_item_in_pocket(0x464, 64, 345)) -- HM07 Waterfall
        table.insert(results, (sb2 ~= 0 and ru8(sb2 + 0x001B) == 0xB9) and 1 or 0) -- National Dex enabled

        if battlers > 0 then
            local gBattleMons = 0x02023BE4
            local enemy_species = ru16(gBattleMons + 0x58)
            local enemy_level = ru8(gBattleMons + 0x58 + 0x2A)
            local enemy_hp = ru16(gBattleMons + 0x58 + 0x28)
            local enemy_maxhp = ru16(gBattleMons + 0x58 + 0x2C)
            table.insert(results, enemy_species)
            table.insert(results, enemy_level)
            table.insert(results, enemy_hp)
            table.insert(results, enemy_maxhp)
            for m = 0, 3 do
                table.insert(results, ru16(gBattleMons + 0x0C + m * 2))
            end
            for m = 0, 3 do
                table.insert(results, ru8(gBattleMons + 0x24 + m))
            end
        else
            table.insert(results, 0)
            table.insert(results, 0)
            table.insert(results, 0)
            table.insert(results, 0)
            for m = 0, 7 do
                table.insert(results, 0)
            end
        end

        -- Expose the raw callback so Python can recognize UI states such as
        -- the nickname/naming screen without relying on brittle text heuristics.
        table.insert(results, cb2)

        local mem_str = table.concat(results, ",")

        local charset = {}
        charset[0x00] = " "
        for i = 0, 25 do charset[0xBB + i] = string.char(65 + i) end
        for i = 0, 25 do charset[0xD5 + i] = string.char(97 + i) end
        for i = 0, 9 do charset[0xA1 + i] = string.char(48 + i) end
        charset[0xAE] = "."  charset[0xAF] = "-"  charset[0xB0] = "..."
        charset[0xA8] = "!"  charset[0xA9] = "?"  charset[0xB8] = ","
        charset[0xAB] = "!"  charset[0xAC] = "?"  charset[0xB4] = "'"
        charset[0xB1] = "\"" charset[0xB2] = "\"" charset[0xB3] = "'"
        charset[0xB5] = "M"  charset[0xB6] = "F"
        charset[0xB7] = "$"  charset[0xB9] = "/"  charset[0xBA] = "("
        charset[0xFE] = "\n"

        local function read_text(addr, max_len)
            local chars = {}
            local skip_next = false
            for i = 0, max_len - 1 do
                if skip_next then
                    skip_next = false
                else
                    local byte = ru8(addr + i)
                    if byte == 0xFF then break end
                    if byte == 0xFD then
                        skip_next = true
                        table.insert(chars, "?")
                    elseif byte == 0xFC then
                        skip_next = true
                    elseif charset[byte] then
                        table.insert(chars, charset[byte])
                    end
                end
            end
            return table.concat(chars)
        end

        local dialogue_text = ""
        if dialogue > 0 then
            -- 0x02021D18 is commonly documented as the active message-box string.
            -- The nearby 0x02021CD0/CF0/D04 buffers are scratch string vars and can
            -- contain stale names or placeholders, so only use them as fallback.
            local active = read_text(0x02021D18, 160)
            local buf0 = read_text(0x02021CD0, 64)
            local buf1 = read_text(0x02021CF0, 32)
            local buf2 = read_text(0x02021D04, 32)
            dialogue_text = active
            if #dialogue_text == 0 then dialogue_text = buf0 end
            if #buf1 > #dialogue_text then dialogue_text = buf1 end
            if #buf2 > #dialogue_text then dialogue_text = buf2 end
        end

        -- Append object events inline
        local OBJ_BASE = 0x02036E38
        local OBJ_SIZE = 0x24
        local obj_parts = {}
        local playerFacing = 0

        for i = 0, 15 do
            local base = OBJ_BASE + i * OBJ_SIZE
            local ok_flags, flags = pcall(memory.read_u32_le, base)
            if ok_flags and (flags & 1) == 1 then
                local gfxId = memory.read_u8(base + 0x05)
                local localId = memory.read_u8(base + 0x08)
                local curX = memory.read_s16_le(base + 0x10)
                local curY = memory.read_s16_le(base + 0x12)
                local facing = memory.read_u8(base + 0x18) & 0xF
                local isPlayer = ((flags >> 16) & 1) == 1
                if isPlayer then
                    playerFacing = facing
                else
                    table.insert(obj_parts, localId .. "," .. gfxId .. "," .. curX .. "," .. curY .. "," .. facing)
                end
            end
        end

        local obj_str = "P," .. playerFacing .. "|" .. table.concat(obj_parts, "|")

        -- Read bag data when in bag screen
        local bag_str = ""
        if game_state_str == "bag" then
            local pockets = {
                {name="Items",     offset=0x310, slots=42},
                {name="KeyItems",  offset=0x3B8, slots=30},
                {name="PokeBalls", offset=0x430, slots=16},
                {name="TMs",       offset=0x464, slots=64},
                {name="Berries",   offset=0x564, slots=46},
            }
            local pocket_data = {}
            for _, pocket in ipairs(pockets) do
                local items = {}
                for s = 0, pocket.slots - 1 do
                    local slot_addr = sb1 + pocket.offset + (s * 4)
                    local item_id = ru16(slot_addr)
                    local quantity = ru16(slot_addr + 2)
                    if item_id > 0 and item_id < 400 then
                        table.insert(items, item_id .. ":" .. quantity)
                    end
                end
                table.insert(pocket_data, pocket.name .. "=" .. table.concat(items, ","))
            end

            -- Bag UI state (pocket/cursor/scroll) - addresses need verification
            local bag_pocket = ru8(0x0203AD02)
            local bag_cursor = ru8(0x0203AD04)
            local bag_scroll = ru8(0x0203AD06)

            bag_str = "|BAG:" .. bag_pocket .. "," .. bag_cursor .. "," .. bag_scroll .. ";" .. table.concat(pocket_data, ";")
        end

        return mem_str .. "|" .. game_state_str .. "|" .. dialogue_text .. "|OBJ:" .. obj_str .. bag_str

    elseif cmd:sub(1, 9) == "GAMESTATE" then
        local parts = {}
        for part in cmd:gmatch("%S+") do
            table.insert(parts, part)
        end
        local results = {}
        for i = 2, #parts do
            local addr = tonumber(parts[i])
            local ok, val = pcall(memory.read_u8, addr)
            if ok then
                table.insert(results, tostring(val))
            else
                table.insert(results, "0")
            end
        end
        local mem_str = table.concat(results, ",")
        local img_str = screenshot_base64()
        return mem_str .. "|" .. img_str

    elseif cmd:sub(1, 9) == "READMULTI" then
        local parts = {}
        for part in cmd:gmatch("%S+") do
            table.insert(parts, part)
        end
        local results = {}
        for i = 2, #parts do
            local addr = tonumber(parts[i])
            local ok, val = pcall(memory.read_u8, addr)
            if ok then
                table.insert(results, tostring(val))
            else
                table.insert(results, "0")
            end
        end
        return table.concat(results, ",")

    elseif cmd:sub(1, 4) == "READ" then
        local parts = {}
        for part in cmd:gmatch("%S+") do
            table.insert(parts, part)
        end
        local addr = tonumber(parts[2])
        local size = tonumber(parts[3]) or 1

        local ok, val
        if size == 1 then
            ok, val = pcall(memory.read_u8, addr)
        elseif size == 2 then
            ok, val = pcall(memory.read_u16_le, addr)
        elseif size == 4 then
            ok, val = pcall(memory.read_u32_le, addr)
        else
            return "ERROR: size must be 1, 2, or 4"
        end

        if ok then
            return tostring(val)
        else
            return "ERROR:" .. tostring(val)
        end

    elseif cmd == "DEBUG_OBJECTS" then
        local OBJ_BASE = 0x02036E38
        local OBJ_SIZE = 0x24
        for i = 0, 15 do
            local base = OBJ_BASE + i * OBJ_SIZE
            local flags = memory.read_u32_le(base)
            local gfxId = memory.read_u8(base + 0x05)
            local localId = memory.read_u8(base + 0x08)
            local curX = memory.read_s16_le(base + 0x10)
            local curY = memory.read_s16_le(base + 0x12)
            local facing = memory.read_u8(base + 0x18) & 0xF
            local active = (flags & 1) == 1
            local isPlayer = ((flags >> 16) & 1) == 1
            console.log(string.format("  Slot %2d: flags=0x%08X active=%s player=%s gfx=0x%02X local=%d pos=(%d,%d) facing=%d",
                i, flags, tostring(active), tostring(isPlayer), gfxId, localId, curX, curY, facing))
        end
        return "OK"

    elseif cmd == "OBJECTS" then
        local OBJ_BASE = 0x02036E38
        local OBJ_SIZE = 0x24
        local results = {}
        local playerFacing = 0

        for i = 0, 15 do
            local base = OBJ_BASE + i * OBJ_SIZE
            local ok_flags, flags = pcall(memory.read_u32_le, base)
            if ok_flags and (flags & 1) == 1 then
                local gfxId = memory.read_u8(base + 0x05)
                local localId = memory.read_u8(base + 0x08)
                local curX = memory.read_s16_le(base + 0x10)
                local curY = memory.read_s16_le(base + 0x12)
                local facing = memory.read_u8(base + 0x18) & 0xF
                local isPlayer = ((flags >> 16) & 1) == 1
                if isPlayer then
                    playerFacing = facing
                else
                    table.insert(results, localId .. "," .. gfxId .. "," .. curX .. "," .. curY .. "," .. facing)
                end
            end
        end

        return "P," .. playerFacing .. "|" .. table.concat(results, "|")

    elseif cmd == "BG_EVENTS" then
        local map_header = 0x02036DFC
        local events_ptr = memory.read_u32_le(map_header + 0x04)
        if events_ptr == 0 then return "ERROR:no_events_ptr" end

        local bg_count = memory.read_u8(events_ptr + 0x03)
        local bg_ptr = memory.read_u32_le(events_ptr + 0x10)
        local coord_count = memory.read_u8(events_ptr + 0x02)
        local coord_ptr = memory.read_u32_le(events_ptr + 0x0C)

        local results = {}

        if bg_ptr ~= 0 then
            for i = 0, math.min(bg_count - 1, 31) do
                local base = bg_ptr + i * 12
                local x = memory.read_u16_le(base + 0x00)
                local y = memory.read_u16_le(base + 0x02)
                local kind = memory.read_u8(base + 0x05)
                local label = "sign"
                if kind == 5 or kind == 7 then label = "hidden_item"
                elseif kind >= 1 and kind <= 4 then label = "script" end
                table.insert(results, "B," .. x .. "," .. y .. "," .. kind .. "," .. label)
            end
        end

        if coord_ptr ~= 0 then
            for i = 0, math.min(coord_count - 1, 31) do
                local base = coord_ptr + i * 16
                local x = memory.read_s16_le(base + 0x00)
                local y = memory.read_s16_le(base + 0x02)
                table.insert(results, "C," .. x .. "," .. y .. ",0,trigger")
            end
        end

        return table.concat(results, "|")

    elseif cmd == "MAP_CONNECTIONS" then
        local map_header = 0x02036DFC
        local conn_header = memory.read_u32_le(map_header + 0x0C)
        if conn_header == 0 then
            return ""
        end

        local count = memory.read_u32_le(conn_header + 0x00)
        local conn_ptr = memory.read_u32_le(conn_header + 0x04)
        if conn_ptr == 0 or count == 0 then
            return ""
        end

        local direction_names = {
            [1] = "south",
            [2] = "north",
            [3] = "west",
            [4] = "east",
            [5] = "dive",
            [6] = "emerge",
        }

        local results = {}
        for i = 0, math.min(count - 1, 15) do
            local base = conn_ptr + i * 12
            local direction = memory.read_u8(base + 0x00)
            local offset = memory.read_s32_le(base + 0x04)
            local mapGroup = memory.read_u8(base + 0x08)
            local mapNum = memory.read_u8(base + 0x09)
            local dir_name = direction_names[direction] or tostring(direction)
            table.insert(results, dir_name .. "," .. offset .. "," .. mapGroup .. "," .. mapNum)
        end

        return table.concat(results, "|")

    elseif cmd == "COLLISION" then
        local MAP_OFFSET = 7
        local NUM_PRIMARY = 640

        local function ru8_addr(addr)
            local ok, v = pcall(memory.read_u8, addr)
            return ok and v or 0
        end
        local function ru16_addr(addr)
            local ok, v = pcall(memory.read_u16_le, addr)
            return ok and v or 0
        end
        local function ru32_addr(addr)
            local ok, v = pcall(memory.read_u32_le, addr)
            return ok and v or 0
        end

        local vmap_w = ru32_addr(0x03005040)
        local vmap_h = ru32_addr(0x03005044)
        local map_w = vmap_w - 15
        local map_h = vmap_h - 14

        if map_w <= 0 or map_h <= 0 or map_w > 120 or map_h > 120 then
            return "ERROR:bad_map_dims_" .. map_w .. "x" .. map_h
        end

        local map_header = 0x02036DFC
        local layout_ptr = ru32_addr(map_header)
        local primary_tileset_ptr = ru32_addr(layout_ptr + 0x10)
        local secondary_tileset_ptr = ru32_addr(layout_ptr + 0x14)
        local primary_attr_ptr = ru32_addr(primary_tileset_ptr + 0x14)
        local secondary_attr_ptr = ru32_addr(secondary_tileset_ptr + 0x14)

        local function get_behavior(metatile_id)
            local attr_addr
            if metatile_id < NUM_PRIMARY then
                attr_addr = primary_attr_ptr + metatile_id * 4
            elseif metatile_id < 1024 then
                attr_addr = secondary_attr_ptr + (metatile_id - NUM_PRIMARY) * 4
            else
                return 0
            end
            local attr = ru32_addr(attr_addr)
            return attr & 0x1FF
        end

        local function tile_char(behavior, collision)
            if behavior == 0x02 then
                return "G"
            elseif behavior >= 0x60 and behavior <= 0x6F then
                if behavior >= 0x61 and behavior <= 0x63 then
                    return "S"
                else
                    return "D"
                end
            elseif collision == 0 then
                return "0"
            else
                return "1"
            end
        end

        local rows = {}
        for row = 0, map_h - 1 do
            local chars = {}
            for col = 0, map_w - 1 do
                local vx = col + MAP_OFFSET
                local vy = row + MAP_OFFSET
                local index = vx + vmap_w * vy
                local addr = 0x02031DFC + index * 2
                local tile = ru16_addr(addr)
                local collision = (tile >> 10) & 0x3
                local metatile_id = tile & 0x3FF
                local behavior = get_behavior(metatile_id)
                chars[col + 1] = tile_char(behavior, collision)
            end
            rows[row + 1] = table.concat(chars)
        end

        return map_w .. "," .. map_h .. "|" .. table.concat(rows)

    else
        return "ERROR: unknown command '" .. cmd .. "'"
    end
end

-- Clean up old files
os.remove(CMD_FILE)
os.remove(RESP_FILE)
write_ready()
console.log("Bridge ready (file-based).")

while true do
    frame_count = frame_count + 1

    -- Handle non-blocking button press (one frame at a time)
    if pending_button and pending_frames > 0 then
        joypad.set({[pending_button] = true})
        pending_frames = pending_frames - 1
        if pending_frames <= 0 then
            pending_button = nil
            write_response("OK")
        end
    -- Only poll for new commands when no press is active
    elseif frame_count % POLL_INTERVAL == 0 then
        local cmd = read_command()
        if cmd then
            local ok, response = pcall(handle_command, cmd)
            if ok then
                if response ~= nil then
                    write_response(response)
                end
            else
                write_response("ERROR:" .. tostring(response))
            end
        end
    end

    -- Draw overlay text on screen
    if overlay_action ~= "" then
        local w = client.bufferwidth()
        local h = client.bufferheight()

        gui.drawText(w - 2, 2, overlay_action, "white", "black", 10, "Consolas", "right")

        local max_chars = 40
        local y = h - 24
        local line1 = overlay_text:sub(1, max_chars)
        local line2 = overlay_text:sub(max_chars + 1, max_chars * 2)
        if #line2 > 0 then
            gui.drawText(2, y, line1, "white", "black", 8, "Consolas")
            gui.drawText(2, y + 10, line2, "white", "black", 8, "Consolas")
        else
            gui.drawText(2, y + 10, line1, "white", "black", 8, "Consolas")
        end
    end

    emu.frameadvance()
end
