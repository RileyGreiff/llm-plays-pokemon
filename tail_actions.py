import sqlite3
import json
import time

DB_PATH = "logs/runs.db"
last_id = 0

def print_action(r):
    (id_, ts, action, reason, model, inp_tok, out_tok, cache_read, cache_create,
     cost, gs_json, px, py, map_id, in_battle,
     screen_desc, progress_sum, exploration_sum, warnings) = r

    timestamp = ts[11:19]
    model_short = "Haiku" if "haiku" in (model or "") else "Sonnet"
    battle = " [BATTLE]" if in_battle else ""

    print(f"{'='*80}")
    print(f"  ACTION #{id_}  |  {timestamp}  |  {model_short}  |  ${cost:.4f}{battle}")
    print(f"{'='*80}")

    # LLM's screen description
    print(f"\n  [SCREEN DESCRIPTION]")
    print(f"  {screen_desc or '(not recorded)'}")

    # LLM's decision
    print(f"\n  [DECISION]")
    print(f"  Button: {action}")
    print(f"  Reason: {reason}")

    # Game state
    if gs_json:
        gs = json.loads(gs_json)
        print(f"\n  [GAME STATE]")
        print(f"  Map: {gs.get('map_name', '?')} (bank:{gs.get('map_bank')} num:{gs.get('map_num')} id:{gs.get('map_id')})")
        print(f"  Position: ({gs.get('player_x')}, {gs.get('player_y')})")
        print(f"  Badges: {gs.get('badges', 0)}  |  In Battle: {gs.get('in_battle')}  |  Dialogue: {gs.get('in_dialogue')}")

        party = gs.get("party", [])
        if party:
            print(f"  Party ({len(party)}):")
            for p in party:
                print(f"    {p['name']} Lv{p['level']}  HP: {p['hp']}/{p['max_hp']}")

        if gs.get("in_battle") and gs.get("enemy_species"):
            print(f"  Enemy: species={gs.get('enemy_species')} Lv{gs.get('enemy_level')} HP:{gs.get('enemy_hp')}")

    # Progress summary
    if progress_sum:
        print(f"\n  [PROGRESS]")
        print(f"  {progress_sum}")

    # Exploration minimap
    if exploration_sum:
        print(f"\n  [EXPLORATION MINIMAP]")
        for line in exploration_sum.split("\n"):
            print(f"  {line}")

    # Warnings / stuck detection
    if warnings:
        print(f"\n  [WARNINGS]")
        for line in warnings.split("\n"):
            print(f"  {line}")

    # Token usage
    print(f"\n  [TOKENS] in={inp_tok} out={out_tok} cached={cache_read} cache_write={cache_create}")
    print()

QUERY = """SELECT id, timestamp, action, reason, model,
    input_tokens, output_tokens, cache_read, cache_creation, cost_usd,
    game_state, player_x, player_y, map_id, in_battle,
    COALESCE(screen_description, ''), COALESCE(progress_summary, ''),
    COALESCE(exploration_summary, ''), COALESCE(warnings, '')
    FROM actions"""

# Show last 3 actions
try:
    db = sqlite3.connect(DB_PATH)
    rows = db.execute(QUERY + " ORDER BY id DESC LIMIT 3").fetchall()
    db.close()
    if rows:
        last_id = rows[0][0]
        for r in reversed(rows):
            print_action(r)
except Exception as e:
    print(f"Error: {e}")

print("--- Watching for new actions every 15s ---\n")

while True:
    time.sleep(15)
    try:
        db = sqlite3.connect(DB_PATH)
        rows = db.execute(QUERY + " WHERE id > ? ORDER BY id", (last_id,)).fetchall()
        db.close()
        for r in rows:
            print_action(r)
            last_id = r[0]
    except Exception:
        pass
