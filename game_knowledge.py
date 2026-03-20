"""Pokemon FireRed game knowledge reference for local LLMs.

Provides map connectivity, town buildings, story gates, and battle tips
organized by phase (badge count). Designed to give small models the game
knowledge they lack compared to larger frontier models.
"""

# ---------------------------------------------------------------------------
# Phase data: keyed by badge count (0-8)
# ---------------------------------------------------------------------------

PHASE_DATA = {
    0: {
        "map_connectivity": {
            "PALLET_TOWN": "North: Route 1.",
            "ROUTE_1": "South: Pallet Town. North: Viridian City.",
            "VIRIDIAN_CITY": "South: Route 1. North: Route 2 (to Viridian Forest). West: Route 22 (Pokemon League gate, too early).",
            "ROUTE_2": "South: Viridian City. North: Viridian Forest / Pewter City.",
            "VIRIDIAN_FOREST": "South: Route 2 (Viridian side). North: Route 2 (Pewter side).",
            "PEWTER_CITY": "South: Route 2. East: Route 3 (need Badge 1 first).",
        },
        "town_buildings": {
            "PALLET_TOWN": "Oak's Lab (starter Pokemon + Pokedex). Player's house. Rival's house.",
            "VIRIDIAN_CITY": "Poke Mart (Oak's Parcel errand). Pokemon Center. Gym (locked until 7 badges).",
            "PEWTER_CITY": "Pewter Gym (Brock, Rock-type). Pokemon Center. Museum.",
        },
        "story_gates": [
            "Get starter Pokemon from Oak's Lab.",
            "Deliver Oak's Parcel from Viridian Mart back to Oak's Lab to get Pokedex.",
            "Travel north through Route 1 -> Viridian City -> Route 2 -> Viridian Forest -> Pewter City.",
            "Defeat Brock at Pewter Gym for Boulder Badge.",
        ],
        "recommended_level": 12,
        "training_areas": "Route 1 (Lv2-3 Pidgey/Rattata), Route 22 (Lv2-4), Viridian Forest (Lv3-5 Caterpie/Weedle/Pikachu).",
        "catch_tips": "Catch a Pidgey or Nidoran on Route 22 for a second party member. Pikachu in Viridian Forest is rare but useful.",
        "battle_tips": "Brock uses Rock-type (Geodude, Onix). Water and Grass are super effective. If you picked Charmander, level up and use Ember (still does OK). Bulbasaur's Vine Whip or Squirtle's Water Gun make Brock easy.",
    },
    1: {
        "map_connectivity": {
            "PEWTER_CITY": "East: Route 3.",
            "ROUTE_3": "West: Pewter City. East: Mt. Moon entrance.",
            "MT_MOON": "West entrance from Route 3. East exit to Route 4.",
            "ROUTE_4": "West: Mt. Moon exit. East: Cerulean City.",
            "CERULEAN_CITY": "South: Route 5. North: Route 24 (Nugget Bridge) -> Route 25 (Bill's house). East: Route 9 (need Cut). West: Route 4.",
            "ROUTE_24": "South: Cerulean City. North: Route 25.",
            "ROUTE_25": "South: Route 24. East end: Bill's house (Sea Cottage).",
        },
        "town_buildings": {
            "CERULEAN_CITY": "Cerulean Gym (Misty, Water-type). Pokemon Center. Bike Shop (need Bike Voucher).",
        },
        "story_gates": [
            "Travel through Route 3 and Mt. Moon to reach Cerulean City.",
            "Go north across Nugget Bridge (Route 24) and east on Route 25 to visit Bill.",
            "Bill gives S.S. Ticket after helping him.",
            "Defeat Misty at Cerulean Gym for Cascade Badge.",
        ],
        "recommended_level": 18,
        "training_areas": "Route 3 (Lv5-8), Mt. Moon (Lv7-10 Zubat/Geodude), Route 24-25 (Lv7-14).",
        "catch_tips": "Catch Geodude in Mt. Moon (strong vs many early gyms). Nidoran evolves early and learns good moves.",
        "battle_tips": "Misty uses Water-type (Staryu, Starmie). Grass and Electric are super effective. Starmie is strong — overlevel if possible. Bulbasaur/Ivysaur dominates here.",
    },
    2: {
        "map_connectivity": {
            "CERULEAN_CITY": "South: Route 5 (through burgled house backyard).",
            "ROUTE_5": "North: Cerulean City. South: Saffron Gate -> Underground Path -> Route 6.",
            "ROUTE_6": "North: Underground Path exit. South: Vermilion City.",
            "VERMILION_CITY": "North: Route 6. East: Route 11. South: Vermilion Port (S.S. Anne with ticket).",
        },
        "town_buildings": {
            "VERMILION_CITY": "Vermilion Gym (Lt. Surge, Electric-type, need Cut to access). Pokemon Center. S.S. Anne dock (S.S. Ticket required). Pokemon Fan Club (Bike Voucher).",
        },
        "story_gates": [
            "Go south from Cerulean through Route 5 -> Underground Path -> Route 6 to Vermilion City.",
            "Board S.S. Anne with S.S. Ticket. Find the Captain to get HM01 Cut.",
            "Teach Cut to a Pokemon. Cut the bush blocking Vermilion Gym.",
            "Defeat Lt. Surge at Vermilion Gym for Thunder Badge.",
        ],
        "recommended_level": 22,
        "training_areas": "Route 6 (Lv10-14), S.S. Anne trainers (good exp before it leaves).",
        "catch_tips": "Oddish/Bellsprout on Route 5-6 for Grass coverage. Diglett in Diglett's Cave (east of Vermilion) for Ground moves vs Surge.",
        "battle_tips": "Lt. Surge uses Electric-type (Voltorb, Pikachu, Raichu). Ground-type moves are super effective (Dig is great). The gym has trash can switches — solve the puzzle to reach Surge.",
    },
    3: {
        "map_connectivity": {
            "CERULEAN_CITY": "East: Route 9 (need Cut on bush).",
            "ROUTE_9": "West: Cerulean City. East: Route 10 / Rock Tunnel entrance.",
            "ROUTE_10": "North: Rock Tunnel entrance / Power Plant area. South: Lavender Town.",
            "ROCK_TUNNEL": "West entrance from Route 10 north. East exit to Route 10 south near Lavender.",
            "LAVENDER_TOWN": "West: Route 8 -> Saffron gate (need Tea). North: Route 10. South: Route 12. Pokemon Tower (need Silph Scope to complete).",
            "ROUTE_8": "East: Lavender Town. West: Saffron gate (blocked without Tea) / Underground Path to Route 7.",
            "ROUTE_7": "East: Saffron gate. West: Celadon City.",
            "CELADON_CITY": "East: Route 7. West: Route 16 (need Cut/Surf later).",
        },
        "town_buildings": {
            "CELADON_CITY": "Celadon Gym (Erika, Grass-type). Dept Store (buy items). Celadon Mansion back entrance (Tea from old lady on 1F). Game Corner (Rocket Hideout entrance).",
            "LAVENDER_TOWN": "Pokemon Tower (haunted, need Silph Scope). Pokemon Center. Name Rater.",
        },
        "story_gates": [
            "From Cerulean, cut bush east -> Route 9 -> Rock Tunnel -> Lavender Town.",
            "From Lavender, go west via Route 8 -> Underground Path -> Route 7 -> Celadon City.",
            "Get Tea from old lady in Celadon Mansion (back entrance, 1F).",
            "Defeat Erika at Celadon Gym for Rainbow Badge.",
        ],
        "recommended_level": 28,
        "training_areas": "Route 9-10 (Lv13-17), Rock Tunnel (Lv15-18 Machop/Geodude).",
        "catch_tips": "Growlithe on Route 8 (FireRed only) for Fire coverage. Machop in Rock Tunnel is a great Fighting type.",
        "battle_tips": "Erika uses Grass-type (Victreebel, Tangela, Vileplume). Fire, Ice, Flying, Poison are super effective. Most Fire or Flying types sweep her easily.",
    },
    4: {
        "map_connectivity": {
            "CELADON_CITY": "Game Corner has hidden stairs to Rocket Hideout B1F-B4F.",
            "LAVENDER_TOWN": "Pokemon Tower: 7 floors. Top floor: rescue Mr. Fuji.",
            "SAFFRON_CITY": "Accessible from Routes 5/6/7/8 gates after giving Tea to any guard. Central hub connecting to Celadon, Lavender, Cerulean, Vermilion.",
        },
        "town_buildings": {
            "CELADON_CITY": "Game Corner (poster hides Rocket Hideout stairs). Rocket Hideout B4F has Lift Key and Giovanni.",
            "LAVENDER_TOWN": "Pokemon Tower — need Silph Scope to identify ghosts. Mr. Fuji on 7F gives Poke Flute.",
        },
        "story_gates": [
            "Clear Rocket Hideout under Celadon Game Corner. Get Lift Key on B2F to reach Giovanni on B4F.",
            "Defeat Giovanni to get Silph Scope.",
            "Use Silph Scope in Pokemon Tower to identify ghost Marowak on 6F.",
            "Rescue Mr. Fuji on 7F of Pokemon Tower to get Poke Flute.",
        ],
        "recommended_level": 32,
        "training_areas": "Route 7-8 (Lv17-22), Pokemon Tower floors 3-6 (Lv13-20 Gastly, good exp).",
        "catch_tips": "Cubone in Pokemon Tower is useful. Eevee available in Celadon Mansion roof.",
        "battle_tips": "Giovanni in Rocket Hideout uses Ground-type (Onix, Rhyhorn, Kangaskhan). Water and Grass are super effective. Pokemon Tower has Ghost-type Gastly/Haunter — Normal-type moves won't work, need special moves.",
    },
    5: {
        "map_connectivity": {
            "SAFFRON_CITY": "North: Route 5 gate. South: Route 6 gate. East: Route 8 gate. West: Route 7 gate. Silph Co is the tall building in center.",
            "FUCHSIA_CITY": "North: Route 15 or Route 18. Accessible via Route 12/13/14/15 from Lavender or Route 16/17/18 from Celadon (need Cut or Poke Flute for Snorlax).",
            "ROUTE_12": "North: Lavender Town. South: Route 13. Snorlax blocks path (use Poke Flute).",
            "ROUTE_16": "East: Celadon City. West: Route 17 (Cycling Road). Snorlax blocks path (use Poke Flute). Need Bicycle.",
        },
        "town_buildings": {
            "SAFFRON_CITY": "Silph Co (11 floors, Card Key on 5F, Giovanni on 11F). Saffron Gym (Sabrina, Psychic-type, after clearing Silph Co). Fighting Dojo (free Pokemon).",
            "FUCHSIA_CITY": "Fuchsia Gym (Koga, Poison-type). Safari Zone entrance (HM03 Surf in Secret House, Gold Teeth in Area 3). Safari Warden (give Gold Teeth for HM04 Strength).",
        },
        "story_gates": [
            "Clear Silph Co: get Card Key on 5F, defeat Giovanni on 11F.",
            "Defeat Sabrina at Saffron Gym for Marsh Badge (optional order with Koga).",
            "Travel to Fuchsia City via Route 12-15 (wake Snorlax with Poke Flute) or Route 16-18.",
            "Safari Zone: get HM03 Surf from Secret House, Gold Teeth from Area 3.",
            "Give Gold Teeth to Safari Warden for HM04 Strength.",
            "Defeat Koga at Fuchsia Gym for Soul Badge.",
        ],
        "recommended_level": 38,
        "training_areas": "Route 12-15 (Lv22-28), Safari Zone (Lv22-30), Cycling Road Route 16-17 (Lv20-28).",
        "catch_tips": "Safari Zone has rare Pokemon (Chansey, Scyther, Tauros). Consider catching a Water-type that can learn Surf.",
        "battle_tips": "Sabrina uses Psychic-type (Kadabra, Mr. Mime, Venomoth, Alakazam). Bug, Ghost, Dark moves are super effective. Koga uses Poison-type (Koffing, Muk, Koffing, Weezing). Ground and Psychic are super effective. Watch out for Selfdestruct.",
    },
    6: {
        "map_connectivity": {
            "FUCHSIA_CITY": "South: Route 19 (Surf south).",
            "ROUTE_19": "North: Fuchsia City. South: Route 20 (water route).",
            "ROUTE_20": "East: Route 19. West: Cinnabar Island. Seafoam Islands in between.",
            "CINNABAR_ISLAND": "East: Route 20 (Surf). North: Route 21 (Surf to Pallet Town). Pokemon Mansion. Cinnabar Gym (locked, need Secret Key).",
            "ROUTE_21": "South: Cinnabar Island. North: Pallet Town.",
        },
        "town_buildings": {
            "CINNABAR_ISLAND": "Pokemon Mansion (Secret Key on B1F). Cinnabar Gym (Blaine, Fire-type, need Secret Key). Pokemon Center. Fossil Lab.",
        },
        "story_gates": [
            "Surf south from Fuchsia City via Route 19/20 to Cinnabar Island.",
            "Explore Pokemon Mansion to find Secret Key (B1F).",
            "Use Secret Key to enter Cinnabar Gym.",
            "Defeat Blaine at Cinnabar Gym for Volcano Badge.",
        ],
        "recommended_level": 42,
        "training_areas": "Pokemon Mansion (Lv28-34 Koffing/Grimer/Ponyta), Seafoam Islands (Lv28-34).",
        "catch_tips": "Articuno in Seafoam Islands (legendary, Lv50). Good Water types available via Surfing.",
        "battle_tips": "Blaine uses Fire-type (Growlithe, Ponyta, Rapidash, Arcanine). Water, Ground, Rock are super effective. Surf is your best move here.",
    },
    7: {
        "map_connectivity": {
            "CINNABAR_ISLAND": "North: Route 21 (Surf to Pallet Town).",
            "PALLET_TOWN": "North: Route 1 -> Viridian City.",
            "VIRIDIAN_CITY": "Viridian Gym is now open! West: Route 22 -> Route 23 -> Victory Road.",
        },
        "town_buildings": {
            "VIRIDIAN_CITY": "Viridian Gym (Giovanni, Ground-type). Pokemon Center.",
        },
        "story_gates": [
            "Surf north from Cinnabar or walk south from already-visited cities to reach Viridian City.",
            "Viridian Gym is now unlocked.",
            "Defeat Giovanni at Viridian Gym for Earth Badge (Badge #8).",
        ],
        "recommended_level": 45,
        "training_areas": "Victory Road (Lv36-42 Machoke/Graveler/Marowak), Route 23 (Lv32-40).",
        "catch_tips": "Moltres in Victory Road (legendary, Lv50).",
        "battle_tips": "Giovanni uses Ground-type (Rhyhorn, Dugtrio, Nidoqueen, Nidoking, Rhydon). Water, Grass, Ice are super effective. Surf and Ice Beam are great choices.",
    },
    8: {
        "map_connectivity": {
            "VIRIDIAN_CITY": "West: Route 22.",
            "ROUTE_22": "East: Viridian City. West: Route 23 (badge check gates).",
            "ROUTE_23": "East: Route 22. West: Victory Road entrance.",
            "VICTORY_ROAD": "East entrance from Route 23. West exit to Indigo Plateau.",
            "INDIGO_PLATEAU": "Pokemon Center (last chance to heal/shop). Elite Four entrance.",
        },
        "town_buildings": {
            "INDIGO_PLATEAU": "Pokemon Center. Elite Four challenge (Lorelei -> Bruno -> Agatha -> Lance -> Champion).",
        },
        "story_gates": [
            "Travel west from Viridian via Route 22 -> Route 23 (badge gates) -> Victory Road.",
            "Navigate Victory Road (need Strength to push boulders).",
            "Defeat Elite Four in order: Lorelei, Bruno, Agatha, Lance, then Champion.",
        ],
        "recommended_level": 55,
        "training_areas": "Victory Road (Lv36-42), rebattle trainers via VS Seeker, or grind at Cerulean Cave (post-game).",
        "catch_tips": "Make sure team has 6 Pokemon with diverse type coverage before challenging Elite Four.",
        "battle_tips": "Lorelei: Ice/Water — use Electric, Fighting, Rock. Bruno: Fighting/Rock — use Water, Psychic, Flying. Agatha: Ghost/Poison — use Ground, Psychic. Lance: Dragon/Flying — use Ice, Electric, Rock. Champion: mixed team, bring diverse coverage.",
    },
}

# ---------------------------------------------------------------------------
# Region extraction from map names
# ---------------------------------------------------------------------------

def _extract_region(map_name: str) -> str:
    """Extract the base region/town/route name from a full map name.

    E.g. 'VIRIDIAN_CITY_POKECENTER_1F' -> 'VIRIDIAN_CITY'
         'ROUTE_3' -> 'ROUTE_3'
         'PALLET_TOWN_OAKS_LAB' -> 'PALLET_TOWN'
    """
    upper = map_name.upper().replace(" ", "_")
    # Route maps: keep ROUTE_##
    if upper.startswith("ROUTE_"):
        parts = upper.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return f"ROUTE_{parts[1]}"
        return upper

    # Known town/city suffixes
    for suffix in ("_TOWN", "_CITY", "_ISLAND", "_PLATEAU", "_FOREST"):
        idx = upper.find(suffix)
        if idx != -1:
            return upper[:idx + len(suffix)]

    # Special areas
    for area in ("MT_MOON", "ROCK_TUNNEL", "VICTORY_ROAD", "SAFARI_ZONE",
                 "SILPH_CO", "POKEMON_TOWER", "POKEMON_MANSION",
                 "SS_ANNE", "SEAFOAM_ISLANDS", "POWER_PLANT",
                 "ROCKET_HIDEOUT", "VIRIDIAN_FOREST", "DIGLETTS_CAVE"):
        if upper.startswith(area):
            return area

    # Fallback: first two tokens
    parts = upper.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return upper


def _get_adjacent_maps(badge_count: int, region: str) -> list[str]:
    """Return map names directly connected to the given region in this phase."""
    phase = PHASE_DATA.get(badge_count) or PHASE_DATA.get(min(badge_count, 8))
    if not phase:
        return []
    connectivity = phase["map_connectivity"]
    desc = connectivity.get(region, "")
    # Pull out map names mentioned in the connectivity text
    adjacent = []
    for map_key in connectivity:
        if map_key != region and map_key.lower().replace("_", " ") in desc.lower().replace("_", " "):
            adjacent.append(map_key)
    # Also check if the current region is mentioned in other maps' connectivity
    for map_key, map_desc in connectivity.items():
        if map_key != region and region.lower().replace("_", " ") in map_desc.lower().replace("_", " "):
            if map_key not in adjacent:
                adjacent.append(map_key)
    return adjacent


# ---------------------------------------------------------------------------
# Main lookup function
# ---------------------------------------------------------------------------

def _find_navigation_knowledge(map_name: str) -> str:
    """Search all phases for navigation info about a map. Used when badge count is unknown."""
    region = _extract_region(map_name)
    for phase_key in range(9):
        phase = PHASE_DATA[phase_key]
        conn = phase["map_connectivity"].get(region, "")
        if conn:
            return f"[Game Knowledge — Map Connections]\n{region}: {conn}"
    return ""


def _check_gate_completion(gate: str, game_state: dict | None) -> bool:
    """Check if a story gate has been completed based on game state."""
    if not game_state:
        return False
    gl = gate.lower()
    # Starter Pokemon
    if "starter pokemon" in gl or "get starter" in gl:
        return bool(game_state.get("party"))
    # Oak's Parcel delivery + Pokedex
    if "parcel" in gl and ("deliver" in gl or "oak" in gl):
        return game_state.get("has_pokedex", False)
    if "pokedex" in gl and "get" in gl:
        return game_state.get("has_pokedex", False)
    # Badge-gated checks
    if "brock" in gl or "boulder badge" in gl:
        return game_state.get("badges", 0) >= 1
    if "misty" in gl or "cascade badge" in gl:
        return game_state.get("badges", 0) >= 2
    if "lt. surge" in gl or "thunder badge" in gl:
        return game_state.get("badges", 0) >= 3
    if "erika" in gl or "rainbow badge" in gl:
        return game_state.get("badges", 0) >= 4
    if "koga" in gl or "soul badge" in gl:
        return game_state.get("badges", 0) >= 5
    if "sabrina" in gl or "marsh badge" in gl:
        return game_state.get("badges", 0) >= 6
    if "blaine" in gl or "volcano badge" in gl:
        return game_state.get("badges", 0) >= 7
    if "giovanni" in gl and "gym" in gl or "earth badge" in gl:
        return game_state.get("badges", 0) >= 8
    # Key items
    if "s.s. ticket" in gl:
        return game_state.get("has_ss_ticket", False)
    if "hm01" in gl or "hm01 cut" in gl:
        return game_state.get("has_hm01_cut", False)
    if "silph scope" in gl:
        return game_state.get("has_silph_scope", False)
    if "lift key" in gl:
        return game_state.get("has_lift_key", False)
    if "poke flute" in gl or "poké flute" in gl:
        return game_state.get("has_poke_flute", False)
    if "card key" in gl:
        return game_state.get("has_card_key", False)
    if "secret key" in gl:
        return game_state.get("has_secret_key", False)
    if "hm03" in gl or "surf" in gl and "safari" in gl:
        return game_state.get("has_hm03_surf", False)
    if "gold teeth" in gl:
        return game_state.get("has_gold_teeth", False)
    if "hm04" in gl or "strength" in gl and "warden" in gl:
        return game_state.get("has_hm04_strength", False)
    if "tea" in gl and "celadon" in gl:
        return game_state.get("has_tea", False)
    return False


def get_relevant_knowledge(badge_count: int, map_name: str, context: str = "strategy",
                           game_state: dict | None = None) -> str:
    """Return game knowledge text appropriate for the given context.

    Args:
        badge_count: Current number of badges (0-8).
        map_name: Current map name from game state.
        context: One of "strategy", "tactical", "navigation", "battle".
        game_state: Optional game state dict for marking completed gates.

    Returns:
        A formatted string of relevant game knowledge to inject into prompts.
    """
    phase_key = min(badge_count, 8)
    phase = PHASE_DATA.get(phase_key)
    if not phase:
        return ""

    region = _extract_region(map_name)

    if context == "battle":
        return f"[Game Knowledge — Battle Tips]\n{phase['battle_tips']}"

    if context == "navigation":
        conn = phase["map_connectivity"].get(region, "")
        if conn:
            return f"[Game Knowledge — Map Connections]\n{region}: {conn}"
        # Search all phases if current region not in this phase
        return _find_navigation_knowledge(map_name)

    if context == "tactical":
        lines = []
        # Current map connectivity
        conn = phase["map_connectivity"].get(region, "")
        if conn:
            lines.append(f"{region}: {conn}")
        # Adjacent maps
        for adj in _get_adjacent_maps(phase_key, region):
            adj_conn = phase["map_connectivity"].get(adj, "")
            if adj_conn:
                lines.append(f"{adj}: {adj_conn}")
        # Include buildings for current region to prevent hallucination
        buildings = phase["town_buildings"].get(region, "")
        if buildings:
            lines.append(f"Buildings in {region}: {buildings}")
        if lines:
            return "[Game Knowledge — Nearby Area]\n" + "\n".join(lines)
        return ""

    # context == "strategy" (default)
    lines = []
    # Story gates with completion status
    lines.append("[Game Knowledge — Current Phase Story Gates]")
    for gate in phase["story_gates"]:
        done = _check_gate_completion(gate, game_state)
        marker = "DONE" if done else "TODO"
        lines.append(f"- [{marker}] {gate}")
    # Connectivity for current region and neighbors
    lines.append("\n[Game Knowledge — Map Connections]")
    conn = phase["map_connectivity"].get(region, "")
    if conn:
        lines.append(f"{region}: {conn}")
    for adj in _get_adjacent_maps(phase_key, region):
        adj_conn = phase["map_connectivity"].get(adj, "")
        if adj_conn:
            lines.append(f"{adj}: {adj_conn}")
    # Town buildings for current region
    buildings = phase["town_buildings"].get(region, "")
    if buildings:
        lines.append(f"\n[Game Knowledge — Buildings in {region}]")
        lines.append(buildings)
    # Level and training recommendations
    rec_level = phase.get("recommended_level", 0)
    training = phase.get("training_areas", "")
    catch = phase.get("catch_tips", "")
    if rec_level:
        lines.append(f"\n[Game Knowledge — Training]")
        lines.append(f"Recommended level for next gym: {rec_level}")
        # Compare to actual party level if available
        if game_state and game_state.get("party"):
            party = game_state["party"]
            lead_level = party[0].get("level", 0)
            max_level = max(p.get("level", 0) for p in party)
            party_size = len(party)
            lines.append(f"Your lead Pokemon is Lv{lead_level} (highest: Lv{max_level}, party size: {party_size}).")
            if max_level < rec_level - 5:
                lines.append(f"UNDERLEVELED — you should train before the next gym!")
            elif max_level < rec_level:
                lines.append(f"Slightly underleveled — consider some training.")
            if party_size < 2:
                lines.append("Only 1 Pokemon — try to catch more for a balanced team!")
        if training:
            lines.append(f"Training areas: {training}")
        if catch:
            lines.append(f"Catch tips: {catch}")
    return "\n".join(lines)
