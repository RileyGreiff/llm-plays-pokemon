"""Pokemon FireRed (GBA) memory address map for BizHawk.

FireRed uses dynamic memory allocation (DMA) for save data.
Some addresses require pointer dereference from IWRAM.
Party Pokemon species data is encrypted (XOR with PID ^ OTID).
"""

# --- Pointer addresses (IWRAM, read as u32 to get EWRAM target) ---
SAVEBLOCK1_PTR = 0x03005008  # -> player position, badges, flags
SAVEBLOCK2_PTR = 0x0300500C  # -> trainer data, pokedex

# --- SaveBlock1 offsets (add to dereferenced pointer) ---
SB1_PLAYER_X = 0x000   # u16
SB1_PLAYER_Y = 0x002   # u16
SB1_MAP_NUM = 0x004     # u8
SB1_MAP_BANK = 0x005    # u8
SB1_BADGE_OFFSET = 0xFE4  # u16, bits 0-7 = badges

# --- Static EWRAM addresses (no pointer needed) ---
# Map (alternative static addresses)
MAP_BANK_STATIC = 0x02031DBC      # u8
MAP_NUMBER_STATIC = 0x02031DBD    # u8

# Battle state
BATTLE_TYPE_FLAGS = 0x02022B4C    # u32, non-zero = in battle
BATTLERS_COUNT = 0x02023BCC       # u8, 0 = not in battle
BATTLE_OUTCOME = 0x02023E8A       # u8, 0 = in battle

# Active battle Pokemon (unencrypted, 0x58 bytes each)
BATTLE_MONS = 0x02023BE4          # species at +0x00 (u16), level at +0x54 (u8)

# Party
PARTY_COUNT = 0x02024029          # u8
PARTY_BASE = 0x02024284           # 100 bytes per Pokemon
PARTY_STRIDE = 0x64               # 100 bytes

# Enemy party (same structure)
ENEMY_PARTY_BASE = 0x0202402C

# Per-Pokemon offsets (within 100-byte structure)
MON_PID_OFFSET = 0x00             # u32 - personality value
MON_OTID_OFFSET = 0x04            # u32 - trainer ID
MON_ENCRYPTED_OFFSET = 0x20       # 48 bytes of encrypted substructure data
MON_LEVEL_OFFSET = 0x54           # u8 (unencrypted)
MON_HP_OFFSET = 0x56              # u16 (unencrypted)
MON_MAX_HP_OFFSET = 0x58          # u16 (unencrypted)
MON_ATTACK_OFFSET = 0x5A          # u16
MON_DEFENSE_OFFSET = 0x5C         # u16
MON_SPEED_OFFSET = 0x5E           # u16

# Substructure order table (personality % 24)
# G=Growth(0), A=Attacks(1), E=EVs(2), M=Misc(3)
SUBSTRUCTURE_ORDER = [
    [0,1,2,3], [0,1,3,2], [0,2,1,3], [0,3,1,2], [0,2,3,1], [0,3,2,1],
    [1,0,2,3], [1,0,3,2], [1,2,0,3], [1,3,0,2], [1,2,3,0], [1,3,2,0],
    [2,0,1,3], [2,0,3,1], [2,1,0,3], [2,1,3,0], [2,3,0,1], [2,3,1,0],
    [3,0,1,2], [3,0,2,1], [3,1,0,2], [3,1,2,0], [3,2,0,1], [3,2,1,0],
]

# Gen 3 National Pokedex species IDs (index number -> name)
POKEMON_NAMES = {
    1: "Bulbasaur", 2: "Ivysaur", 3: "Venusaur", 4: "Charmander",
    5: "Charmeleon", 6: "Charizard", 7: "Squirtle", 8: "Wartortle",
    9: "Blastoise", 10: "Caterpie", 11: "Metapod", 12: "Butterfree",
    13: "Weedle", 14: "Kakuna", 15: "Beedrill", 16: "Pidgey",
    17: "Pidgeotto", 18: "Pidgeot", 19: "Rattata", 20: "Raticate",
    21: "Spearow", 22: "Fearow", 23: "Ekans", 24: "Arbok",
    25: "Pikachu", 26: "Raichu", 27: "Sandshrew", 28: "Sandslash",
    29: "Nidoran_F", 30: "Nidorina", 31: "Nidoqueen", 32: "Nidoran_M",
    33: "Nidorino", 34: "Nidoking", 35: "Clefairy", 36: "Clefable",
    37: "Vulpix", 38: "Ninetales", 39: "Jigglypuff", 40: "Wigglytuff",
    41: "Zubat", 42: "Golbat", 43: "Oddish", 44: "Gloom",
    45: "Vileplume", 46: "Paras", 47: "Parasect", 48: "Venonat",
    49: "Venomoth", 50: "Diglett", 51: "Dugtrio", 52: "Meowth",
    53: "Persian", 54: "Psyduck", 55: "Golduck", 56: "Mankey",
    57: "Primeape", 58: "Growlithe", 59: "Arcanine", 60: "Poliwag",
    61: "Poliwhirl", 62: "Poliwrath", 63: "Abra", 64: "Kadabra",
    65: "Alakazam", 66: "Machop", 67: "Machoke", 68: "Machamp",
    69: "Bellsprout", 70: "Weepinbell", 71: "Victreebel", 72: "Tentacool",
    73: "Tentacruel", 74: "Geodude", 75: "Graveler", 76: "Golem",
    77: "Ponyta", 78: "Rapidash", 79: "Slowpoke", 80: "Slowbro",
    81: "Magnemite", 82: "Magneton", 83: "Farfetchd", 84: "Doduo",
    85: "Dodrio", 86: "Seel", 87: "Dewgong", 88: "Grimer",
    89: "Muk", 90: "Shellder", 91: "Cloyster", 92: "Gastly",
    93: "Haunter", 94: "Gengar", 95: "Onix", 96: "Drowzee",
    97: "Hypno", 98: "Krabby", 99: "Kingler", 100: "Voltorb",
    101: "Electrode", 102: "Exeggcute", 103: "Exeggutor", 104: "Cubone",
    105: "Marowak", 106: "Hitmonlee", 107: "Hitmonchan", 108: "Lickitung",
    109: "Koffing", 110: "Weezing", 111: "Rhyhorn", 112: "Rhydon",
    113: "Chansey", 114: "Tangela", 115: "Kangaskhan", 116: "Horsea",
    117: "Seadra", 118: "Goldeen", 119: "Seaking", 120: "Staryu",
    121: "Starmie", 122: "Mr.Mime", 123: "Scyther", 124: "Jynx",
    125: "Electabuzz", 126: "Magmar", 127: "Pinsir", 128: "Tauros",
    129: "Magikarp", 130: "Gyarados", 131: "Lapras", 132: "Ditto",
    133: "Eevee", 134: "Vaporeon", 135: "Jolteon", 136: "Flareon",
    137: "Porygon", 138: "Omanyte", 139: "Omastar", 140: "Kabuto",
    141: "Kabutops", 142: "Aerodactyl", 143: "Snorlax", 144: "Articuno",
    145: "Zapdos", 146: "Moltres", 147: "Dratini", 148: "Dragonair",
    149: "Dragonite", 150: "Mewtwo", 151: "Mew",
    252: "Treecko", 253: "Grovyle", 254: "Sceptile",
    255: "Torchic", 256: "Combusken", 257: "Blaziken",
    258: "Mudkip", 259: "Marshtomp", 260: "Swampert",
}

# FireRed map bank:number -> name lookup (from pokefirered decomp)
MAP_NAMES = {
    # Bank 0: Link/Multiplayer
    (0, 0): "BATTLE_COLOSSEUM_2P",
    (0, 1): "TRADE_CENTER",
    (0, 2): "RECORD_CORNER",
    (0, 3): "BATTLE_COLOSSEUM_4P",
    (0, 4): "UNION_ROOM",

    # Bank 1: Dungeons
    (1, 0): "VIRIDIAN_FOREST",
    (1, 1): "MT_MOON_1F",
    (1, 2): "MT_MOON_B1F",
    (1, 3): "MT_MOON_B2F",
    (1, 4): "SS_ANNE_EXTERIOR",
    (1, 5): "SS_ANNE_1F_CORRIDOR",
    (1, 6): "SS_ANNE_2F_CORRIDOR",
    (1, 7): "SS_ANNE_3F_CORRIDOR",
    (1, 8): "SS_ANNE_B1F_CORRIDOR",
    (1, 9): "SS_ANNE_DECK",
    (1, 10): "SS_ANNE_KITCHEN",
    (1, 11): "SS_ANNE_CAPTAINS_OFFICE",
    (1, 12): "SS_ANNE_1F_ROOM1",
    (1, 13): "SS_ANNE_1F_ROOM2",
    (1, 14): "SS_ANNE_1F_ROOM3",
    (1, 15): "SS_ANNE_1F_ROOM4",
    (1, 16): "SS_ANNE_1F_ROOM5",
    (1, 17): "SS_ANNE_1F_ROOM7",
    (1, 18): "SS_ANNE_2F_ROOM1",
    (1, 19): "SS_ANNE_2F_ROOM2",
    (1, 20): "SS_ANNE_2F_ROOM3",
    (1, 21): "SS_ANNE_2F_ROOM4",
    (1, 22): "SS_ANNE_2F_ROOM5",
    (1, 23): "SS_ANNE_2F_ROOM6",
    (1, 24): "SS_ANNE_B1F_ROOM1",
    (1, 25): "SS_ANNE_B1F_ROOM2",
    (1, 26): "SS_ANNE_B1F_ROOM3",
    (1, 27): "SS_ANNE_B1F_ROOM4",
    (1, 28): "SS_ANNE_B1F_ROOM5",
    (1, 29): "SS_ANNE_1F_ROOM6",
    (1, 30): "UNDERGROUND_PATH_NORTH_ENTRANCE",
    (1, 31): "UNDERGROUND_PATH_NS_TUNNEL",
    (1, 32): "UNDERGROUND_PATH_SOUTH_ENTRANCE",
    (1, 33): "UNDERGROUND_PATH_WEST_ENTRANCE",
    (1, 34): "UNDERGROUND_PATH_EW_TUNNEL",
    (1, 35): "UNDERGROUND_PATH_EAST_ENTRANCE",
    (1, 36): "DIGLETTS_CAVE_NORTH_ENTRANCE",
    (1, 37): "DIGLETTS_CAVE_B1F",
    (1, 38): "DIGLETTS_CAVE_SOUTH_ENTRANCE",
    (1, 39): "VICTORY_ROAD_1F",
    (1, 40): "VICTORY_ROAD_2F",
    (1, 41): "VICTORY_ROAD_3F",
    (1, 42): "ROCKET_HIDEOUT_B1F",
    (1, 43): "ROCKET_HIDEOUT_B2F",
    (1, 44): "ROCKET_HIDEOUT_B3F",
    (1, 45): "ROCKET_HIDEOUT_B4F",
    (1, 46): "ROCKET_HIDEOUT_ELEVATOR",
    (1, 47): "SILPH_CO_1F",
    (1, 48): "SILPH_CO_2F",
    (1, 49): "SILPH_CO_3F",
    (1, 50): "SILPH_CO_4F",
    (1, 51): "SILPH_CO_5F",
    (1, 52): "SILPH_CO_6F",
    (1, 53): "SILPH_CO_7F",
    (1, 54): "SILPH_CO_8F",
    (1, 55): "SILPH_CO_9F",
    (1, 56): "SILPH_CO_10F",
    (1, 57): "SILPH_CO_11F",
    (1, 58): "SILPH_CO_ELEVATOR",
    (1, 59): "POKEMON_MANSION_1F",
    (1, 60): "POKEMON_MANSION_2F",
    (1, 61): "POKEMON_MANSION_3F",
    (1, 62): "POKEMON_MANSION_B1F",
    (1, 63): "SAFARI_ZONE_CENTER",
    (1, 64): "SAFARI_ZONE_EAST",
    (1, 65): "SAFARI_ZONE_NORTH",
    (1, 66): "SAFARI_ZONE_WEST",
    (1, 67): "SAFARI_ZONE_CENTER_RESTHOUSE",
    (1, 68): "SAFARI_ZONE_EAST_RESTHOUSE",
    (1, 69): "SAFARI_ZONE_NORTH_RESTHOUSE",
    (1, 70): "SAFARI_ZONE_WEST_RESTHOUSE",
    (1, 71): "SAFARI_ZONE_SECRET_HOUSE",
    (1, 72): "CERULEAN_CAVE_1F",
    (1, 73): "CERULEAN_CAVE_2F",
    (1, 74): "CERULEAN_CAVE_B1F",
    (1, 75): "POKEMON_LEAGUE_LORELEIS_ROOM",
    (1, 76): "POKEMON_LEAGUE_BRUNOS_ROOM",
    (1, 77): "POKEMON_LEAGUE_AGATHAS_ROOM",
    (1, 78): "POKEMON_LEAGUE_LANCES_ROOM",
    (1, 79): "POKEMON_LEAGUE_CHAMPIONS_ROOM",
    (1, 80): "POKEMON_LEAGUE_HALL_OF_FAME",
    (1, 81): "ROCK_TUNNEL_1F",
    (1, 82): "ROCK_TUNNEL_B1F",
    (1, 83): "SEAFOAM_ISLANDS_1F",
    (1, 84): "SEAFOAM_ISLANDS_B1F",
    (1, 85): "SEAFOAM_ISLANDS_B2F",
    (1, 86): "SEAFOAM_ISLANDS_B3F",
    (1, 87): "SEAFOAM_ISLANDS_B4F",
    (1, 88): "POKEMON_TOWER_1F",
    (1, 89): "POKEMON_TOWER_2F",
    (1, 90): "POKEMON_TOWER_3F",
    (1, 91): "POKEMON_TOWER_4F",
    (1, 92): "POKEMON_TOWER_5F",
    (1, 93): "POKEMON_TOWER_6F",
    (1, 94): "POKEMON_TOWER_7F",
    (1, 95): "POWER_PLANT",
    (1, 96): "MT_EMBER_RUBY_PATH_B4F",
    (1, 97): "MT_EMBER_EXTERIOR",
    (1, 98): "MT_EMBER_SUMMIT_PATH_1F",
    (1, 99): "MT_EMBER_SUMMIT_PATH_2F",
    (1, 100): "MT_EMBER_SUMMIT_PATH_3F",
    (1, 101): "MT_EMBER_SUMMIT",
    (1, 102): "MT_EMBER_RUBY_PATH_B5F",
    (1, 103): "MT_EMBER_RUBY_PATH_1F",
    (1, 104): "MT_EMBER_RUBY_PATH_B1F",
    (1, 105): "MT_EMBER_RUBY_PATH_B2F",
    (1, 106): "MT_EMBER_RUBY_PATH_B3F",
    (1, 107): "MT_EMBER_RUBY_PATH_B1F_STAIRS",
    (1, 108): "MT_EMBER_RUBY_PATH_B2F_STAIRS",
    (1, 109): "BERRY_FOREST",
    (1, 110): "ICEFALL_CAVE_ENTRANCE",
    (1, 111): "ICEFALL_CAVE_1F",
    (1, 112): "ICEFALL_CAVE_B1F",
    (1, 113): "ICEFALL_CAVE_BACK",
    (1, 114): "ROCKET_WAREHOUSE",
    (1, 115): "DOTTED_HOLE_1F",
    (1, 116): "DOTTED_HOLE_B1F",
    (1, 117): "DOTTED_HOLE_B2F",
    (1, 118): "DOTTED_HOLE_B3F",
    (1, 119): "DOTTED_HOLE_B4F",
    (1, 120): "DOTTED_HOLE_SAPPHIRE_ROOM",
    (1, 121): "PATTERN_BUSH",
    (1, 122): "ALTERING_CAVE",

    # Bank 2: Special Areas
    (2, 0): "NAVEL_ROCK_EXTERIOR",
    (2, 1): "TRAINER_TOWER_1F",
    (2, 2): "TRAINER_TOWER_2F",
    (2, 3): "TRAINER_TOWER_3F",
    (2, 4): "TRAINER_TOWER_4F",
    (2, 5): "TRAINER_TOWER_5F",
    (2, 6): "TRAINER_TOWER_6F",
    (2, 7): "TRAINER_TOWER_7F",
    (2, 8): "TRAINER_TOWER_8F",
    (2, 9): "TRAINER_TOWER_ROOF",
    (2, 10): "TRAINER_TOWER_LOBBY",
    (2, 11): "TRAINER_TOWER_ELEVATOR",
    (2, 12): "LOST_CAVE_ENTRANCE",
    (2, 13): "LOST_CAVE_ROOM1",
    (2, 14): "LOST_CAVE_ROOM2",
    (2, 15): "LOST_CAVE_ROOM3",
    (2, 16): "LOST_CAVE_ROOM4",
    (2, 17): "LOST_CAVE_ROOM5",
    (2, 18): "LOST_CAVE_ROOM6",
    (2, 19): "LOST_CAVE_ROOM7",
    (2, 20): "LOST_CAVE_ROOM8",
    (2, 21): "LOST_CAVE_ROOM9",
    (2, 22): "LOST_CAVE_ROOM10",
    (2, 23): "LOST_CAVE_ROOM11",
    (2, 24): "LOST_CAVE_ROOM12",
    (2, 25): "LOST_CAVE_ROOM13",
    (2, 26): "LOST_CAVE_ROOM14",
    (2, 27): "TANOBY_RUINS_MONEAN_CHAMBER",
    (2, 28): "TANOBY_RUINS_LIPTOO_CHAMBER",
    (2, 29): "TANOBY_RUINS_WEEPTH_CHAMBER",
    (2, 30): "TANOBY_RUINS_DILFORD_CHAMBER",
    (2, 31): "TANOBY_RUINS_SCUFIB_CHAMBER",
    (2, 32): "TANOBY_RUINS_RIXY_CHAMBER",
    (2, 33): "TANOBY_RUINS_VIAPOIS_CHAMBER",
    (2, 34): "DUNSPARCE_TUNNEL",
    (2, 35): "SEVAULT_CANYON_TANOBY_KEY",
    (2, 36): "NAVEL_ROCK_1F",
    (2, 37): "NAVEL_ROCK_SUMMIT",
    (2, 38): "NAVEL_ROCK_BASE",
    (2, 39): "NAVEL_ROCK_SUMMIT_PATH_2F",
    (2, 40): "NAVEL_ROCK_SUMMIT_PATH_3F",
    (2, 41): "NAVEL_ROCK_SUMMIT_PATH_4F",
    (2, 42): "NAVEL_ROCK_SUMMIT_PATH_5F",
    (2, 43): "NAVEL_ROCK_BASE_PATH_B1F",
    (2, 44): "NAVEL_ROCK_BASE_PATH_B2F",
    (2, 45): "NAVEL_ROCK_BASE_PATH_B3F",
    (2, 46): "NAVEL_ROCK_BASE_PATH_B4F",
    (2, 47): "NAVEL_ROCK_BASE_PATH_B5F",
    (2, 48): "NAVEL_ROCK_BASE_PATH_B6F",
    (2, 49): "NAVEL_ROCK_BASE_PATH_B7F",
    (2, 50): "NAVEL_ROCK_BASE_PATH_B8F",
    (2, 51): "NAVEL_ROCK_BASE_PATH_B9F",
    (2, 52): "NAVEL_ROCK_BASE_PATH_B10F",
    (2, 53): "NAVEL_ROCK_BASE_PATH_B11F",
    (2, 54): "NAVEL_ROCK_B1F",
    (2, 55): "NAVEL_ROCK_FORK",
    (2, 56): "BIRTH_ISLAND_EXTERIOR",
    (2, 57): "KINDLE_ROAD_EMBER_SPA",
    (2, 58): "BIRTH_ISLAND_HARBOR",
    (2, 59): "NAVEL_ROCK_HARBOR",

    # Bank 3: Towns and Routes
    (3, 0): "PALLET_TOWN",
    (3, 1): "VIRIDIAN_CITY",
    (3, 2): "PEWTER_CITY",
    (3, 3): "CERULEAN_CITY",
    (3, 4): "LAVENDER_TOWN",
    (3, 5): "VERMILION_CITY",
    (3, 6): "CELADON_CITY",
    (3, 7): "FUCHSIA_CITY",
    (3, 8): "CINNABAR_ISLAND",
    (3, 9): "INDIGO_PLATEAU_EXTERIOR",
    (3, 10): "SAFFRON_CITY",
    (3, 11): "SAFFRON_CITY_CONNECTION",
    (3, 12): "ONE_ISLAND",
    (3, 13): "TWO_ISLAND",
    (3, 14): "THREE_ISLAND",
    (3, 15): "FOUR_ISLAND",
    (3, 16): "FIVE_ISLAND",
    (3, 17): "SEVEN_ISLAND",
    (3, 18): "SIX_ISLAND",
    (3, 19): "ROUTE_1",
    (3, 20): "ROUTE_2",
    (3, 21): "ROUTE_3",
    (3, 22): "ROUTE_4",
    (3, 23): "ROUTE_5",
    (3, 24): "ROUTE_6",
    (3, 25): "ROUTE_7",
    (3, 26): "ROUTE_8",
    (3, 27): "ROUTE_9",
    (3, 28): "ROUTE_10",
    (3, 29): "ROUTE_11",
    (3, 30): "ROUTE_12",
    (3, 31): "ROUTE_13",
    (3, 32): "ROUTE_14",
    (3, 33): "ROUTE_15",
    (3, 34): "ROUTE_16",
    (3, 35): "ROUTE_17",
    (3, 36): "ROUTE_18",
    (3, 37): "ROUTE_19",
    (3, 38): "ROUTE_20",
    (3, 39): "ROUTE_21_NORTH",
    (3, 40): "ROUTE_21_SOUTH",
    (3, 41): "ROUTE_22",
    (3, 42): "ROUTE_23",
    (3, 43): "ROUTE_24",
    (3, 44): "ROUTE_25",
    (3, 45): "KINDLE_ROAD",
    (3, 46): "TREASURE_BEACH",
    (3, 47): "CAPE_BRINK",
    (3, 48): "BOND_BRIDGE",
    (3, 49): "THREE_ISLAND_PORT",
    (3, 54): "RESORT_GORGEOUS",
    (3, 55): "WATER_LABYRINTH",
    (3, 56): "FIVE_ISLAND_MEADOW",
    (3, 57): "MEMORIAL_PILLAR",
    (3, 58): "OUTCAST_ISLAND",
    (3, 59): "GREEN_PATH",
    (3, 60): "WATER_PATH",
    (3, 61): "RUIN_VALLEY",
    (3, 62): "TRAINER_TOWER_EXTERIOR",
    (3, 63): "SEVAULT_CANYON_ENTRANCE",
    (3, 64): "SEVAULT_CANYON",
    (3, 65): "TANOBY_RUINS",

    # Bank 4: Indoor Pallet Town
    (4, 0): "PLAYERS_HOUSE_1F",
    (4, 1): "PLAYERS_HOUSE_2F",
    (4, 2): "RIVALS_HOUSE",
    (4, 3): "OAKS_LAB",

    # Bank 5: Indoor Viridian
    (5, 0): "VIRIDIAN_HOUSE",
    (5, 1): "VIRIDIAN_GYM",
    (5, 2): "VIRIDIAN_SCHOOL",
    (5, 3): "VIRIDIAN_MART",
    (5, 4): "VIRIDIAN_POKECENTER_1F",
    (5, 5): "VIRIDIAN_POKECENTER_2F",

    # Bank 6: Indoor Pewter
    (6, 0): "PEWTER_MUSEUM_1F",
    (6, 1): "PEWTER_MUSEUM_2F",
    (6, 2): "PEWTER_GYM",
    (6, 3): "PEWTER_MART",
    (6, 4): "PEWTER_HOUSE1",
    (6, 5): "PEWTER_POKECENTER_1F",
    (6, 6): "PEWTER_POKECENTER_2F",
    (6, 7): "PEWTER_HOUSE2",

    # Bank 7: Indoor Cerulean
    (7, 0): "CERULEAN_HOUSE1",
    (7, 1): "CERULEAN_HOUSE2",
    (7, 2): "CERULEAN_HOUSE3",
    (7, 3): "CERULEAN_POKECENTER_1F",
    (7, 4): "CERULEAN_POKECENTER_2F",
    (7, 5): "CERULEAN_GYM",
    (7, 6): "CERULEAN_BIKE_SHOP",
    (7, 7): "CERULEAN_MART",
    (7, 8): "CERULEAN_HOUSE4",
    (7, 9): "CERULEAN_HOUSE5",

    # Bank 8: Indoor Lavender
    (8, 0): "LAVENDER_POKECENTER_1F",
    (8, 1): "LAVENDER_POKECENTER_2F",
    (8, 2): "LAVENDER_VOLUNTEER_POKEMON_HOUSE",
    (8, 3): "LAVENDER_HOUSE1",
    (8, 4): "LAVENDER_HOUSE2",
    (8, 5): "LAVENDER_MART",

    # Bank 9: Indoor Vermilion
    (9, 0): "VERMILION_HOUSE1",
    (9, 1): "VERMILION_POKECENTER_1F",
    (9, 2): "VERMILION_POKECENTER_2F",
    (9, 3): "VERMILION_POKEMON_FAN_CLUB",
    (9, 4): "VERMILION_HOUSE2",
    (9, 5): "VERMILION_MART",
    (9, 6): "VERMILION_GYM",
    (9, 7): "VERMILION_HOUSE3",

    # Bank 10: Indoor Celadon
    (10, 0): "CELADON_DEPT_STORE_1F",
    (10, 1): "CELADON_DEPT_STORE_2F",
    (10, 2): "CELADON_DEPT_STORE_3F",
    (10, 3): "CELADON_DEPT_STORE_4F",
    (10, 4): "CELADON_DEPT_STORE_5F",
    (10, 5): "CELADON_DEPT_STORE_ROOF",
    (10, 6): "CELADON_DEPT_STORE_ELEVATOR",
    (10, 7): "CELADON_CONDOMINIUMS_1F",
    (10, 8): "CELADON_CONDOMINIUMS_2F",
    (10, 9): "CELADON_CONDOMINIUMS_3F",
    (10, 10): "CELADON_CONDOMINIUMS_ROOF",
    (10, 11): "CELADON_CONDOMINIUMS_ROOF_ROOM",
    (10, 12): "CELADON_POKECENTER_1F",
    (10, 13): "CELADON_POKECENTER_2F",
    (10, 14): "CELADON_GAME_CORNER",
    (10, 15): "CELADON_GAME_CORNER_PRIZE_ROOM",
    (10, 16): "CELADON_GYM",
    (10, 17): "CELADON_RESTAURANT",
    (10, 18): "CELADON_HOUSE1",
    (10, 19): "CELADON_HOTEL",

    # Bank 11: Indoor Fuchsia
    (11, 0): "FUCHSIA_SAFARI_ZONE_ENTRANCE",
    (11, 1): "FUCHSIA_MART",
    (11, 2): "FUCHSIA_SAFARI_ZONE_OFFICE",
    (11, 3): "FUCHSIA_GYM",
    (11, 4): "FUCHSIA_HOUSE1",
    (11, 5): "FUCHSIA_POKECENTER_1F",
    (11, 6): "FUCHSIA_POKECENTER_2F",
    (11, 7): "FUCHSIA_WARDENS_HOUSE",
    (11, 8): "FUCHSIA_HOUSE2",
    (11, 9): "FUCHSIA_HOUSE3",

    # Bank 12: Indoor Cinnabar
    (12, 0): "CINNABAR_GYM",
    (12, 1): "CINNABAR_POKEMON_LAB_ENTRANCE",
    (12, 2): "CINNABAR_POKEMON_LAB_LOUNGE",
    (12, 3): "CINNABAR_POKEMON_LAB_RESEARCH",
    (12, 4): "CINNABAR_POKEMON_LAB_EXPERIMENT",
    (12, 5): "CINNABAR_POKECENTER_1F",
    (12, 6): "CINNABAR_POKECENTER_2F",
    (12, 7): "CINNABAR_MART",

    # Bank 13: Indoor Indigo Plateau
    (13, 0): "INDIGO_PLATEAU_POKECENTER_1F",
    (13, 1): "INDIGO_PLATEAU_POKECENTER_2F",

    # Bank 14: Indoor Saffron
    (14, 0): "SAFFRON_COPYCATS_HOUSE_1F",
    (14, 1): "SAFFRON_COPYCATS_HOUSE_2F",
    (14, 2): "SAFFRON_DOJO",
    (14, 3): "SAFFRON_GYM",
    (14, 4): "SAFFRON_HOUSE",
    (14, 5): "SAFFRON_MART",
    (14, 6): "SAFFRON_POKECENTER_1F",
    (14, 7): "SAFFRON_POKECENTER_2F",
    (14, 8): "SAFFRON_MR_PSYCHICS_HOUSE",
    (14, 9): "SAFFRON_POKEMON_TRAINER_FAN_CLUB",

    # Bank 15: Indoor Route 2
    (15, 0): "ROUTE_2_VIRIDIAN_FOREST_SOUTH_ENTRANCE",
    (15, 1): "ROUTE_2_HOUSE",
    (15, 2): "ROUTE_2_EAST_BUILDING",
    (15, 3): "ROUTE_2_VIRIDIAN_FOREST_NORTH_ENTRANCE",

    # Bank 16: Indoor Route 4
    (16, 0): "ROUTE_4_POKECENTER_1F",
    (16, 1): "ROUTE_4_POKECENTER_2F",

    # Bank 17: Indoor Route 5
    (17, 0): "ROUTE_5_POKEMON_DAYCARE",
    (17, 1): "ROUTE_5_SOUTH_ENTRANCE",

    # Bank 18: Indoor Route 6
    (18, 0): "ROUTE_6_NORTH_ENTRANCE",
    (18, 1): "ROUTE_6_UNUSED_HOUSE",

    # Bank 19: Indoor Route 7
    (19, 0): "ROUTE_7_EAST_ENTRANCE",

    # Bank 20: Indoor Route 8
    (20, 0): "ROUTE_8_WEST_ENTRANCE",

    # Bank 21: Indoor Route 10
    (21, 0): "ROUTE_10_POKECENTER_1F",
    (21, 1): "ROUTE_10_POKECENTER_2F",

    # Bank 22: Indoor Route 11
    (22, 0): "ROUTE_11_EAST_ENTRANCE_1F",
    (22, 1): "ROUTE_11_EAST_ENTRANCE_2F",

    # Bank 23: Indoor Route 12
    (23, 0): "ROUTE_12_NORTH_ENTRANCE_1F",
    (23, 1): "ROUTE_12_NORTH_ENTRANCE_2F",
    (23, 2): "ROUTE_12_FISHING_HOUSE",

    # Bank 24: Indoor Route 15
    (24, 0): "ROUTE_15_WEST_ENTRANCE_1F",
    (24, 1): "ROUTE_15_WEST_ENTRANCE_2F",

    # Bank 25: Indoor Route 16
    (25, 0): "ROUTE_16_HOUSE",
    (25, 1): "ROUTE_16_NORTH_ENTRANCE_1F",
    (25, 2): "ROUTE_16_NORTH_ENTRANCE_2F",

    # Bank 26: Indoor Route 18
    (26, 0): "ROUTE_18_EAST_ENTRANCE_1F",
    (26, 1): "ROUTE_18_EAST_ENTRANCE_2F",

    # Bank 27: Indoor Route 19
    (27, 0): "ROUTE_19_UNUSED_HOUSE",

    # Bank 28: Indoor Route 22
    (28, 0): "ROUTE_22_NORTH_ENTRANCE",

    # Bank 29: Indoor Route 23
    (29, 0): "ROUTE_23_UNUSED_HOUSE",

    # Bank 30: Indoor Route 25
    (30, 0): "ROUTE_25_SEA_COTTAGE",

    # Bank 31: Indoor Seven Island
    (31, 0): "SEVEN_ISLAND_HOUSE_ROOM1",
    (31, 1): "SEVEN_ISLAND_HOUSE_ROOM2",
    (31, 2): "SEVEN_ISLAND_MART",
    (31, 3): "SEVEN_ISLAND_POKECENTER_1F",
    (31, 4): "SEVEN_ISLAND_POKECENTER_2F",
    (31, 5): "SEVEN_ISLAND_UNUSED_HOUSE",
    (31, 6): "SEVEN_ISLAND_HARBOR",

    # Bank 32: Indoor One Island
    (32, 0): "ONE_ISLAND_POKECENTER_1F",
    (32, 1): "ONE_ISLAND_POKECENTER_2F",
    (32, 2): "ONE_ISLAND_HOUSE1",
    (32, 3): "ONE_ISLAND_HOUSE2",
    (32, 4): "ONE_ISLAND_HARBOR",

    # Bank 33: Indoor Two Island
    (33, 0): "TWO_ISLAND_JOYFUL_GAME_CORNER",
    (33, 1): "TWO_ISLAND_HOUSE",
    (33, 2): "TWO_ISLAND_POKECENTER_1F",
    (33, 3): "TWO_ISLAND_POKECENTER_2F",
    (33, 4): "TWO_ISLAND_HARBOR",

    # Bank 34: Indoor Three Island
    (34, 0): "THREE_ISLAND_HOUSE1",
    (34, 1): "THREE_ISLAND_POKECENTER_1F",
    (34, 2): "THREE_ISLAND_POKECENTER_2F",
    (34, 3): "THREE_ISLAND_MART",
    (34, 4): "THREE_ISLAND_HOUSE2",
    (34, 5): "THREE_ISLAND_HOUSE3",
    (34, 6): "THREE_ISLAND_HOUSE4",
    (34, 7): "THREE_ISLAND_HOUSE5",

    # Bank 35: Indoor Four Island
    (35, 0): "FOUR_ISLAND_POKEMON_DAYCARE",
    (35, 1): "FOUR_ISLAND_POKECENTER_1F",
    (35, 2): "FOUR_ISLAND_POKECENTER_2F",
    (35, 3): "FOUR_ISLAND_HOUSE1",
    (35, 4): "FOUR_ISLAND_LORELEIS_HOUSE",
    (35, 5): "FOUR_ISLAND_HARBOR",
    (35, 6): "FOUR_ISLAND_HOUSE2",
    (35, 7): "FOUR_ISLAND_MART",

    # Bank 36: Indoor Five Island
    (36, 0): "FIVE_ISLAND_POKECENTER_1F",
    (36, 1): "FIVE_ISLAND_POKECENTER_2F",
    (36, 2): "FIVE_ISLAND_HARBOR",
    (36, 3): "FIVE_ISLAND_HOUSE1",
    (36, 4): "FIVE_ISLAND_HOUSE2",

    # Bank 37: Indoor Six Island
    (37, 0): "SIX_ISLAND_POKECENTER_1F",
    (37, 1): "SIX_ISLAND_POKECENTER_2F",
    (37, 2): "SIX_ISLAND_HARBOR",
    (37, 3): "SIX_ISLAND_HOUSE",
    (37, 4): "SIX_ISLAND_MART",

    # Bank 38: Indoor Three Island Route
    (38, 0): "THREE_ISLAND_HARBOR",

    # Bank 39: Indoor Five Island Route
    (39, 0): "RESORT_GORGEOUS_HOUSE",

    # Bank 40: Indoor Two Island Route
    (40, 0): "CAPE_BRINK_HOUSE",

    # Bank 41: Indoor Six Island Route
    (41, 0): "WATER_PATH_HOUSE1",
    (41, 1): "WATER_PATH_HOUSE2",

    # Bank 42: Indoor Seven Island Route
    (42, 0): "SEVAULT_CANYON_HOUSE",
}

# Gen 3 Move IDs (index -> name)
MOVE_NAMES = {
    0: "(none)", 1: "Pound", 2: "Karate Chop", 3: "Double Slap", 4: "Comet Punch",
    5: "Mega Punch", 6: "Pay Day", 7: "Fire Punch", 8: "Ice Punch", 9: "Thunder Punch",
    10: "Scratch", 11: "Vice Grip", 12: "Guillotine", 13: "Razor Wind", 14: "Swords Dance",
    15: "Cut", 16: "Gust", 17: "Wing Attack", 18: "Whirlwind", 19: "Fly",
    20: "Bind", 21: "Slam", 22: "Vine Whip", 23: "Stomp", 24: "Double Kick",
    25: "Mega Kick", 26: "Jump Kick", 27: "Rolling Kick", 28: "Sand Attack", 29: "Headbutt",
    30: "Horn Attack", 31: "Fury Attack", 32: "Horn Drill", 33: "Tackle", 34: "Body Slam",
    35: "Wrap", 36: "Take Down", 37: "Thrash", 38: "Double-Edge", 39: "Tail Whip",
    40: "Poison Sting", 41: "Twineedle", 42: "Pin Missile", 43: "Leer", 44: "Bite",
    45: "Growl", 46: "Roar", 47: "Sing", 48: "Supersonic", 49: "Sonic Boom",
    50: "Disable", 51: "Acid", 52: "Ember", 53: "Flamethrower", 54: "Mist",
    55: "Water Gun", 56: "Hydro Pump", 57: "Surf", 58: "Ice Beam", 59: "Blizzard",
    60: "Psybeam", 61: "Bubble Beam", 62: "Aurora Beam", 63: "Hyper Beam", 64: "Peck",
    65: "Drill Peck", 66: "Submission", 67: "Low Kick", 68: "Counter", 69: "Seismic Toss",
    70: "Strength", 71: "Absorb", 72: "Mega Drain", 73: "Leech Seed", 74: "Growth",
    75: "Razor Leaf", 76: "Solar Beam", 77: "Poison Powder", 78: "Stun Spore", 79: "Sleep Powder",
    80: "Petal Dance", 81: "String Shot", 82: "Dragon Rage", 83: "Fire Spin", 84: "Thunder Shock",
    85: "Thunderbolt", 86: "Thunder Wave", 87: "Thunder", 88: "Rock Throw", 89: "Earthquake",
    90: "Fissure", 91: "Dig", 92: "Toxic", 93: "Confusion", 94: "Psychic",
    95: "Hypnosis", 96: "Meditate", 97: "Agility", 98: "Quick Attack", 99: "Rage",
    100: "Teleport", 101: "Night Shade", 102: "Mimic", 103: "Screech", 104: "Double Team",
    105: "Recover", 106: "Harden", 107: "Minimize", 108: "Smokescreen", 109: "Confuse Ray",
    110: "Withdraw", 111: "Defense Curl", 112: "Barrier", 113: "Light Screen", 114: "Haze",
    115: "Reflect", 116: "Focus Energy", 117: "Bide", 118: "Metronome", 119: "Mirror Move",
    120: "Self-Destruct", 121: "Egg Bomb", 122: "Lick", 123: "Smog", 124: "Sludge",
    125: "Bone Club", 126: "Fire Blast", 127: "Waterfall", 128: "Clamp", 129: "Swift",
    130: "Skull Bash", 131: "Spike Cannon", 132: "Constrict", 133: "Amnesia", 134: "Kinesis",
    135: "Soft-Boiled", 136: "High Jump Kick", 137: "Glare", 138: "Dream Eater", 139: "Poison Gas",
    140: "Barrage", 141: "Leech Life", 142: "Lovely Kiss", 143: "Sky Attack", 144: "Transform",
    145: "Bubble", 146: "Dizzy Punch", 147: "Spore", 148: "Flash", 149: "Psywave",
    150: "Splash", 151: "Acid Armor", 152: "Crabhammer", 153: "Explosion", 154: "Fury Swipes",
    155: "Bonemerang", 156: "Rest", 157: "Rock Slide", 158: "Hyper Fang", 159: "Sharpen",
    160: "Conversion", 161: "Tri Attack", 162: "Super Fang", 163: "Slash", 164: "Substitute",
    165: "Struggle", 166: "Sketch", 167: "Triple Kick", 168: "Thief", 169: "Spider Web",
    170: "Mind Reader", 171: "Nightmare", 172: "Flame Wheel", 173: "Snore", 174: "Curse",
    175: "Flail", 176: "Conversion 2", 177: "Aeroblast", 178: "Cotton Spore", 179: "Reversal",
    180: "Spite", 181: "Powder Snow", 182: "Protect", 183: "Mach Punch", 184: "Scary Face",
    185: "Faint Attack", 186: "Sweet Kiss", 187: "Belly Drum", 188: "Sludge Bomb", 189: "Mud-Slap",
    190: "Octazooka", 191: "Spikes", 192: "Zap Cannon", 193: "Foresight", 194: "Destiny Bond",
    195: "Perish Song", 196: "Icy Wind", 197: "Detect", 198: "Bone Rush", 199: "Lock-On",
    200: "Outrage", 201: "Sandstorm", 202: "Giga Drain", 203: "Endure", 204: "Charm",
    205: "Rollout", 206: "False Swipe", 207: "Swagger", 208: "Milk Drink", 209: "Spark",
    210: "Fury Cutter", 211: "Steel Wing", 212: "Mean Look", 213: "Attract", 214: "Sleep Talk",
    215: "Heal Bell", 216: "Return", 217: "Present", 218: "Frustration", 219: "Safeguard",
    220: "Pain Split", 221: "Sacred Fire", 222: "Magnitude", 223: "Dynamic Punch", 224: "Megahorn",
    225: "Dragon Breath", 226: "Baton Pass", 227: "Encore", 228: "Pursuit", 229: "Rapid Spin",
    230: "Sweet Scent", 231: "Iron Tail", 232: "Metal Claw", 233: "Vital Throw", 234: "Morning Sun",
    235: "Synthesis", 236: "Moonlight", 237: "Hidden Power", 238: "Cross Chop", 239: "Twister",
    240: "Rain Dance", 241: "Sunny Day", 242: "Crunch", 243: "Mirror Coat", 244: "Psych Up",
    245: "Extreme Speed", 246: "Ancient Power", 247: "Shadow Ball", 248: "Future Sight", 249: "Rock Smash",
    250: "Whirlpool", 251: "Beat Up", 252: "Fake Out", 253: "Uproar", 254: "Stockpile",
    255: "Spit Up", 256: "Swallow", 257: "Heat Wave", 258: "Hail", 259: "Torment",
    260: "Flatter", 261: "Will-O-Wisp", 262: "Memento", 263: "Facade", 264: "Focus Punch",
    265: "Smelling Salts", 266: "Follow Me", 267: "Nature Power", 268: "Charge", 269: "Taunt",
    270: "Helping Hand", 271: "Trick", 272: "Role Play", 273: "Wish", 274: "Assist",
    275: "Ingrain", 276: "Superpower", 277: "Magic Coat", 278: "Recycle", 279: "Revenge",
    280: "Brick Break", 281: "Yawn", 282: "Knock Off", 283: "Endeavor", 284: "Eruption",
    285: "Skill Swap", 286: "Imprison", 287: "Refresh", 288: "Grudge", 289: "Snatch",
    290: "Secret Power", 291: "Dive", 292: "Arm Thrust", 293: "Camouflage", 294: "Tail Glow",
    295: "Luster Purge", 296: "Mist Ball", 297: "Feather Dance", 298: "Teeter Dance", 299: "Blaze Kick",
    300: "Mud Sport", 301: "Ice Ball", 302: "Needle Arm", 303: "Slack Off", 304: "Hyper Voice",
    305: "Poison Fang", 306: "Crush Claw", 307: "Blast Burn", 308: "Hydro Cannon", 309: "Meteor Mash",
    310: "Astonish", 311: "Weather Ball", 312: "Aromatherapy", 313: "Fake Tears", 314: "Air Cutter",
    315: "Overheat", 316: "Odor Sleuth", 317: "Rock Tomb", 318: "Silver Wind", 319: "Metal Sound",
    320: "Grass Whistle", 321: "Tickle", 322: "Cosmic Power", 323: "Water Spout", 324: "Signal Beam",
    325: "Shadow Punch", 326: "Extrasensory", 327: "Sky Uppercut", 328: "Sand Tomb", 329: "Sheer Cold",
    330: "Muddy Water", 331: "Bullet Seed", 332: "Aerial Ace", 333: "Icicle Spear", 334: "Iron Defense",
    335: "Block", 336: "Howl", 337: "Dragon Claw", 338: "Frenzy Plant", 339: "Bulk Up",
    340: "Bounce", 341: "Mud Shot", 342: "Poison Tail", 343: "Covet", 344: "Volt Tackle",
    345: "Magical Leaf", 346: "Water Sport", 347: "Calm Mind", 348: "Leaf Blade", 349: "Dragon Dance",
    350: "Rock Blast", 351: "Shock Wave", 352: "Water Pulse", 353: "Doom Desire", 354: "Psycho Boost",
}

# --- Bag pocket layout (offsets from SaveBlock1 pointer) ---
# Each item slot = 4 bytes: u16 itemId + u16 quantity
BAG_POCKETS = [
    ("Items", 0x310, 42),
    ("Key Items", 0x3B8, 30),
    ("Poke Balls", 0x430, 16),
    ("TMs & HMs", 0x464, 64),
    ("Berries", 0x564, 46),
]

BAG_POCKET_NAMES = {0: "Items", 1: "Key Items", 2: "Poke Balls", 3: "TMs & HMs", 4: "Berries"}

# Gen 3 item IDs (FireRed/LeafGreen)
ITEM_NAMES = {
    0: "(none)",
    # Poke Balls
    1: "Master Ball", 2: "Ultra Ball", 3: "Great Ball", 4: "Poke Ball",
    5: "Safari Ball", 6: "Net Ball", 7: "Dive Ball", 8: "Nest Ball",
    9: "Repeat Ball", 10: "Timer Ball", 11: "Luxury Ball", 12: "Premier Ball",
    # Medicine / Recovery
    13: "Potion", 14: "Antidote", 15: "Burn Heal", 16: "Ice Heal",
    17: "Awakening", 18: "Parlyz Heal", 19: "Full Restore", 20: "Max Potion",
    21: "Hyper Potion", 22: "Super Potion", 23: "Full Heal", 24: "Revive",
    25: "Max Revive", 26: "Fresh Water", 27: "Soda Pop", 28: "Lemonade",
    29: "Moomoo Milk", 30: "Energy Powder", 31: "Energy Root", 32: "Heal Powder",
    33: "Revival Herb", 34: "Ether", 35: "Max Ether", 36: "Elixir", 37: "Max Elixir",
    38: "Lava Cookie", 39: "Blue Flute", 40: "Yellow Flute", 41: "Red Flute",
    42: "Black Flute", 43: "White Flute", 44: "Berry Juice",
    # Stat boosters (permanent)
    45: "Sacred Ash", 46: "Shoal Salt", 47: "Shoal Shell",
    48: "Red Shard", 49: "Blue Shard", 50: "Yellow Shard", 51: "Green Shard",
    # Battle items
    63: "HP Up", 64: "Protein", 65: "Iron", 66: "Carbos",
    67: "Calcium", 68: "Rare Candy", 69: "PP Up", 70: "Zinc",
    71: "PP Max",
    73: "Guard Spec.", 74: "Dire Hit", 75: "X Attack", 76: "X Defend",
    77: "X Speed", 78: "X Accuracy", 79: "X Special", 80: "Poke Doll",
    81: "Fluffy Tail",
    83: "Fire Stone", 84: "Max Repel", 85: "Escape Rope", 86: "Repel",
    93: "Sun Stone", 94: "Moon Stone", 95: "Fire Stone", 96: "Thunder Stone",
    97: "Water Stone", 98: "Leaf Stone",
    # Held items
    179: "King's Rock", 180: "Silver Powder", 181: "Amulet Coin",
    183: "Exp. Share", 185: "Soothe Bell",
    187: "Metal Coat", 189: "Dragon Scale",
    191: "Leftovers",
    # Evolution items
    196: "Dragon Scale",
    # Repels and Escape
    83: "Super Repel", 84: "Max Repel", 85: "Escape Rope", 86: "Repel",
    # Key Items
    259: "Bicycle", 260: "Pokedex",
    261: "Old Rod", 262: "Good Rod", 263: "Super Rod",
    264: "S.S. Ticket", 265: "Contest Pass",
    266: "Wailmer Pail",
    269: "Basement Key", 270: "Acro Bike", 271: "Pokeblock Case",
    272: "Letter", 273: "Eon Ticket", 274: "Red Orb", 275: "Blue Orb",
    276: "Scanner", 277: "Go-Goggles", 278: "Meteorite",
    279: "Rm. 1 Key", 280: "Rm. 2 Key", 281: "Rm. 4 Key", 282: "Rm. 6 Key",
    283: "Storage Key", 284: "Root Fossil", 285: "Claw Fossil",
    286: "Devon Scope",
    # FRLG Key Items
    349: "Oaks Parcel", 350: "Poke Flute", 351: "Secret Key",
    352: "Bike Voucher", 353: "Gold Teeth", 354: "Old Amber",
    355: "Card Key", 356: "Lift Key", 357: "Helix Fossil", 358: "Dome Fossil",
    359: "Silph Scope", 360: "SS Ticket", 361: "Rain Dance",
    362: "Mystic Ticket", 363: "Aurora Ticket", 364: "Powder Jar",
    365: "Ruby", 366: "Sapphire",
    369: "Town Map", 370: "VS Seeker", 371: "Fame Checker",
    372: "TM Case", 373: "Berry Pouch", 374: "Teachy TV",
    375: "Tri-Pass", 376: "Rainbow Pass", 377: "Tea", 378: "Mystic Ticket",
    # TMs
    289: "TM01", 290: "TM02", 291: "TM03", 292: "TM04", 293: "TM05",
    294: "TM06", 295: "TM07", 296: "TM08", 297: "TM09", 298: "TM10",
    299: "TM11", 300: "TM12", 301: "TM13", 302: "TM14", 303: "TM15",
    304: "TM16", 305: "TM17", 306: "TM18", 307: "TM19", 308: "TM20",
    309: "TM21", 310: "TM22", 311: "TM23", 312: "TM24", 313: "TM25",
    314: "TM26", 315: "TM27", 316: "TM28", 317: "TM29", 318: "TM30",
    319: "TM31", 320: "TM32", 321: "TM33", 322: "TM34", 323: "TM35",
    324: "TM36", 325: "TM37", 326: "TM38", 327: "TM39", 328: "TM40",
    329: "TM41", 330: "TM42", 331: "TM43", 332: "TM44", 333: "TM45",
    334: "TM46", 335: "TM47", 336: "TM48", 337: "TM49", 338: "TM50",
    # HMs
    339: "HM01 Cut", 340: "HM02 Fly", 341: "HM03 Surf", 342: "HM04 Strength",
    343: "HM05 Flash", 344: "HM06 Rock Smash", 345: "HM07 Waterfall",
    # Berries
    133: "Cheri Berry", 134: "Chesto Berry", 135: "Pecha Berry",
    136: "Rawst Berry", 137: "Aspear Berry", 138: "Leppa Berry",
    139: "Oran Berry", 140: "Persim Berry", 141: "Lum Berry", 142: "Sitrus Berry",
    143: "Figy Berry", 144: "Wiki Berry", 145: "Mago Berry",
    146: "Aguav Berry", 147: "Iapapa Berry",
    175: "Liechi Berry", 176: "Ganlon Berry", 177: "Salac Berry", 178: "Petaya Berry",
}

# --- Story flag system (SaveBlock1 flag bit array) ---
# Flags are stored as a bit array at SaveBlock1 + 0xEE0
# To check flag N: read byte at SB1 + 0xEE0 + (N // 8), test bit (N % 8)
SB1_FLAGS_OFFSET = 0xEE0

# System flags (0x800+)
FLAG_SYS_POKEMON_GET = 0x800        # player has at least one pokemon
FLAG_SYS_POKEDEX_GET = 0x801        # player has pokedex
FLAG_SYS_POKENAV_GET = 0x802        # has pokenav (unused in FRLG)
FLAG_SYS_NATIONAL_DEX = 0x804       # national dex unlocked

# Badge flags
FLAG_BADGE01_GET = 0x820  # Boulder Badge (Brock)
FLAG_BADGE02_GET = 0x821  # Cascade Badge (Misty)
FLAG_BADGE03_GET = 0x822  # Thunder Badge (Surge)
FLAG_BADGE04_GET = 0x823  # Rainbow Badge (Erika)
FLAG_BADGE05_GET = 0x824  # Soul Badge (Koga)
FLAG_BADGE06_GET = 0x825  # Marsh Badge (Sabrina)
FLAG_BADGE07_GET = 0x826  # Volcano Badge (Blaine)
FLAG_BADGE08_GET = 0x827  # Earth Badge (Giovanni)

# Story progression flags (from pokefirered decomp)
FLAG_GOT_STARTER = 0x102            # chose starter in Oak's lab
FLAG_GOT_OAKS_PARCEL = 0x121       # picked up parcel from Viridian Mart
FLAG_DELIVERED_OAKS_PARCEL = 0x122  # gave parcel to Oak
FLAG_GOT_POKEDEX = 0x123            # received pokedex from Oak
FLAG_HIDE_ROUTE22_RIVAL = 0x159     # rival on route 22 beaten/gone

# Trainer defeated flags (set when trainer beaten, prevents rematches)
FLAG_TRAINER_BROCK = 0x149          # Brock defeated

# Item IDs relevant to progression detection
ITEM_OAKS_PARCEL = 349
