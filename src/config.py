from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = APP_ROOT / "data" / "raw" / "mushrooms.csv"
TARGET_COLUMN = "type"
POSITIVE_LABEL = "p"
NEGATIVE_LABEL = "e"
RANDOM_STATE = 42

CLASS_LABELS = {
    0: "Edible",
    1: "Poisonous",
}

BUSINESS_FRAME = {
    "primary_risk": "False-safe prediction",
    "definition": "A poisonous mushroom classified as edible.",
    "policy": "Prioritize poisonous recall and lower the decision threshold when the use case is safety-sensitive.",
}

FEATURE_LABELS = {
    "cap_shape": "Cap shape",
    "cap_surface": "Cap surface",
    "cap_color": "Cap color",
    "bruises": "Bruises",
    "odor": "Odor",
    "gill_attachment": "Gill attachment",
    "gill_spacing": "Gill spacing",
    "gill_size": "Gill size",
    "gill_color": "Gill color",
    "stalk_shape": "Stalk shape",
    "stalk_root": "Stalk root",
    "stalk_surface_above_ring": "Stalk surface above ring",
    "stalk_surface_below_ring": "Stalk surface below ring",
    "stalk_color_above_ring": "Stalk color above ring",
    "stalk_color_below_ring": "Stalk color below ring",
    "veil_type": "Veil type",
    "veil_color": "Veil color",
    "ring_number": "Ring number",
    "ring_type": "Ring type",
    "spore_print_color": "Spore print color",
    "population": "Population",
    "habitat": "Habitat",
}
