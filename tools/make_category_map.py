"""Write yumeko_category_map.json, the reviewed tag list behind Yumeko's autocomplete and Phase 3. Edit the lists, run again."""
import json
from pathlib import Path

G = {
    ("shot", "shot"): "portrait close-up upper_body cowboy_shot full_body",
    ("pose", "view"): "from_side from_behind from_above from_below three_quarter_view",
    ("pose", "body_position"): "standing sitting walking kneeling leaning_forward",
    ("expression", "expression"): "smile light_smile open_mouth closed_eyes looking_to_the_side laughing serious",
    ("setting", "setting"): "library forest beach city_street garden cafe rooftop castle_interior bedroom_interior classroom mountain snowfield",
    ("lighting", "lighting"): "daytime sunset night moonlight candlelight overcast",
}
EXTRA = {
    "appearance": "long_hair short_hair medium_hair black_hair white_hair blonde_hair brown_hair red_hair blue_hair pink_hair purple_hair streaked_hair braid ponytail twintails bangs blue_eyes red_eyes green_eyes purple_eyes brown_eyes pale_skin tattoo mole_under_eye",
    "clothing": "dress white_dress black_dress school_uniform jacket black_jacket cape corset skirt shirt sweater kimono boots high_heels gloves thighhighs",
    "prop": "book umbrella tea_cup staff sword flower phone",
    "action": "holding_book holding_umbrella arms_crossed hand_on_hip waving reading running",
    "background": "simple_background white_background cityscape sky clouds stars",
    "composition": "centered_composition wide_shot dutch_angle",
    "style": "cyberpunk fantasy watercolor",
}
tags = []
for (category, facet), names in G.items():
    tags += [{"tag": t, "yumeko_category": category, "facet": facet} for t in names.split()]
for category, names in EXTRA.items():
    tags += [{"tag": t, "yumeko_category": category} for t in names.split()]
out = {"schema_version": 1, "single_choice": ["shot", "view", "body_position", "lighting"],
       "no_pose_shots": ["portrait", "close-up"], "aliases": {"closeup": "close-up", "braided_hair": "braid", "tea": "tea_cup"}, "tags": tags}
Path(__file__).parent.parent.joinpath("yumeko_category_map.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(len(tags))
