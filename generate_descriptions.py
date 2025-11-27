import csv
import random
from pathlib import Path

"""
Generate text descriptions for five classes:
mandarin, potato, onion, apple, banana.

- Descriptions do NOT mention the class names.
- Output: descriptions.csv with columns: id, label, description
"""

desc_per_class = 600
output_dir = Path("data/processed/descriptions.csv")

# shared words that can be reused across categories
AROMA_MILD = ["mild aroma", "subtle scent", "light fragrance"]
AROMA_STRONG = ["strong smell", "sharp aroma", "pungent scent", "intense odor"]
FLAVOR_SWEET = ["sweet", "pleasantly sweet", "mildly sweet", "sweet and refreshing", "gentle sweetness"]
FLAVOR_TANGY = ["tangy", "bright and tangy", "sweet and tangy", "slightly tart", "zesty and refreshing"]
FLAVOR_STARCHY = ["neutral, starchy taste", "mild flavor", "earthy, starchy taste", "subtle earthy flavor"]
FLAVOR_SAVORY = ["sharp, savory taste", "strong, savory flavor", "intense, biting flavor"]

# category specific templates
TEMPLATE = {
    "mandarin": {
        "templates": [
            "A {size} {shape} citrus fruit with {peel_color} skin and {segment_texture} segments inside.",
            "A {size} {shape} snack with {peel_color} peel and a {flavor_profile} taste.",
            "A {shape} winter fruit with {peel_color} rind, {segment_texture} segments, and a {flavor_profile} flavor.",
            "This {size} citrus fruit has {peel_color} skin, {segment_texture} wedges, and a {aroma_profile}.",
            "A {size} fruit with {peel_color} peel that comes off easily by hand and reveals {segment_texture} pieces.",
            "A {size} {shape} citrusy treat with {peel_color} skin and a {flavor_profile} taste, giving off a {aroma_profile}."
        ],
        "values": {
            "size": ["small", "medium sized", "palm sized"],
            "shape": ["round", "slightly flattened"],
            "peel_color": ["bright orange", "deep orange", "light orange", "greenish orange", "greenish yellow", "green"],
            "segment_texture": [
                "soft, juicy", "tender, pulpy", "plump, juicy",
                "delicate, juicy"
            ],
            "flavor_profile": FLAVOR_TANGY + FLAVOR_SWEET,
            "aroma_profile": [
                "fresh citrus aroma", "bright citrus scent",
                "refreshing, zesty smell", "zesty smell", "refresing smell"
            ],
        },
    },

    "potato": {
        "templates": [
            "A {size} {shape} tuber with {skin_color} skin and {flesh_texture} flesh inside.",
            "An {size} {shape} root vegetable with {skin_texture} {skin_color} surface and {flesh_texture} interior.",
            "A {shape} starchy vegetable covered in {skin_texture} {skin_color} skin and a {flesh_texture} center.",
            "A {size} {shape} item with {skin_color} peel, {skin_texture} patches, and a {flavor_profile}.",
            "A {shape} tuber that feels {weight_feel} in the hand, with {skin_color} skin and {flesh_texture} insides."
        ],
        "values": {
            "size": ["medium sized", "large", "plump"],
            "shape": ["oval", "oblong", "round", "slightly irregular"],
            "skin_color": ["light brown", "dull brown", "tan", "dark brown"],
            "skin_texture": ["rough", "slightly rough", "dusty", "patchy", "bumpy"],
            "flesh_texture": [
                "firm, dense", "solid, starchy", "slightly waxy",
                "dry and starchy"
            ],
            "flavor_profile": FLAVOR_STARCHY,
            "weight_feel": ["heavy", "solid", "surprisingly dense", "dense"],
        },
    },

    "onion": {
        "templates": [
            "A {size} {shape} bulb with {skin_color} papery skin and {layer_texture} inner layers.",
            "A {shape} bulb vegetable wrapped in {skin_texture} {skin_color} skin and filled with {flavor_profile} flavor.",
            "A {size} layered vegetable with {skin_color} outer shells and a {aroma_profile}.",
            "A {shape} bulb whose {skin_texture} skin peels away to reveal {layer_texture} inner rings and {flavor_profile} flavor.",
            "A {size} globe shaped bulb with {skin_color} husk and {layer_texture} layers that release a {aroma_profile}."
        ],
        "values": {
            "size": ["small", "medium sized", "large"],
            "shape": ["round", "layered", "globe shaped"],
            "skin_color": ["pale yellow", "golden", "white", "light brown", "purple"],
            "skin_texture": ["thin, flaky", "dry, papery", "delicate, brittle"],
            "layer_texture": [
                "crisp, translucent", "firm, juicy", "slightly crunchy",
                "tight, concentric"
            ],
            "aroma_profile": AROMA_STRONG,
            "flavor_profile": FLAVOR_SAVORY
        },
    },

    "apple": {
        "templates": [
            "A {size} {shape} fruit with {skin_color} skin and a {flesh_texture} bite.",
            "A {shape} snack with {skin_color} surface and a {flavor_profile} taste.",
            "A {size} piece of tree fruit with {skin_color} skin and {flesh_texture} flesh inside.",
            "A {shape} fruit that feels {weight_feel} in the hand, with {skin_color} peel and a {flavor_profile} flavor.",
            "A {size} {shape} fruit with {skin_color} skin, {flesh_texture} flesh, and a {aroma_profile}."
        ],
        "values": {
            "size": ["small", "medium sized", "large", "palm sized"],
            "shape": ["round", "spheric"],
            "skin_color": [
                "bright red", "greenish yellow", "rosy red",
                "pale red", "red and yellow", "green"
            ],
            "flesh_texture": [
                "crisp and juicy", "firm and crunchy",
                "slightly soft and juicy", "crunchy", "crisp", "juicy", "firm"
            ],
            "flavor_profile": FLAVOR_SWEET + [
                "sweet with a hint of tartness",
                "balanced sweet tart taste"
            ],
            "weight_feel": ["solid", "fairly light", "moderately heavy"],
            "aroma_profile": [
                "light fruity fragrance", "sweet, fresh scent",
                "subtle, clean aroma"
            ],
        },
    },

    "banana": {
        "templates": [
            "A {size} curved fruit with {peel_color} peel and {flesh_texture} flesh.",
            "A {shape} tropical snack with {peel_color} skin and a {flavor_profile} taste.",
            "A {size} tropical fruit that peels easily to reveal {flesh_texture} flesh with a {flavor_profile} flavor.",
            "A {shape} piece of fruit with {peel_color} peel, {flesh_texture} interior, and a {aroma_profile}.",
            "A {size} {shape} fruit with {peel_color} skin that comes off in strips, exposing {flesh_texture} insides."
        ],
        "values": {
            "size": ["small", "medium sized", "long", "short"],
            "shape": ["curved", "slightly curved", "gently arched"],
            "peel_color": [
                "bright yellow", "pale yellow", "brownish yellow",
                "yellow with small brown spots", "greenish yellow"
            ],
            "flesh_texture": [
                "soft and creamy", "tender and smooth",
                "slightly firm yet creamy", "soft", "mushy", "creamy"
            ],
            "flavor_profile": FLAVOR_SWEET + [
                "sweet and mellow", "rich, sweet taste"
            ],
            "aroma_profile": [
                "mild, sweet aroma", "soft fruity smell",
                "gentle tropical scent"
            ],
        },
    },
}


def generate_sentence(category_dict: dict) -> str:
    """Generate a single description from a category template."""
    tpl_str = random.choice(category_dict["templates"])
    values = category_dict["values"]

    fill_dict = {
        key: random.choice(val_list)
        for key, val_list in values.items()
    }

    text = tpl_str.format(**fill_dict)
    return " ".join(text.split()).strip()

def generate_descriptions(n_per_class: int):
    random.seed(42)
    rows = []

    for label, cat_dict in TEMPLATE.items():
        seen = set()
        print(f"Generating descriptions for '{label}'...")

        while len(seen) < n_per_class:
            s = generate_sentence(cat_dict)
            seen.add(s)

        for idx, desc in enumerate(sorted(seen)):
            rows.append({
                "id": f"{label}_{idx:03d}",
                "label": label,
                "description": desc,
            })

        print(f"{len(seen)} unique descriptions generated.")

    return rows

def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "label", "description"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# main
if __name__ == "__main__":
    all_rows = generate_descriptions(desc_per_class)
    write_csv(all_rows, output_dir)
    print(f"Done. Wrote {len(all_rows)} rows to {output_dir.resolve()}")
