"""Build the 512-area multi-domain qualitative suite (64 cases = one per DP rank at 64 cards)."""

import json, sys
from pathlib import Path

root = Path(sys.argv[1])
base = json.loads((root / "validation_assets/suite.json").read_text())
S, W, T, P, L = (512, 512), (384, 640), (640, 384), (576, 448), (448, 576)  # (height,width); all <= 262144 px, /16
domains = [
    ("photo_object", S, "a photo of a red apple on a wooden table"),
    (
        "photo_object",
        S,
        "A vintage brass pocket watch lying open on a dark velvet cloth, soft window light, shallow depth of field, macro photograph.",
    ),
    ("photo_animal", S, "a photo of a cat"),
    (
        "photo_animal",
        W,
        "A golden retriever running through shallow water on a beach at sunset, water droplets frozen in the air, warm backlight.",
    ),
    (
        "photo_animal",
        S,
        "A close-up photograph of a blue-and-yellow macaw perched on a branch, sharp feather detail, blurred green jungle background.",
    ),
    (
        "portrait",
        P,
        "A studio portrait of an elderly fisherman with a weathered face and a grey beard, wearing a yellow raincoat, dramatic side lighting.",
    ),
    (
        "portrait",
        P,
        "A young woman with curly red hair smiling at the camera in a sunlit cafe, natural light, 85mm lens, bokeh background.",
    ),
    ("people_scene", W, "Three children flying a kite on a grassy hill under a bright blue sky with scattered clouds."),
    ("people_scene", S, "A chef in a white uniform tossing vegetables in a flaming wok in a busy restaurant kitchen."),
    (
        "landscape",
        W,
        "A snow-capped mountain range reflected in a perfectly still alpine lake at sunrise, pink and orange sky, pine trees in the foreground.",
    ),
    (
        "landscape",
        W,
        "A winding road through rolling green hills in Tuscany with cypress trees, golden hour light, aerial view.",
    ),
    (
        "landscape",
        T,
        "A tall waterfall plunging into a misty gorge surrounded by lush rainforest, long exposure photograph.",
    ),
    (
        "cityscape",
        W,
        "A rainy night street in Tokyo with glowing neon signs reflected on the wet pavement and people holding umbrellas.",
    ),
    (
        "architecture",
        T,
        "A Gothic cathedral interior with tall stained glass windows casting colorful light onto stone columns.",
    ),
    (
        "interior",
        W,
        "A cozy Scandinavian living room with a grey sofa, a wooden coffee table, a floor lamp, and large windows with white curtains.",
    ),
    (
        "food",
        S,
        "A bowl of ramen with sliced pork, a soft-boiled egg, green onions and nori, viewed from above on a dark wooden table.",
    ),
    (
        "food",
        S,
        "A stack of fluffy pancakes topped with fresh blueberries and maple syrup dripping down the sides, morning light.",
    ),
    (
        "product",
        S,
        "A pair of white sneakers on a pastel pink background, clean product photography with soft shadows.",
    ),
    ("vehicle", W, "a photo of a red car"),
    (
        "vehicle",
        W,
        "A steam locomotive crossing a stone viaduct in the Scottish Highlands, white smoke trailing behind.",
    ),
    ("compose_count", S, "a photo of two clocks"),
    ("compose_count", S, "Three yellow rubber ducks in a row on the edge of a white bathtub."),
    ("compose_color", S, "A green bench next to a red fire hydrant on a city sidewalk."),
    (
        "compose_spatial",
        S,
        "A black cat sitting on top of a stack of books, with a blue teacup to the left of the books.",
    ),
    (
        "art_oil",
        S,
        "An oil painting of a sailing ship in a stormy sea, thick impasto brush strokes, in the style of the Dutch Golden Age.",
    ),
    (
        "art_watercolor",
        S,
        "A delicate watercolor painting of cherry blossoms on a branch with a small bird, soft pastel colors on white paper.",
    ),
    (
        "art_anime",
        P,
        "An anime-style illustration of a girl with long silver hair and a school uniform standing on a rooftop at dusk, wind blowing.",
    ),
    ("art_pixel", S, "Pixel art of a small knight standing in front of a castle gate, 16-bit video game style."),
    (
        "art_3d",
        S,
        "A cute 3D rendered robot holding a small potted plant, soft studio lighting, pastel background, Pixar style.",
    ),
    ("art_sketch", S, "A detailed pencil sketch of an old oak tree with twisted roots, black and white."),
    (
        "art_ink",
        T,
        "A traditional Chinese ink wash painting of misty mountains, a small pavilion and a lone boat on a river.",
    ),
    (
        "art_fantasy",
        W,
        "A fantasy concept art of a floating island with waterfalls pouring into the clouds, and a dragon flying in the distance.",
    ),
    ("text_sign", S, 'A wooden sign that says "OPEN" hanging on the glass door of a small bakery.'),
    (
        "text_sign",
        W,
        'A green highway road sign with the words "NEW YORK 25 MILES" in white letters against a blue sky.',
    ),
    (
        "text_object",
        S,
        'A white coffee mug with the text "Good Morning" printed in black cursive letters, on a kitchen counter.',
    ),
    ("text_neon", S, 'A pink neon sign on a brick wall that reads "DREAM BIG".'),
    (
        "text_book",
        P,
        'A book cover with the title "The Silent Forest" in gold serif letters above an illustration of dark pine trees.',
    ),
    (
        "poster",
        T,
        'A minimalist movie poster for a science fiction film titled "ORBIT", showing a lone astronaut in front of a huge ringed planet, bold white title text at the bottom.',
    ),
    (
        "poster",
        T,
        'A colorful music festival poster with the headline "SUMMER FEST 2026", palm trees, a sunset gradient and geometric shapes.',
    ),
    (
        "poster",
        T,
        'A retro travel poster of Paris with the Eiffel Tower, flat vector illustration, the word "PARIS" in large letters.',
    ),
    (
        "infographic",
        S,
        'A bar chart with five blue bars of increasing height on a white background, with the title "Annual Sales" at the top.',
    ),
    (
        "infographic",
        S,
        "A clean infographic showing the water cycle with labeled arrows: evaporation, condensation, precipitation, and collection.",
    ),
    ("infographic", S, "A pie chart divided into four colored segments with a legend on the right side, flat design."),
    ("logo_icon", S, "A flat vector logo of a fox head in orange and white on a dark blue circular background."),
    (
        "ui_screen",
        L,
        'A smartphone screen showing a weather app with a large sun icon, the temperature "24°" and a five-day forecast below.',
    ),
    ("zh_photo", S, "一只橘猫趴在窗台上晒太阳，窗外是秋天的金黄色银杏树。"),
    ("zh_art", T, "水墨画风格的江南水乡，小桥流水，白墙黑瓦，远处有淡淡的烟雨。"),
    ("zh_text", S, '一张红色的春节海报，中间写着金色的大字"新年快乐"，周围有灯笼和烟花。'),
]
assert len(domains) == 48
cases = []
for i, (domain, (h, w), prompt) in enumerate(domains):
    cases.append(
        dict(
            id=f"{i:02d}_{domain}",
            task="T2I",
            prompt=prompt,
            height=h,
            width=w,
            steps=30,
            guidance=4.0,
            shift=3.0,
            seed=5000 + i,
            source=dict(dataset="authored", domain=domain),
        )
    )
# The same eight prompts without guidance: tracks the raw conditional model.
for i in (2, 5, 9, 15, 20, 26, 32, 37):
    domain, (h, w), prompt = domains[i]
    cases.append(
        dict(
            id=f"{i:02d}_{domain}_nocfg",
            task="T2I",
            prompt=prompt,
            height=h,
            width=w,
            steps=30,
            guidance=1.0,
            shift=3.0,
            seed=5000 + i,
            source=dict(dataset="authored", domain=domain),
        )
    )
for case in base["cases"]:
    if case["task"] == "I2T":
        case = dict(case, image=str((root / case["image"]).resolve()))
        cases.append(case)
# Single-turn edits in the released layout: input image turn, instruction turn, image answer.
shas = {c["image"]: c["image_sha256"] for c in base["cases"] if c["task"] == "I2T"}
edits = [
    ("COCO_val2014_000000391895.jpg", "edit_color", "Change the red helmet to a blue helmet."),
    ("COCO_val2014_000000391895.jpg", "edit_weather", "Make it a snowy winter scene."),
    ("COCO_val2014_000000060623.jpg", "edit_style", "Turn this photo into a watercolor painting."),
    ("COCO_val2014_000000060623.jpg", "edit_add", "Add a colorful party hat on the girl’s head."),
    ("COCO_val2014_000000483108.jpg", "edit_remove", "Remove the man on the bicycle."),
    ("COCO_val2014_000000483108.jpg", "edit_time", "Change the scene to night time."),
    ("COCO_val2014_000000384213.jpg", "edit_add", "Add a bowl of fresh fruit on the counter."),
    ("COCO_val2014_000000384213.jpg", "edit_style", "Make it look like a pencil sketch."),
]
for i, (name, domain, prompt) in enumerate(edits):
    cases.append(
        dict(
            id=f"e{i}_{domain}",
            task="EDIT",
            prompt=prompt,
            image=str((root / "validation_assets" / name).resolve()),
            image_sha256=shas["validation_assets/" + name],
            steps=30,
            guidance=4.0,
            shift=3.0,
            seed=6000 + i,
            source=dict(dataset="COCO", domain=domain),
        )
    )
assert len(cases) == 72 and len({c["id"] for c in cases}) == 72
for c in cases:
    if c["task"] == "T2I":
        assert c["height"] % 16 == 0 and c["width"] % 16 == 0 and c["height"] * c["width"] <= 262144
suite = dict(
    schema="cq_qualitative_v1",
    purpose="Stage-1 512-area multi-domain qualitative tracking: 48 T2I prompts over photo, people, landscape, "
    "food, composition, art styles, text rendering, posters, infographics and Chinese prompts at guidance 4; "
    "eight repeated without guidance; eight COCO I2T captions; eight single-turn edits of the COCO images.",
    training_overlap="Prompts are authored for this suite; not audited against the corpus.",
    cases=cases,
)
out = root / "validation_assets/suite_domains_512.json"
out.write_text(json.dumps(suite, indent=1, ensure_ascii=False) + "\n")
print(out, len(cases))
