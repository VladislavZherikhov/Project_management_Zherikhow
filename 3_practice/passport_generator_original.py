"""
Synthetic Russian passport main-page generator for classroom practice.
All positions calibrated to a 608 x 759 template.
"""

import argparse
import json
import random
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

TEMPLATE_PATH    = Path("templates/passport_template.jpg")
PORTRAITS_FOLDER = Path("portraits")
SIGNS_FOLDER     = Path("signs")
OUTPUT_FOLDER    = Path("generated_passports")

TEXT_POSITIONS = {
    'police_department': (130, 90),
    'given_date':        (130, 187),
    'dep_number':        (380, 187),
    'surname':           (280, 480),
    'name':              (280, 540),
    'patronymic':        (280, 570),
    'gender':            (250, 610),
    'birth_date':        (350, 610),
    'birth_place':       (280, 635),
}

FIELD_MAX_WIDTH = {
    'police_department': 400,
    'surname':           280,
    'name':              280,
    'patronymic':        280,
    'birth_place':       250,
    'given_date':        110,
    'dep_number':        130,
    'gender':            55,
    'birth_date':        120,
}
DEFAULT_MAX_WIDTH = 300

PHOTO_POSITION = (30, 493)
PHOTO_SIZE     = (160, 222)

SIGN_POSITIONS = [(365, 200), (365, 270)]
SIGN_SIZE      = (150, 90)

FIELD_COLOR  = '#3a3540'
NUMBER_COLOR = '#80383b'

FONT_SIZE_FIELD  = 13
FONT_SIZE_NUMBER = 22

MONTHS = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}

REGION_PACKS = [
    {
        "police_department": "ГУ МВД РОССИИ ПО ИРКУТСКОЙ ОБЛАСТИ",
        "birth_places": ["Г. ИРКУТСК", "Г. АНГАРСК", "Г. БРАТСК"],
    },
    {
        "police_department": "ГУ МВД РОССИИ ПО МОСКОВСКОЙ ОБЛАСТИ",
        "birth_places": ["Г. МОСКВА", "Г. ПОДОЛЬСК", "Г. ХИМКИ"],
    },
    {
        "police_department": "ГУ МВД РОССИИ ПО НОВОСИБИРСКОЙ ОБЛАСТИ",
        "birth_places": ["Г. НОВОСИБИРСК", "Г. БЕРДСК"],
    },
    {
        "police_department": "ГУ МВД РОССИИ ПО ХАБАРОВСКОМУ КРАЮ",
        "birth_places": ["Г. ХАБАРОВСК", "Г. КОМСОМОЛЬСК-НА-АМУРЕ"],
    },
]

FONT_CANDIDATES_MAIN = [
    "C:/Windows/Fonts/timesbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "timesbd.ttf",
]

FONT_CANDIDATES_SECONDARY = [
    "C:/Windows/Fonts/courbd.ttf",
    "C:/Windows/Fonts/courier.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "courbd.ttf",
]


def load_font(candidates, size):
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    print("Warning: no TrueType font found, using Pillow default font.")
    return ImageFont.load_default()


def fit_font(draw, text, max_width, base_size, min_size, candidates):
    size = base_size
    while size > min_size:
        font = load_font(candidates, size)
        if draw.textlength(text, font=font) <= max_width:
            return font
        size -= 1
    return load_font(candidates, min_size)


def add_watermark(img, text="SYNTHETIC TRAINING SAMPLE - NOT A REAL DOCUMENT"):
    draw = ImageDraw.Draw(img)
    font_size = max(14, min(img.size) // 45)
    font = load_font(FONT_CANDIDATES_MAIN, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = max(10, (img.width - w) // 2)
    y = max(10, img.height - h - 15)
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(210, 0, 0))
    return img


def fit_portrait(img, size):
    """Scale + center-crop to exactly fill `size` — no white bars, no distortion."""
    target_w, target_h = size
    target_ratio = target_w / target_h
    w, h = img.size
    ratio = w / h

    if ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def add_rotated_number(image, text, font, position, color=NUMBER_COLOR,
                       digit_gap=6):
    """Draw a passport number rotated -90°, one digit at a time with gaps."""
    x, y = position

    bbox = font.getbbox("0")
    digit_h = bbox[3] - bbox[1] + 2
    digit_w = bbox[2] - bbox[0]

    total_len = len(text) * (digit_h + digit_gap)

    canvas = Image.new("RGBA", (digit_w + 4, total_len + 4),
                       (255, 255, 255, 0))
    cdraw = ImageDraw.Draw(canvas)

    for i, ch in enumerate(text):
        cdraw.text((2, i * (digit_h + digit_gap)), ch, font=font, fill=color)

    rotated = canvas.rotate(-90, expand=True)
    image.paste(rotated, (x, y), rotated)


def generate_person_data(fake, rng):
    pack = rng.choice(REGION_PACKS)
    gender = rng.choice(["МУЖ.", "ЖЕН."])

    if gender == "МУЖ.":
        name       = fake.first_name_male().upper()
        surname    = fake.last_name_male().upper()
        patronymic = fake.middle_name_male().upper()
    else:
        name       = fake.first_name_female().upper()
        surname    = fake.last_name_female().upper()
        patronymic = fake.middle_name_female().upper()

    birth_date = fake.date_of_birth(minimum_age=18, maximum_age=90)
    given_date = fake.date_between(start_date='-30y', end_date='today')

    return {
        'gender':            gender,
        'name':              name,
        'surname':           surname,
        'patronymic':        patronymic,
        'birth_date':        birth_date.strftime("%d.%m.%Y"),
        'given_date':        given_date.strftime("%d.%m.%Y"),
        'police_department': pack["police_department"],
        'birth_place':       rng.choice(pack["birth_places"]),
        'dep_number':        f"{rng.randint(0, 999):03d}-{rng.randint(0, 999):03d}",
        'pass_number': (
            f"{rng.randint(0, 99):02d} "
            f"{rng.randint(0, 99):02d} "
            f"{rng.randint(0, 999999):06d}"
        ),
    }


def generate_dataset(n, seed, template_path, portraits_folder, signs_folder, output_folder):
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    fake = Faker('ru_RU')
    if seed is not None:
        fake.seed_instance(seed)
    rng = random.Random(seed)

    template_path    = Path(template_path)
    portraits_folder = Path(portraits_folder)
    signs_folder     = Path(signs_folder)
    output_folder    = Path(output_folder)

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    if not portraits_folder.exists():
        raise FileNotFoundError(f"Portraits folder not found: {portraits_folder}")
    if not signs_folder.exists():
        raise FileNotFoundError(f"Signs folder not found: {signs_folder}")

    template = Image.open(template_path).convert("RGB")

    portrait_files = sorted(
        f.name for f in portraits_folder.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )
    sign_files = sorted(
        f.name for f in signs_folder.iterdir()
        if f.suffix.lower() == '.png'
    )

    if not portrait_files:
        raise FileNotFoundError(f"No portraits found in {portraits_folder}")
    if len(sign_files) < 2:
        raise FileNotFoundError(f"At least 2 sign PNGs required in {signs_folder}")

    if n is None or n > len(portrait_files):
        n = len(portrait_files)

    output_folder.mkdir(parents=True, exist_ok=True)
    metadata_path = output_folder / "metadata.jsonl"

    # --- rotated number geometry (computed once) ---
    font_main = load_font(FONT_CANDIDATES_MAIN, FONT_SIZE_NUMBER)
    digit_gap = 6
    bbox = font_main.getbbox("0")
    digit_h = bbox[3] - bbox[1] + 2
    digit_w = bbox[2] - bbox[0]

    RIGHT_X = template.width - digit_h - 4

    page_h = template.height
    half_h = page_h // 2

    with metadata_path.open("w", encoding="utf-8") as mf:
        for index in range(1, n + 1):
            photo_name  = portrait_files[index - 1]
            person_data = generate_person_data(fake, rng)

            passport = template.copy()
            draw = ImageDraw.Draw(passport)

            # --- text fields ---
            for field, pos in TEXT_POSITIONS.items():
                text = str(person_data.get(field, ""))
                if not text:
                    continue
                max_w = FIELD_MAX_WIDTH.get(field, DEFAULT_MAX_WIDTH)
                font = fit_font(
                    draw, text, max_w,
                    FONT_SIZE_FIELD, 8,
                    FONT_CANDIDATES_SECONDARY,
                )
                draw.text(tuple(pos), text, font=font, fill=FIELD_COLOR)

            # --- signatures ---
            selected_signs = rng.sample(sign_files, 2)
            for sign_name, pos in zip(selected_signs, SIGN_POSITIONS):
                sign = Image.open(signs_folder / sign_name).convert("RGBA")
                sign = sign.resize(SIGN_SIZE, Image.Resampling.LANCZOS)
                passport.paste(sign, tuple(pos), sign)

            # --- portrait ---
            with Image.open(portraits_folder / photo_name) as portrait:
                photo = fit_portrait(portrait.convert("RGB"), PHOTO_SIZE)
                passport.paste(photo, tuple(PHOTO_POSITION))

            # --- rotated passport number, one digit at a time ---
            total_len = len(person_data['pass_number']) * (digit_h + digit_gap)

            top_y    = (half_h - total_len) // 2
            bottom_y = half_h + (half_h - total_len) // 2

            add_rotated_number(
                passport,
                person_data['pass_number'],
                font_main,
                (RIGHT_X, top_y),
                digit_gap=digit_gap,
            )
            add_rotated_number(
                passport,
                person_data['pass_number'],
                font_main,
                (RIGHT_X, bottom_y),
                digit_gap=digit_gap,
            )

            # --- watermark ---
            passport = add_watermark(passport)

            image_name = f"passport_{index:05d}.jpg"
            image_path = output_folder / image_name
            passport.save(image_path, quality=95)

            record = {
                "image_id":         f"passport_{index:05d}",
                "document_type":    "PASSPORT_MAIN",
                "synthetic":        True,
                "seed":             seed,
                "quality_label":    "good",
                "defect_type":      "none",
                "fields":           person_data,
                "portrait_source":  photo_name,
                "signs_used":       selected_signs,
                "template":         str(template_path),
                "image_path":       str(image_path),
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

            if index % 10 == 0 or index == n:
                print(f"Saved {index}/{n}")

    print(f"\nGenerated {n} passport images in: {output_folder}")
    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic passport main-page images (educational)."
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260001)
    parser.add_argument("--template",  type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--portraits", type=Path, default=PORTRAITS_FOLDER)
    parser.add_argument("--signs",     type=Path, default=SIGNS_FOLDER)
    parser.add_argument("--output",    type=Path, default=OUTPUT_FOLDER)
    args = parser.parse_args()

    generate_dataset(
        n=args.n,
        seed=args.seed,
        template_path=args.template,
        portraits_folder=args.portraits,
        signs_folder=args.signs,
        output_folder=args.output,
    )


if __name__ == "__main__":
    main()