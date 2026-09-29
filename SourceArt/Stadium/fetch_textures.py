"""Download the stadium's CC0 PBR texture sets from Poly Haven (1K JPG: diffuse, OpenGL normal, ARM = AO/rough/metal).

Usage: python SourceArt/Stadium/fetch_textures.py      -> SourceArt/Stadium/textures/<id>_{diff,nor_gl,arm}_1k.jpg
Poly Haven assets are CC0 (https://polyhaven.com/license). build_realism.py assigns them to the stadium materials.
"""

import json
import os
import urllib.request

SETS = ["concrete_floor_worn_001", "hexagonal_concrete_paving", "rubberized_track", "grass_ground", "metal_plate_02",
        "brown_planks_03", "rock_face_03", "slab_tiles", "rubber_tiles", "painted_plaster_wall"]
MAPS = ["Diffuse", "nor_gl", "arm"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "PoOlympic-stadium-builder"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for asset in SETS:
        files = json.loads(get(f"https://api.polyhaven.com/files/{asset}"))
        for m in MAPS:
            url = files[m]["1k"]["jpg"]["url"]
            path = os.path.join(OUT, os.path.basename(url))
            if not os.path.exists(path):
                data = get(url)
                with open(path, "wb") as f:
                    f.write(data)
            total += os.path.getsize(path)
            print(os.path.basename(path))
    print(f"{len(SETS) * len(MAPS)} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
