"""Download sample mesh and render input image."""

import sys
import requests
import imageio
import numpy as np
from pathlib import Path

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import ActiveHallucinationConfig
from src.simulator import VirtualTabletopSimulator

OUTPUT_DIR = PROJECT_ROOT / "assets" / "sample_duck"
BASE_URL = "https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/master/models/Rubber_Duck"
URLS = {
    "model.obj": f"{BASE_URL}/model.obj",
    "texture.png": f"{BASE_URL}/texture.png"
}


def download_file(url: str, save_path: Path) -> None:
    r = requests.get(url)
    if r.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(r.content)
        print(f"Downloaded {save_path.name}")
    else:
        print(f"Failed to download {url}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        download_file(url, OUTPUT_DIR / name)

    print("Rendering input image...")
    try:
        cfg = ActiveHallucinationConfig()
        cfg.simulator.mesh_path = str(OUTPUT_DIR / "model.obj")
        cfg.simulator.intrinsics.width = 512
        cfg.simulator.intrinsics.height = 512

        sim = VirtualTabletopSimulator(cfg.simulator)
        rgb, _ = sim.render_view(idx=0)

        image_path = OUTPUT_DIR / "input_rgb.png"
        imageio.imwrite(image_path, rgb.astype(np.uint8))
        print(f"Saved input pair to {OUTPUT_DIR}")

    except Exception as e:
        print(f"Rendering failed (check simulator/pyrender): {e}")


if __name__ == "__main__":
    main()