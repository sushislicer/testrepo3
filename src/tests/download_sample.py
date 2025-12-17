"""
Download a sample mesh from GSO (Google Scanned Objects).
Automatically tries multiple common object names until one works.
"""

import sys
import requests
import imageio
import numpy as np
from pathlib import Path

# --- Path Setup ---
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import ActiveHallucinationConfig
from src.simulator import VirtualTabletopSimulator

# Save to a generic folder so downstream scripts don't break
OUTPUT_DIR = PROJECT_ROOT / "assets" / "sample_object"
BASE_URL = "https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/master/models"

# List of likely models to try (GSO naming conventions can vary)
CANDIDATE_MODELS = [
    "Apple", 
    "Banana", 
    "Lemon", 
    "Pear",
    "Strawberry",
    "Cup",
    "Mug", 
    "Bowl",
    "Shoe",
    "Cylindrical_Plastic_Bottle",
    "Rubber_Duck", # Retry just in case
    "Duck"
]

def check_url_exists(url: str) -> bool:
    try:
        r = requests.head(url)
        return r.status_code == 200
    except:
        return False

def download_file(url: str, save_path: Path) -> bool:
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            print(f"  [OK] Saved {save_path.name}")
            return True
    except Exception as e:
        print(f"  [Error] {e}")
    return False

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    selected_model = None
    print(f"Searching for a valid model in {BASE_URL}...")

    # 1. Find a valid model
    for model_name in CANDIDATE_MODELS:
        test_url = f"{BASE_URL}/{model_name}/model.obj"
        print(f"Checking: {model_name}...", end=" ", flush=True)
        if check_url_exists(test_url):
            print("FOUND!")
            selected_model = model_name
            break
        else:
            print("Not found.")

    if not selected_model:
        print("\n[Error] Could not find any of the candidate models in the repository.")
        print("Please manually check 'kevinzakka/mujoco_scanned_objects' on GitHub for a valid name.")
        return

    # 2. Download Files
    print(f"\nDownloading '{selected_model}' to {OUTPUT_DIR}...")
    success_obj = download_file(f"{BASE_URL}/{selected_model}/model.obj", OUTPUT_DIR / "model.obj")
    success_tex = download_file(f"{BASE_URL}/{selected_model}/texture.png", OUTPUT_DIR / "texture.png")

    if not success_obj:
        print("Failed to download model.obj")
        return

    # 3. Render Input Image
    print("\nRendering input view...")
    try:
        cfg = ActiveHallucinationConfig()
        cfg.simulator.mesh_path = str(OUTPUT_DIR / "model.obj")
        cfg.simulator.intrinsics.width = 512
        cfg.simulator.intrinsics.height = 512

        sim = VirtualTabletopSimulator(cfg.simulator)
        rgb, _ = sim.render_view(idx=0)

        image_path = OUTPUT_DIR / "input_rgb.png"
        imageio.imwrite(image_path, rgb.astype(np.uint8))
        print(f"[Success] Setup complete.")
        print(f"Mesh:  {OUTPUT_DIR / 'model.obj'}")
        print(f"Image: {image_path}")

    except Exception as e:
        print(f"Rendering failed (check simulator/pyrender installation): {e}")

if __name__ == "__main__":
    main()