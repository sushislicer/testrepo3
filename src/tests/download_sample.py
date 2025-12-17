"""
Download sample mesh (Rubber Duck) from Gazebo Fuel (Google Scanned Objects).
Fallback to Stanford Bunny if Gazebo Fuel is unreachable.
"""

import sys
import requests
import imageio
import numpy as np
import trimesh
from pathlib import Path

# --- Path Setup ---
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import ActiveHallucinationConfig
from src.simulator import VirtualTabletopSimulator

OUTPUT_DIR = PROJECT_ROOT / "assets" / "sample_duck"

# Official Google Scanned Objects hosted on Gazebo Fuel (Bypasses GitHub LFS)
# Note: We use 'tip' to get the latest version.
BASE_FUEL_URL = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/Rubber%20Duck/tip/files"
FUEL_URLS = {
    "model.obj": f"{BASE_FUEL_URL}/meshes/model.obj",
    "texture.png": f"{BASE_FUEL_URL}/materials/textures/texture.png"
}

# Fallback
BUNNY_URL = "https://raw.githubusercontent.com/mikedh/trimesh/master/models/bunny.ply"


def download_from_fuel(save_dir: Path) -> bool:
    """Downloads the Rubber Duck from Gazebo Fuel."""
    print(f"Attempting download from Gazebo Fuel (Rubber Duck)...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Download Mesh
        print(f"  - Downloading OBJ...")
        r_obj = requests.get(FUEL_URLS["model.obj"])
        if r_obj.status_code != 200:
            print(f"  [Failed] HTTP {r_obj.status_code}")
            return False
        with open(save_dir / "model.obj", "wb") as f:
            f.write(r_obj.content)

        # 2. Download Texture
        print(f"  - Downloading Texture...")
        r_tex = requests.get(FUEL_URLS["texture.png"])
        if r_tex.status_code == 200:
            with open(save_dir / "texture.png", "wb") as f:
                f.write(r_tex.content)
        else:
            print("  [Warning] Texture not found (using mesh without texture).")

        print("  [Success] Downloaded GSO Rubber Duck.")
        return True
    except Exception as e:
        print(f"  [Error] {e}")
        return False


def download_fallback_bunny(save_dir: Path) -> None:
    """Fallback to Stanford Bunny if Fuel fails."""
    print("\n[Fallback] Downloading Stanford Bunny...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    mesh = trimesh.load(requests.get(BUNNY_URL).content, file_type='ply')
    if hasattr(mesh.visual, 'vertex_colors'):
        mesh.visual.vertex_colors = [255, 200, 50, 255] # Yellow-ish (Mocking a duck)
    
    mesh.apply_translation(-mesh.centroid)
    mesh.export(save_dir / "model.obj")
    print("  [Success] Saved Stanford Bunny as model.obj")


def main() -> None:
    # 1. Try Gazebo Fuel (GSO)
    success = download_from_fuel(OUTPUT_DIR)
    
    # 2. Fallback
    if not success:
        download_fallback_bunny(OUTPUT_DIR)

    # 3. Render Input Image
    print("\nRendering input view...")
    try:
        cfg = ActiveHallucinationConfig()
        cfg.simulator.mesh_path = str(OUTPUT_DIR / "model.obj")
        cfg.simulator.intrinsics.width = 512
        cfg.simulator.intrinsics.height = 512
        
        # Initialize Simulator
        sim = VirtualTabletopSimulator(cfg.simulator)
        
        # Render
        rgb, _ = sim.render_view(idx=0)
        
        image_path = OUTPUT_DIR / "input_rgb.png"
        imageio.imwrite(image_path, rgb.astype(np.uint8))
        
        print(f"[Complete]")
        print(f"  -> Mesh:  {OUTPUT_DIR / 'model.obj'}")
        print(f"  -> Image: {image_path}")
        
    except Exception as e:
        print(f"Rendering failed: {e}")
        print("Hint: If on Colab, ensure you have EGL installed.")


if __name__ == "__main__":
    main()