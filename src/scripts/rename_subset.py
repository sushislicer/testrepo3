
import os
from pathlib import Path

def rename_subset():
    base_dir = Path("assets/objaverse_subset")
    categories = ["mugs", "pans", "kettles", "hammers"]
    
    for cat in categories:
        cat_dir = base_dir / cat
        if not cat_dir.exists():
            print(f"Skipping {cat}, directory not found.")
            continue
            
        # Get all obj files
        obj_files = sorted(list(cat_dir.glob("*.obj")))
        
        print(f"Renaming {len(obj_files)} files in {cat}...")
        
        for i, obj_file in enumerate(obj_files):
            # Check if already renamed to avoid double renaming if run twice
            # Pattern: category_XX.obj
            if obj_file.name.startswith(f"{cat}_") and obj_file.name[len(cat)+1:len(cat)+3].isdigit():
                print(f"  Skipping {obj_file.name}, already renamed.")
                continue
                
            new_name = f"{cat[:-1]}_{i:02d}.obj" # mug_00.obj, pan_00.obj
            # Handle plural/singular naming preference?
            # User said "number them". "mug_00" is standard.
            # Categories are plural "mugs", objects are singular "mug".
            # I'll use singular for filename if possible, or just category name.
            # "mugs" -> "mug_00.obj"
            # "pans" -> "pan_00.obj"
            # "kettles" -> "kettle_00.obj"
            # "hammers" -> "hammer_00.obj"
            
            prefix = cat[:-1] # remove 's'
            if cat == "pans": prefix = "pan" # pans -> pan
            if cat == "mugs": prefix = "mug"
            if cat == "kettles": prefix = "kettle"
            if cat == "hammers": prefix = "hammer"
            
            new_name = f"{prefix}_{i:02d}.obj"
            new_path = cat_dir / new_name
            
            print(f"  Renaming {obj_file.name} -> {new_name}")
            obj_file.rename(new_path)

if __name__ == "__main__":
    rename_subset()
