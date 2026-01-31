import os
import sys
import subprocess
from pathlib import Path

from config.utils.path_manager import path_manager

def main():
    # 1. Detect Paths
    data_root = path_manager.get_data_detail("nuscenes_base")
    
    if not data_root.exists():
        print(f"❌ Data root not found: {data_root}")
        return

    # 2. Detect Installed Version
    version_found = None
    nuscenes_version_setting = path_manager.get_user_setting("nuscenes_version")

    if (data_root / "v1.0-trainval").exists() and nuscenes_version_setting == "v1.0-trainval":
        version_found = "v1.0-trainval"
        print("✅ Full version detected (v1.0-trainval)")
    elif (data_root / "v1.0-mini").exists() and nuscenes_version_setting == "v1.0-mini":
        version_found = "v1.0-mini"
        print("⚠️ Mini version detected (v1.0-mini). Less data will be used.")
    else:
        print("❌ No version subfolder detected (v1.0-trainval or v1.0-mini) matching user setting.")
        print(f"   Content of {data_root}: {os.listdir(data_root)}")
        return

    # 3. Build Command
    # Note: We use '--extra-tag nuscenes' so it generates standard names
    cmd = [
        sys.executable, "-m", "mmdet3d.utils.create_data", "nuscenes",
        "--root-path", str(data_root),
        "--out-dir", str(data_root),
        "--extra-tag", "nuscenes",
        "--version", version_found
    ]
    
    print(f"\n🚀 Executing data generation for {version_found}...")
    print(f"   Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Generation completed successfully.")
        
        # 4. Verification
        db_infos = path_manager.get_data_detail("nuscenes_dbinfos_train_pkl")
        if db_infos.exists():
            print(f"🎉 Critical file created: {db_infos.name}")
        else:
            print(f"⚠️ Warning: Script finished but {db_infos.name} is not seen")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error executing create_data: {e}")

if __name__ == "__main__":
    main()