import os
import re
import subprocess
import sys
from argparse import ArgumentParser

def is_target_dir(path):
    """
    Checks if a directory matches the pattern: [Number]K/[Number].[Number]
    Example: 300K/0.200
    """
    parts = Path(path).parts
    if len(parts) < 2:
        return False
    
    # Match 300K, 1200K, etc.
    temp_match = re.match(r"^\d+K$", parts[-2])
    # Match 0.200, 1.5, etc.
    rate_match = re.match(r"^\d+\.\d+$", parts[-1])
    
    return bool(temp_match and rate_match)

def main():
    # 1. Setup Argument Parser
    # We use parse_known_args so the driver can handle its own logic
    # while passing everything else to the target script.
    parser = ArgumentParser(description="Driver for graphene breaking rate analysis.")
    parser.add_argument("--search_root", type=str, default=".", help="Base directory to search for data")
    parser.add_argument("--script", type=str, default="InferRate.py", help="Path to your analysis script")
    
    driver_args, passthrough_args = parser.parse_known_args()

    search_root = os.path.abspath(driver_args.search_root)
    script_path = os.path.abspath(driver_args.script)

    if not os.path.exists(script_path):
        print(f"[DRIVER ERROR] Script not found: {script_path}")
        sys.exit(1)

    # 2. Recursively find target directories
    target_dirs = []
    print(f"[DRIVER] Searching for patterns in: {search_root}")
    
    for root, dirs, files in os.walk(search_root):
        # We check the leaf directory and its parent
        if re.match(r"^\d+\.\d+$", os.path.basename(root)):
            parent = os.path.basename(os.path.dirname(root))
            if re.match(r"^\d+K$", parent):
                target_dirs.append(root)

    if not target_dirs:
        print("[DRIVER] No matching directories found (e.g., 300K/0.200).")
        return

    print(f"[DRIVER] Found {len(target_dirs)} directories. Starting processing...\n")

    # 3. Execute script for each directory
    for study_dir in target_dirs:
        print("-" * 60)
        print(f"[RUNNING] Root: {study_dir}")
        
        # Build the command: python script.py --root study_dir [other args]
        cmd = [sys.executable, script_path, "--root", study_dir] + passthrough_args
        
        try:
            # Running with check=True will raise an error if the script crashes
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Script failed for {study_dir} with exit code {e.returncode}")
        except KeyboardInterrupt:
            print("\n[DRIVER] Interrupted by user. Exiting.")
            sys.exit(1)

    print("\n[DRIVER] All tasks completed.")

if __name__ == "__main__":
    from pathlib import Path
    main()