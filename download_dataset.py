"""
download_dataset.py
Downloads the raw_faces and reference_identities from Google Drive.

Usage: python download_dataset.py
"""

import subprocess
import sys
import os

# Google Drive folder ID from the assignment
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1oBBQOLeeVySabKUQt8eGRsu7AkYTofkQ?usp=sharing"
GDRIVE_FOLDER_ID = "1oBBQOLeeVySabKUQt8eGRsu7AkYTofkQ"

def main():
    # Ensure gdown is installed
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    print("Downloading dataset from Google Drive...")
    print(f"Folder: {GDRIVE_FOLDER_URL}")
    print()

    # Download entire folder
    output_dir = os.path.dirname(os.path.abspath(__file__))
    gdown.download_folder(
        url=GDRIVE_FOLDER_URL,
        output=output_dir,
        quiet=False,
        use_cookies=False
    )

    # Check what was downloaded
    raw_faces = os.path.join(output_dir, "raw_faces")
    ref_ids = os.path.join(output_dir, "reference_identities")

    if os.path.isdir(raw_faces):
        count = len([f for f in os.listdir(raw_faces) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"\nraw_faces/: {count} images found")
    else:
        print("\nWARNING: raw_faces/ directory not found.")
        print("You may need to manually download from:")
        print(f"  {GDRIVE_FOLDER_URL}")
        print("And place the images in raw_faces/ and reference_identities/ folders.")

    if os.path.isdir(ref_ids):
        count = len([f for f in os.listdir(ref_ids) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"reference_identities/: {count} images found")
    else:
        print("WARNING: reference_identities/ directory not found.")

    print("\nDone! You can now run: python solution.py")


if __name__ == "__main__":
    main()
