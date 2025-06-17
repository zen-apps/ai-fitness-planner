import requests
import zipfile
import os
from pathlib import Path
import time
from tqdm import tqdm


def download_usda_branded_foods(save_directory="./fast_api/app/api/nutrition_data"):
    """
    Download and extract the latest USDA Branded Foods dataset
    """

    # Create directory if it doesn't exist
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    # URL for the latest branded foods dataset
    url = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_branded_food_json_2025-04-24.zip"
    zip_filename = save_path / "branded_foods_2025_04.zip"

    print(f"Downloading USDA Branded Foods dataset...")
    print(f"URL: {url}")
    print(f"Saving to: {zip_filename}")

    try:
        # Download the file with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get file size for progress bar
        total_size = int(response.headers.get("content-length", 0))

        with open(zip_filename, "wb") as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

        print(f"\n✅ Download completed: {zip_filename}")
        print(f"File size: {zip_filename.stat().st_size / (1024*1024):.1f} MB")

        # Extract the ZIP file
        print("\n📦 Extracting ZIP file...")
        extract_path = save_path / "extracted"
        extract_path.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_filename, "r") as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            print(f"Files in archive: {file_list}")

            # Extract all files
            for file in tqdm(file_list, desc="Extracting"):
                zip_ref.extract(file, extract_path)

        # Find the JSON file
        json_files = list(extract_path.glob("*.json"))
        if json_files:
            json_file = json_files[0]
            print(f"\n✅ Extraction completed!")
            print(f"JSON file location: {json_file}")
            print(
                f"JSON file size: {json_file.stat().st_size / (1024*1024*1024):.2f} GB"
            )

            return str(json_file)
        else:
            print("❌ No JSON file found in extracted archive")
            return None

    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return None
    except zipfile.BadZipFile as e:
        print(f"❌ ZIP extraction failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def verify_json_structure(json_file_path):
    """
    Quick verification of the JSON file structure
    """
    import json

    print(f"\n🔍 Verifying JSON structure...")

    try:
        with open(json_file_path, "r") as f:
            # Read just the first few characters to check structure
            first_chunk = f.read(1000)
            print(f"First 500 characters:")
            print(first_chunk[:500])

        # Try to parse the beginning to verify it's valid JSON
        with open(json_file_path, "r") as f:
            first_line = f.readline()
            if first_line.strip().startswith("{"):
                print("✅ File appears to be valid JSON format")
            else:
                print("⚠️  File may not be standard JSON format")

    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
