import pandas as pd
import json
from pathlib import Path
from code.config.helpers import parse_filename_metadata

def _parse_metadata(input_folder:str):
    folder = Path(input_folder)
    records = []
    
    for file_path in folder.glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            data["_source_file"] = file_path.name
            if "nviews" in file_path.name:
                extracted_info = parse_filename_metadata(file_path)
                data.update(extracted_info)
            records.append(data)
            
    if not records:
        print("No JSON files found in the specified folder.")
        return []
    
    return records
def build_master_dataset(input_folder: str, master_csv_path: str):
    """
    Reads all JSON files, flattens them into a Pandas DataFrame, and saves to CSV.
    """
    records = _parse_metadata(input_folder)
    if not  records:
        print(f"Nothing created for {input_folder}")
    # pd.json_normalize automatically flattens 'metrics.accuracy', 'metrics.auc', etc.
    df = pd.json_normalize(records)
    
    # Save directly to CSV
    df.to_csv(master_csv_path, index=False)
    print(f"Created master dataset with {len(df)} records at {master_csv_path}")


def update_master_dataset(input_folder: str, master_csv_path: str):
    """
    Loads existing CSV, finds unprocessed JSON files, appends them, and updates the CSV.
    """
    master_path = Path(master_csv_path)
    
    # If the master file doesn't exist yet, build it from scratch
    if not master_path.exists():
        print("Master dataset not found. Building from scratch...")
        return build_master_dataset(input_folder, master_csv_path)

    # Load existing data so we know what we've already parsed
    existing_df = pd.read_csv(master_path)
    
    # Create a set of already processed files for fast lookup
    processed_files = set(existing_df["_source_file"].dropna().unique())
    
    folder = Path(input_folder)
    new_records = []
    
    for file_path in folder.glob("*.json"):
        if file_path.name not in processed_files:
            with open(file_path, "r") as f:
                data = json.load(f)
                data["_source_file"] = file_path.name
                new_records.append(data)
                
    if new_records:
        # Flatten the new records
        new_df = pd.json_normalize(new_records)
        
        # Concatenate the old and new DataFrames
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Overwrite the CSV with the combined data
        updated_df.to_csv(master_path, index=False)
        print(f"Updated master dataset with {len(new_df)} new records.")
    else:
        print("No new files found. Master dataset is up to date.")

if __name__ == "__main__":
    MASTER_FILE = "./results/master_results.csv"
    result_dir = f"/home/stud/nemmler/style_tta/results/ablation/adain_tta/tta_inference/results/imagenet/test_r"
    build_master_dataset(result_dir, MASTER_FILE)
    sec_results_dir = "/home/stud/nemmler/style_tta/results/ablation/style_tta/tta_inference/results/imagenet/test_r"
    geometric_dir = "/home/stud/nemmler/style_tta/results/baseline/tta_inference/results/imagenet/test_r"
    update_master_dataset(sec_results_dir, MASTER_FILE)
    update_master_dataset(geometric_dir,MASTER_FILE)
