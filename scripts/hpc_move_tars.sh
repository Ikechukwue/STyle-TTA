#!/bin/bash

# Configuration
SOURCE_BASE="$HPCVAULT/augmented_cache/test_abl"
DEST_BASE="$HPCVAULT/augmented_cache/test_abl/complete"
mkdir -p "$DEST_BASE"

# Loop through each experiment subfolder
for folder_path in "$SOURCE_BASE"/*/; do
    # Skip the 'complete' folder itself
    folder_name=$(basename "$folder_path")
    if [ "$folder_name" == "complete" ]; then continue; fi

    # --- SKIP MECHANIC ---
    if [ -f "$DEST_BASE/${folder_name}.tar" ]; then
        echo "Skipping $folder_name: ${folder_name}.tar already exists in $DEST_BASE"
        continue
    fi
    # ---------------------

    echo "------------------------------------------------"
    echo "Processing experiment: $folder_name"

    # Create a fresh temp workspace on the SSD
    TMP_EXTRACT="$TMPDIR/extract_$folder_name"
    TMP_MERGED="$TMPDIR/merged_$folder_name"
    rm -rf "$TMP_EXTRACT" "$TMP_MERGED"
    mkdir -p "$TMP_EXTRACT" "$TMP_MERGED"

    # 1. Identify the tar with the highest number
    # Assumes format part_XXXXXX.tar
    LATEST_TAR=$(ls "$folder_path"part_*.tar 2>/dev/null | sort -V | tail -n 1)

    if [ -z "$LATEST_TAR" ]; then
        echo "No tar files found in $folder_name. Skipping."
        continue
    fi

    echo "Latest tar identified: $(basename "$LATEST_TAR")"

    # 2. Extract the latest tar
    echo "Extracting..."
    tar -xf "$LATEST_TAR" -C "$TMP_EXTRACT"

    # 3. Merging Logic (rank_0 and rank_1)
    echo "Merging ranks..."
    for rank_dir in "$TMP_EXTRACT"/*/rank_*; do
        if [ -d "$rank_dir" ]; then
            # cp -r handles the directory structure
            cp -r "$rank_dir"/* "$TMP_MERGED/"
        fi
    done

    # 4. Tar up the merged result
    MERGED_TAR="$TMPDIR/${folder_name}.tar"
    echo "Creating merged tarball: $folder_name.tar"
    tar -cf "$MERGED_TAR" -C "$TMP_MERGED" .

    # 5. Rsync to Vault
    echo "Syncing to Vault..."
    rsync -ahP "$MERGED_TAR" "$DEST_BASE/"

    # Cleanup SSD to save space for the next folder
    rm -rf "$TMP_EXTRACT" "$TMP_MERGED" "$MERGED_TAR"
    echo "Done with $folder_name"
done

echo "================================================"
echo "All experiments merged and moved to $DEST_BASE"
