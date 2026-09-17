import json
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def debug_wordnet_ordering(name_json_path: str = "./data/imagenet/imagenet1k/imagenet_class_index.json"):
    print("Step 1: Initializing WordNet...")
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    print(f"Step 2: Loading class index from {name_json_path}...")
    try:
        with open(name_json_path) as f:
            class_index_map = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load JSON file: {e}")
        return

    # Extract synsets and human names
    resolved_synsets = {}
    human_names = {}
    valid_indices = []
    
    for idx_str, (wnid, human_label) in class_index_map.items():
        idx = int(idx_str)
        try:
            offset = int(wnid[1:])
            pos = wnid[0]
            synset = wn.synset_from_pos_and_offset(pos, offset)
            resolved_synsets[idx] = synset
            human_names[idx] = human_label
            valid_indices.append(idx)
        except Exception:
            continue

    valid_indices.sort()
    n_classes = len(valid_indices)
    print(f"Successfully resolved {n_classes} valid WordNet synsets.")

    print("Step 3: Building path similarity distance matrix...")
    distance_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        syn_i = resolved_synsets[valid_indices[i]]
        for j in range(i, n_classes):
            syn_j = resolved_synsets[valid_indices[j]]
            sim = syn_i.path_similarity(syn_j)
            if sim is None:
                sim = 0.001
            dist = 1.0 - sim
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    print("Step 4: Computing linkage tree...")
    condensed_distances = squareform(distance_matrix)
    wordnet_linkage = linkage(condensed_distances, method="ward")
    sorted_leaves = leaves_list(wordnet_linkage)
    
    # Get final ordered class keys
    taxonomic_class_order = [valid_indices[i] for i in sorted_leaves]

    print("\n--- DEBUG OUTPUT: FIRST 50 TAXONOMICALLY SORTED CLASSES ---")
    for rank, idx in enumerate(taxonomic_class_order[:50], 1):
        # Print the ordering rank, raw index, and human-readable name
        print(f"Rank {rank:02d} | Index {idx:03d} | Name: {human_names[idx]}")

if __name__ == "__main__":
    # Adjust this path if your script is located in a different subdirectory
    debug_wordnet_ordering("./data/imagenet/imagenet1k/imagenet_class_index.json")
