import os
from PIL import Image

patch_dir = "data/midog22_dataset/patch_images/train@breasts"
blank_count = 0
total = 0
tiny_threshold = 2000  # bytes; healthy 224x224 RGB patches are usually 30-100KB+

for label in ["0", "1"]:
    d = os.path.join(patch_dir, label)
    for fname in os.listdir(d):
        total += 1
        size = os.path.getsize(os.path.join(d, fname))
        if size < tiny_threshold:
            blank_count += 1

print(f"{blank_count}/{total} patches under {tiny_threshold} bytes ({100*blank_count/total:.1f}%)")
