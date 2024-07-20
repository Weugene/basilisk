from __future__ import annotations

import glob
import os
import re

import cv2
import ffmpeg

# Step 1: Find all files matching the pattern
path = "/Users/evgeniisharaborin/basilisk/work/tube/res27/"
# prefix = "dots_metadata_t="
prefix = "pic_t="
postfix = "_Lambda2"
file_pattern = f"{prefix}*{postfix}.png"
image_files = glob.glob(file_pattern)

# Step 2: Extract time information from filenames
pattern = rf"{prefix}(\d*\.\d+){postfix}.png"
filenames = "filelist.txt"
acceleration = 2
time_values = [float(re.search(pattern, filename).group(1)) for filename in image_files]
# Sort images by time
image_files_sorted = [x for _, x in sorted(zip(time_values, image_files))]
time_values = sorted(time_values)
print(time_values)
print(image_files_sorted)
# Create a text with the sorted list of filenames
with open(filenames, "w") as f:
    for i, filename in enumerate(image_files_sorted):
        f.write(f"file {filename}\n")
        if i < len(time_values) - 1:
            dt = (time_values[i + 1] - time_values[i]) / acceleration
            f.write(f"duration {dt}\n")

# Step 3: Compile images into a video using ffmpeg
output_video_path = f"{prefix}output_long_video.mp4"

(
    ffmpeg
    .input(filenames, format='concat', safe=0)  # Adjust framerate as needed
    .filter('scale', 2000, 400, force_original_aspect_ratio="decrease")  # 2000, 400
    .filter('pad', 2000, 400, -1, -1, color="white")
    .output(output_video_path, framerate=1, vcodec="libx264", pix_fmt='yuv420p', fs="10M")
    .overwrite_output()
    .run(quiet=False)
)


print("Video compiled successfully:", output_video_path)
