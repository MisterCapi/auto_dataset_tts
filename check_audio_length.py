import glob
import os

import librosa
from tqdm import tqdm

# Simple script to keep track how much audio you already collected in folders
# Edit the folders list to fit your folders you wish to measure
folders = ['output_audio_segments/2023_10_25_12_34_brave-itinerary_6h/audio',
           'output_audio_segments/2023_10_29_00_10_callous-class/audio',
           'output_audio_segments/2023_10_30_00_00_tempered-reef/audio']

total_length = 0

# Iterate through the list of folders and glob files
for folder in folders:
    # Use os.path.join to create the full path to the folder
    folder_path = os.path.join(folder, '*.wav')

    # Use glob.glob to get a list of files matching the pattern in the folder
    files = glob.glob(folder_path)

    # Iterate through the files and calculate total length
    for file in tqdm(files):
        audio, _ = librosa.load(file, sr=22050, mono=True)
        total_length += librosa.get_duration(y=audio)

# Convert total length to hours
total_length_hours = total_length / 3600  # 1 hour = 3600 seconds

print(f"Total length of all files: {total_length_hours:.2f} hours")
