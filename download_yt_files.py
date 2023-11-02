import glob
import os
import subprocess
from datetime import datetime

import librosa
import randomname
import soundfile as sf
from tqdm import tqdm


# You have to install yt-dlp
def check_ytdlp_availability():
    try:
        # Try running yt-dlp with the --version option to check if it's available
        subprocess.run(["yt-dlp", "--version"], check=True)
        print("yt-dlp is available on your system.")
    except FileNotFoundError:
        print("yt-dlp is not available on your system.")
        print("Please download it from https://github.com/yt-dlp/yt-dlp/releases.")
        exit(0)


check_ytdlp_availability()

run_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_{randomname.get_name()}"
yt_download_dir = os.path.join('yt_downloaded', run_name)
os.makedirs(yt_download_dir, exist_ok=True)

with open('yt_links.txt', 'r', encoding='utf-8') as f:
    all_links = f.readlines()
    all_links = [link for link in all_links if not link.startswith('#')]

os.chdir(yt_download_dir)
for link in all_links:
    os.system(f"yt-dlp -f ba --audio-format wav --extract-audio {link}")

# Convert to mono and 22050 HZ
for file in tqdm(glob.glob('*.wav'), desc='Resampling all files'):
    audio, sr = librosa.load(file, sr=22050, mono=True)
    sf.write(file, audio, 22050)

# Calculate total length of all files in the folder
total_length = 0
for file in glob.glob('*.wav'):
    audio, _ = librosa.load(file, sr=22050, mono=True)
    total_length += librosa.get_duration(y=audio)

# Convert total length to hours
total_length_hours = total_length / 3600  # 1 hour = 3600 seconds

print(f"Total length of all files: {total_length_hours:.2f} hours")
