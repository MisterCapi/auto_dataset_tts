import glob
import librosa
import soundfile as sf
from tqdm import tqdm

input_folder = "full_audio_files/all/*.wav"

for file in tqdm(glob.glob(input_folder)):
    # Load the audio file without specifying sample rate or mono option
    audio, sr = librosa.load(file)

    # Check if the loaded audio is in mono and 22050 Hz format
    if sr == 22050 and audio.ndim == 1:
        print(f"Skipping {file} - already in mono, 22050 Hz format.")
    else:
        # If not, resample and convert to mono if necessary
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        if sr != 22050:
            audio = librosa.resample(audio, sr, 22050)

        # Save the converted audio back to the same file
        sf.write(file, audio, 22050)

        print(f"Converted {file} to mono, 22050 Hz format.")