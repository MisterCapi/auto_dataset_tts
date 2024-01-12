import glob
import librosa
import soundfile as sf
from tqdm import tqdm

input_folder = "full_audio_files/all/*.wav"

for file in tqdm(glob.glob(input_folder)):
    # Load the audio file without specifying sample rate or mono option
    audio, sr = librosa.load(file)

    audio = librosa.to_mono(audio)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)

    # Save the converted audio back to the same file
    sf.write(file, audio, 22050)
