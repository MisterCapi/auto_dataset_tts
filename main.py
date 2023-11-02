import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import librosa
import numpy as np
import randomname
import soundfile as sf
import torch.cuda
import whisperx
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4              # reduce if low on GPU mem
compute_type = "float16"    # change to "int8" if low on GPU mem (may reduce accuracy)
language = "pl"

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=language)
model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

run_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_{randomname.get_name()}"


@dataclass
class Segment:
    text: str
    filepath: str
    duration: float


def cut_sample_to_speech_only(audio_path):
    audio_sample = whisperx.load_audio(audio_path)
    result_sample = model.transcribe(audio_sample, batch_size=batch_size, chunk_size=30)
    result_sample = whisperx.align(result_sample["segments"], model_a, metadata, audio_sample, device)

    try:
        end = result_sample['segments'][-1]['end']
    except IndexError:
        print(f"Skipping {audio_path}")
        return False

    # Load audio with Librosa
    audio, sr = librosa.load(audio_path)

    # Calculate dB amplitude
    amplitude_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)

    # This logic is here to prevent whisper from cutting audio too short (this happened quite often in my experience)
    # If you trust whisper 100% (or have noisy audio) -> commend all "optional" code below
    # ======= OPTIONAL CODE =======
    window_size_sec = 0.05
    silence_threshold_db = -30
    window_size_samples = int(window_size_sec * sr)
    start_frame = int(end * sr)

    audio_windows = [amplitude_db[x:x + window_size_samples] for x in
                     range(start_frame, len(amplitude_db), window_size_samples)]

    audio_windows = [np.max(x) for x in audio_windows]

    for window in audio_windows:
        if window > silence_threshold_db:
            end += window_size_sec
        else:
            break
    # ======= END OPTIONAL CODE =======

    sf.write(audio_path, audio[:int(end * sr)], int(sr))
    return True


def cut_and_save_audio(input_audio_path, segments, target_sampling_rate=22050):
    """
    Load audio from input_audio_path, cut segments based on the provided start and end times,
    and save the segments into output_dir with the specified target_sampling_rate.

    Args:
    input_audio_path (str): Path to the input audio file.
    segments (list): List of lists containing start and end times of audio segments to cut, in seconds.
    target_sampling_rate (int, optional): Sampling rate of the output audio segments. Default is 22050.
    """
    # Create the output directory if it doesn't exist
    output_dir_name = os.path.join('output_audio_segments', run_name, 'audio')
    os.makedirs(output_dir_name, exist_ok=True)

    # Load the input audio
    audio, original_sampling_rate = librosa.load(input_audio_path, sr=target_sampling_rate)

    # Cut and save audio segments
    outputs = []
    output_prefix = os.path.splitext(os.path.basename(input_audio_path))[0]
    for idx, segment in tqdm(enumerate(segments), desc=input_audio_path, total=len(segments), leave=False):
        start_sample = int(segment['start'] * original_sampling_rate)
        end_sample = int(segment['end'] * original_sampling_rate)
        audio_segment = audio[start_sample:end_sample]
        output_path = os.path.join(output_dir_name, f"{output_prefix}_{idx + 1}.wav")
        sf.write(output_path, audio_segment, target_sampling_rate)
        # We use whisper again on each sample to get rid of non-speech sounds
        good_audio = cut_sample_to_speech_only(output_path)
        if good_audio:
            outputs.append(Segment(text=segment['text'],
                                   filepath=output_path,
                                   duration=librosa.get_duration(y=audio, sr=original_sampling_rate)))
        else:
            try:
                os.remove(output_path)
            except FileNotFoundError:
                print(f"File '{output_path}' not found.")
            except PermissionError:
                print(f"You do not have permission to delete '{output_path}'.")
            except Exception as e:
                print(f"An error occurred: {e}")

    return outputs


def create_segments_for_files(files_to_segment: List[str]):

    for audio_file in tqdm(sorted(files_to_segment, key=lambda x: os.path.getsize(x), reverse=True),
                           desc="Transcribing and segmenting files"):
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size, chunk_size=30)

        # Align the whisper output
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=True)

        # segments = [[segment['start'], segment['end']] for segment in result["segments"]]
        target_sampling_rate = 22050

        segment_objects = cut_and_save_audio(audio_file, result["segments"], target_sampling_rate)

        for segment in segment_objects:
            segment.filepath = os.path.join(*segment.filepath.split(os.sep)[-2:])
            segment.filepath = segment.filepath.replace('\\', '/')

        segment_objects = sorted(segment_objects, key=lambda x: x.duration, reverse=True)

        split_id = int(0.95 * len(result["segments"]))
        train_segments = segment_objects[:split_id]
        validation_segments = segment_objects[split_id:]

        output_dir_name = os.path.join('output_audio_segments', run_name)
        with open(os.path.join(output_dir_name, 'train.txt'), 'a', encoding='utf-8') as f:
            for segment in train_segments:
                f.write(f"{segment.filepath}| {segment.text.strip()}\n")

        with open(os.path.join(output_dir_name, 'validation.txt'), 'a', encoding='utf-8') as f:
            for segment in validation_segments:
                f.write(f"{segment.filepath}| {segment.text.strip()}\n")


if __name__ == '__main__':
    print(f"Starting run for: {run_name}")
    files_to_segment = glob.glob("full_audio_files/*.wav")
    create_segments_for_files(files_to_segment)
