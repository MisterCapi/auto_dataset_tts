import glob
import os
from datetime import datetime
from typing import List

import librosa
import randomname
import soundfile as sf
import torch.cuda
import whisperx
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type, language="pl")

run_name = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_{randomname.get_name()}"


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

    # Convert segments from seconds to samples
    segment_samples = [(int(start * original_sampling_rate), int(end * original_sampling_rate)) for start, end in
                       segments]

    # Cut and save audio segments
    outputs = []
    output_prefix = os.path.splitext(os.path.basename(input_audio_path))[0]
    for idx, (start_sample, end_sample) in enumerate(segment_samples):
        audio_segment = audio[start_sample:end_sample]
        output_path = os.path.join(output_dir_name, f"{output_prefix}_{idx + 1}.wav")
        sf.write(output_path, audio_segment, target_sampling_rate)
        outputs.append(output_path)

    return outputs


def create_segments_for_files(files_to_segment: List[str]):

    for audio_file in tqdm(files_to_segment, desc="Transcribing and segmenting files"):
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size, chunk_size=30)

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Save all segments as
        segments = [[segment['start'], segment['end']] for segment in result["segments"]]
        target_sampling_rate = 22050

        segment_filenames = cut_and_save_audio(audio_file, segments, target_sampling_rate)

        for segment, segment_filename in zip(result["segments"], segment_filenames):
            segment_filename = segment_filename.split('\\')[-2:]
            segment_filename = os.path.join(*segment_filename)
            segment['filename'] = segment_filename.replace('\\', '/')
            segment['duration'] = segment['end'] - segment['start']

        result["segments"].sort(key=lambda x: x['duration'], reverse=True)

        split_id = int(0.95 * len(result["segments"]))
        train_segments = result["segments"][:split_id]
        validation_segments = result["segments"][split_id:]

        output_dir_name = os.path.join('output_audio_segments', run_name)
        with open(os.path.join(output_dir_name, 'train.txt'), 'a', encoding='utf-8') as f:
            for segment in train_segments:
                f.write(f"{segment['filename']}| {segment['text'].strip()}\n")

        with open(os.path.join(output_dir_name, 'validation.txt'), 'a', encoding='utf-8') as f:
            for segment in validation_segments:
                f.write(f"{segment['filename']}| {segment['text'].strip()}\n")


if __name__ == '__main__':
    print(f"Starting run for: {run_name}")
    files_to_segment = glob.glob("full_audio_files/*.wav")
    create_segments_for_files(files_to_segment)
