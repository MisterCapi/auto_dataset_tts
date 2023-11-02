# Audio Segmentation and Transcription with WhisperX

This repository contains Python scripts to perform audio segmentation and transcription using the WhisperX ASR (Automatic Speech Recognition) model. WhisperX is a pre-trained ASR model that can be used to transcribe audio files into text.

## Purpose

The purpose of this repository is to facilitate the process of segmenting long audio files into smaller chunks and transcribing those chunks into text. This can be useful for various applications such as creating training data for ASR models, generating subtitles for videos, or extracting specific spoken content from audio recordings.

## Usage

### 1. Prerequisites

Before using the scripts in this repository, ensure that you have the following installed:

- Python 3.x
- `yt-dlp` for downloading YouTube audio files. You can download `yt-dlp` from [https://github.com/yt-dlp/yt-dlp/releases](https://github.com/yt-dlp/yt-dlp/releases).

### 2. Set Up

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/audio-segmentation-transcription.git
   cd audio-segmentation-transcription
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### 3. Audio Segmentation and Transcription

#### Segmentation and Transcription from Local Audio Files

1. Place your audio files (in WAV format) inside the `full_audio_files` directory.

2. Create a list of YouTube links in `yt_links.txt` if you want to download audio from YouTube.

3. Run the `download_yt_files.py` script to download and preprocess audio files from YouTube:

   ```bash
   python download_yt_files.py
   ```

4. Run the `main.py` script to segment and transcribe the audio files:

   ```bash
   python main.py
   ```

   The segmented audio files and corresponding transcriptions will be saved in the `output_audio_segments` directory.

### 4. Output

- **Segmented Audio Files**: The segmented audio files (in WAV format) will be saved in the `output_audio_segments/{run_name}/audio` directory.

- **Transcriptions**: The transcriptions for each segment will be saved in `output_audio_segments/{run_name}/train.txt` (for training data) and `output_audio_segments/{run_name}/validation.txt` (for validation data).

## Notes

- Ensure that you have enough storage space, especially if dealing with large audio files, as the segmented audio files can consume significant disk space.

- You can adjust the parameters in the scripts (such as `batch_size`, `compute_type`, and `language`) to customize the behavior of the WhisperX model according to your requirements.

- For more information about the WhisperX ASR model, refer to the official documentation or repository of the model.
