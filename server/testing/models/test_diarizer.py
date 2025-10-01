"""Integration-style test utility for the diarization model.

This script scans sample_interviews for MP3 files, performs streaming
speaker diarization, and optionally transcribes each detected segment.
It is intended for manual verification and debugging rather than unit tests.
"""

import os
from typing import List, Tuple, Optional

import numpy as np
from pydub import AudioSegment

from models.diarization.diarizer import DiarizationModel
from models.transcription.whisper_transcriber import WhisperTranscriber


def load_mp3_file(file_path: str, target_sample_rate: int = 16000) -> AudioSegment:
    """Load an MP3 file and return an AudioSegment with desired sample rate.

    Args:
        file_path: Path to the MP3 file.
        target_sample_rate: Target sample rate for the audio.

    Returns:
        An AudioSegment at the requested sample rate (mono).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio = AudioSegment.from_mp3(file_path)

    # Ensure mono
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Set sample rate
    audio = audio.set_frame_rate(target_sample_rate)

    return audio


def find_mp3_files(directory: Optional[str] = None) -> List[str]:
    """Find all MP3 files in the specified directory.

    Args:
        directory: Directory to search for MP3 files. Defaults to testing/models/sample_interviews.

    Returns:
        List of MP3 file paths.
    """
    if directory is None:
        directory = os.path.join(os.path.dirname(__file__), "sample_interviews")

    if not os.path.isdir(directory):
        return []

    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".mp3")
    ]


def convert_audiosegment_to_numpy(audio_segment: AudioSegment) -> Tuple[int, np.ndarray]:
    """Convert AudioSegment to (sample_rate, float32 numpy array in [-1, 1])."""
    sample_rate = audio_segment.frame_rate
    audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    # Normalize to [-1, 1] according to bit depth
    if audio_segment.sample_width == 2:  # 16-bit
        audio_array = audio_array / float(2 ** 15)
    elif audio_segment.sample_width == 4:  # 32-bit
        audio_array = audio_array / float(2 ** 31)

    return sample_rate, audio_array


def test_with_mp3_files() -> Optional[list]:
    """Test streaming diarization with real MP3 files split into 10-second chunks."""
    print("Testing streaming diarization with MP3 files…")

    mp3_files = find_mp3_files()

    if not mp3_files:
        print("No MP3 files found. Place MP3s under testing/models/sample_interviews/ and retry.")
        return None

    print(f"Found {len(mp3_files)} MP3 file(s):")
    for file in mp3_files:
        print(f"  - {os.path.basename(file)}")

    results_all = []

    # Process each MP3 file with its own diarizer instance
    for mp3_file in mp3_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {os.path.basename(mp3_file)}")
        print(f"{'=' * 60}")

        try:
            audio_data = load_mp3_file(mp3_file)

            print(f"  Audio duration: {len(audio_data) / 1000:.2f} seconds")
            print(f"  Audio channels: {audio_data.channels}, Sample rate: {audio_data.frame_rate} Hz")

            # Create a new diarization model instance for each MP3 file
            model = DiarizationModel()

            # Initialize Whisper transcriber (optional)
            print("Initializing Whisper transcriber…")
            transcriber = WhisperTranscriber(device="cpu")

            # Split audio into 10-second chunks
            chunk_duration_ms = 10_000
            total_duration_ms = len(audio_data)

            print(f"  Splitting into {chunk_duration_ms / 1000:.0f}-second chunks…")

            file_results = []
            chunk_number = 0

            for start_ms in range(0, total_duration_ms, chunk_duration_ms):
                end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
                chunk_start_time = start_ms / 1000.0  # seconds

                audio_chunk = audio_data[start_ms:end_ms]
                chunk_number += 1

                print(f"\n  Processing chunk {chunk_number}: {chunk_start_time:.1f}s - {end_ms / 1000:.1f}s")

                # Process chunk with streaming diarization
                chunk_results = model.process_chunk(audio_chunk, chunk_start_time)

                print(f"     Found {len(chunk_results)} speaker segments in this chunk:")
                for i, result in enumerate(chunk_results):
                    role = result["role"]
                    chunk_data = result["chunk"]

                    # Get the audio segment for transcription
                    audio_segment = chunk_data["audio_segment"]

                    # Convert AudioSegment to format expected by Whisper
                    audio_numpy = convert_audiosegment_to_numpy(audio_segment)

                    # Transcribe the segment
                    try:
                        transcription = transcriber.transcribe(audio_numpy)
                        transcription = transcription.strip() if isinstance(transcription, str) else str(transcription)
                        if not transcription:
                            transcription = "[No speech detected]"
                    except Exception as e:
                        transcription = f"[Transcription error: {e}]"

                    print(
                        f"       Segment {i + 1}: {role} {chunk_data['start_time']:.1f}s–{chunk_data['end_time']:.1f}s "
                        f"(confidence: {chunk_data['confidence']:.2f})"
                    )
                    print(f"                Transcription: {transcription}")

                file_results.extend(chunk_results)

            print(f"\n  Total segments for {os.path.basename(mp3_file)}: {len(file_results)}")
            results_all.extend(file_results)

        except Exception as e:
            print(f"  Error processing {mp3_file}: {e}")
            import traceback
            traceback.print_exc()

    return results_all


if __name__ == "__main__":
    test_with_mp3_files()




