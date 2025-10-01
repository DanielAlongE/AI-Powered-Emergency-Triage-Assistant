"""Speaker diarization utilities built on top of pyannote.audio.

This module provides a stateful diarization model that maintains persistent
speaker identities across streaming audio chunks by comparing speaker
embeddings (voiceprints).
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from pyannote.audio import Pipeline, Model, Inference
from pydub import AudioSegment

from logger import get_logger


logger = get_logger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of audio with timing information."""

    start_time: float
    end_time: float
    audio_data: AudioSegment
    speaker_id: Optional[str] = None  # Persistent, global ID once assigned
    confidence: float = 0.0


class DiarizationModel:
    """Stateful speaker diarization using pyannote.audio.

    This class runs diarization to obtain speech segments, computes speaker
    embeddings for each segment, and maintains a global registry of speakers
    to ensure consistent IDs over time.
    """

    def __init__(self, use_auth_token: Optional[str] = None, similarity_threshold: float = 0.4) -> None:
        """Initialize the diarization system.

        Args:
            use_auth_token: Hugging Face token for accessing pyannote models.
            similarity_threshold: Cosine similarity threshold (0..1) for matching
                speakers. Higher values are stricter.
        """
        # Global registry for speaker voiceprints
        self.global_speaker_embeddings: Dict[str, np.ndarray] = {}
        self.global_speaker_roles: Dict[str, str] = {}
        self.global_speaker_sample_count: Dict[str, int] = {}
        self.speaker_counter = 0
        self.similarity_threshold = similarity_threshold
        self.use_auth_token = use_auth_token

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=use_auth_token,
        )
        embedding_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=self.use_auth_token
        )
        self.embedding_inference = Inference(embedding_model, window="whole")
        logger.info("Diarization and embedding models loaded.")

    def diarize_audio(self, audio_data: AudioSegment) -> List[AudioChunk]:
        """
        Perform speaker diarization to find speech segments in an audio chunk.
        The speaker labels from this pipeline are temporary for segmentation purposes.

        Args:
            audio_data: AudioSegment object

        Returns:
            List of AudioChunk objects representing speech segments.
        """
        chunks: List[AudioChunk] = []
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_data.export(temp_file.name, format="wav")
            temp_path = temp_file.name

        try:
            logger.info("Running pyannote diarization...")
            diarization = self.pipeline(temp_path)

            for segment, _, speaker_label in diarization.itertracks(yield_label=True):
                start_ms = int(segment.start * 1000)
                end_ms = int(segment.end * 1000)

                chunk = AudioChunk(
                    start_time=segment.start,
                    end_time=segment.end,
                    audio_data=audio_data[start_ms:end_ms],
                    speaker_id=str(speaker_label) if speaker_label is not None else None,
                )
                chunks.append(chunk)

            logger.info("Found %d speech segments.", len(chunks))

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return chunks

    def create_speaker_embedding(self, audio_chunk: AudioSegment) -> np.ndarray:
        """Create a speaker embedding (voiceprint) for the given audio segment."""
        # Ensure mono audio and convert to a normalized float32 numpy array
        if audio_chunk.channels > 1:
            audio_chunk = audio_chunk.set_channels(1)

        samples = np.array(audio_chunk.get_array_of_samples(), dtype=np.float32)
        if audio_chunk.sample_width == 2:  # 16-bit
            samples /= (2 ** 15)
        elif audio_chunk.sample_width == 4:  # 32-bit
            samples /= (2 ** 31)

        waveform = torch.from_numpy(samples).unsqueeze(0)
        embedding = self.embedding_inference({"waveform": waveform, "sample_rate": audio_chunk.frame_rate})

        # Ensure embedding is a 1D L2-normalized vector
        if embedding.ndim == 2:
            embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            logger.warning("Received zero-norm embedding; returning zeros.")
            return np.zeros_like(embedding)

        return embedding / norm

    def register_speaker(self, global_speaker_id: str, role: str, audio_sample: AudioSegment) -> None:
        """Pre-register a known speaker with their role and a voice sample.

        Args:
            global_speaker_id: A unique, persistent identifier for the speaker.
            role: The desired role for the speaker (e.g., "Nurse", "Patient").
            audio_sample: An AudioSegment sample for creating the voiceprint.
        """
        embedding = self.create_speaker_embedding(audio_sample)
        self.global_speaker_embeddings[global_speaker_id] = embedding
        self.global_speaker_roles[global_speaker_id] = role
        self.global_speaker_sample_count[global_speaker_id] = 1
        logger.info("Pre-registered %s as %s.", global_speaker_id, role)

    def process_chunk(self, audio_chunk: AudioSegment, chunk_start_time: float = 0.0) -> List[Dict[str, Any]]:
        """
        Process an audio chunk for robust, stateful streaming diarization.

        This method segments the chunk, creates a voiceprint for each segment,
        and compares it to a global registry of known speakers. It assigns an
        existing speaker ID if a match is found or registers a new one otherwise.

        Args:
            audio_chunk: AudioSegment chunk to process.
            chunk_start_time: Start time of this chunk relative to the stream's origin.

        Returns:
            A list of dictionaries, each containing speaker and segment information.
        """
        speech_segments = self.diarize_audio(audio_chunk)
        results = []
        min_duration_ms = 500

        for segment in speech_segments:
            if len(segment.audio_data) < min_duration_ms:
                logger.debug("Skipping very short segment: %d ms.", len(segment.audio_data))
                continue

            current_embedding = self.create_speaker_embedding(segment.audio_data)

            best_match_id = None
            best_similarity = 0.0
            for global_id, known_embedding in self.global_speaker_embeddings.items():
                similarity = np.dot(current_embedding, known_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = global_id

            final_speaker_id = None
            if best_match_id and best_similarity > self.similarity_threshold:
                # Identified a known speaker; update their master embedding
                final_speaker_id = best_match_id
                logger.info(
                    "Re-identified speaker %s (similarity: %.2f)",
                    self.global_speaker_roles.get(final_speaker_id),
                    best_similarity,
                )

                old_embedding = self.global_speaker_embeddings[final_speaker_id]
                n_samples = self.global_speaker_sample_count[final_speaker_id]
                new_embedding = (old_embedding * n_samples + current_embedding) / (n_samples + 1)

                new_norm = np.linalg.norm(new_embedding)
                if new_norm > 0.0:
                    self.global_speaker_embeddings[final_speaker_id] = new_embedding / new_norm
                else:
                    logger.warning("Averaged embedding has zero norm; keeping old embedding.")

                self.global_speaker_sample_count[final_speaker_id] += 1
            else:
                # Discovered a new speaker; register them
                if best_match_id:
                    logger.warning(
                        "Best match %s below threshold (sim: %.2f, thres: %.2f). Creating new speaker.",
                        self.global_speaker_roles.get(best_match_id), best_similarity, self.similarity_threshold
                    )

                self.speaker_counter += 1
                new_speaker_id = f"Global_Speaker_{self.speaker_counter}"
                role = f"Speaker_{self.speaker_counter}"

                self.global_speaker_embeddings[new_speaker_id] = current_embedding
                self.global_speaker_roles[new_speaker_id] = role
                self.global_speaker_sample_count[new_speaker_id] = 1
                final_speaker_id = new_speaker_id
                logger.info("Discovered new speaker: %s", role)

            role = self.global_speaker_roles.get(final_speaker_id, "Unknown")
            adjusted_start_time = chunk_start_time + segment.start_time
            adjusted_end_time = chunk_start_time + segment.end_time

            result = {
                "role": role,
                "chunk": {
                    "start_time": adjusted_start_time,
                    "end_time": adjusted_end_time,
                    "speaker_id": final_speaker_id,
                    "confidence": best_similarity,
                    "duration": segment.end_time - segment.start_time,
                    "audio_segment": segment.audio_data
                }
            }
            results.append(result)

        return results