"""
transcript_preprocessing.py

Custom preprocessing pipeline used before embedding + storing transcript
chunks in Pinecone via LangChain.

Pipeline:
    1. Transcribe + diarize audio with WhisperX (replaces base Whisper).
    2. Clean the resulting transcript (drop filler words / low-signal
       segments, collapse empty/repeated text).
    3. Chunk the cleaned transcript with a sliding window and ~20% overlap
       so ideas aren't cut off at chunk boundaries (replaces
       RecursiveCharacterTextSplitter).
    4. Convert chunks into LangChain `Document` objects so they drop
       straight into `PineconeVectorStore.from_documents(...)` the same
       way `text_splitter.split_documents(...)` output did before.

WhisperX does the heavy lifting for speech-to-text and speaker separation.
Everything below (cleaning + chunking) is custom logic written for this
project, since feeding raw transcription straight into an embedding model
tends to produce noisy vectors and worse retrieval rankings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.documents import Document

# --------------------------------------------------------------------------
# 1. Transcription + diarization (WhisperX)
# --------------------------------------------------------------------------

# A minimal set of filler / disfluency tokens to strip during cleaning.
# Extend as needed based on the domain of the podcast(s) you're indexing.
FILLER_WORDS = {
    "um", "uh", "uhh", "umm", "erm", "er",
    "like", "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "literally", "right?", "okay?",
}


@dataclass
class TranscriptSegment:
    """One diarized segment produced by WhisperX."""
    speaker: str
    text: str
    start: float
    end: float


def transcribe_with_whisperx(
    audio_path: str,
    model_size: str = "large-v2",
    device: str = "cpu",
    batch_size: int = 16,
) -> list[TranscriptSegment]:
    """
    Transcribe + diarize an audio file with WhisperX.

    WhisperX is used here (instead of base Whisper, which the original
    notebook used) because it adds forced alignment and speaker
    diarization, which produces much cleaner segment boundaries for
    podcasts with crosstalk / multiple speakers.

    Requires: pip install whisperx
    Diarization requires a HuggingFace token with access to the
    pyannote speaker-diarization models (set HF_TOKEN in your .env).
    """
    import os

    import whisperx  # imported lazily so the rest of the pipeline is
                       # testable without the (heavy) whisperx dependency

    model = whisperx.load_model(model_size, device=device)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # Forced alignment for accurate word-level timestamps.
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device=device
    )

    # Speaker diarization.
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=os.environ.get("HF_TOKEN"), device=device
    )
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = [
        TranscriptSegment(
            speaker=seg.get("speaker", "UNKNOWN"),
            text=seg["text"].strip(),
            start=seg["start"],
            end=seg["end"],
        )
        for seg in result["segments"]
        if seg.get("text", "").strip()
    ]
    return segments


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------

_FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b",
    flags=re.IGNORECASE,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_segment_text(text: str) -> str:
    """Strip filler words and normalize whitespace in a single segment."""
    text = _FILLER_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_transcript(
    segments: Iterable[TranscriptSegment],
    min_words: int = 3,
) -> list[TranscriptSegment]:
    """
    Remove empty/near-empty segments and repeated back-to-back lines,
    and strip filler words from what remains.

    min_words: segments with fewer than this many words after cleaning
               are dropped as low-signal (e.g. "yeah", "okay okay").
    """
    cleaned: list[TranscriptSegment] = []
    last_text = None

    for seg in segments:
        text = clean_segment_text(seg.text)

        if not text or len(text.split()) < min_words:
            continue
        if last_text is not None and text.lower() == last_text.lower():
            # Drop immediate repeats (common ASR artifact).
            continue

        cleaned.append(
            TranscriptSegment(speaker=seg.speaker, text=text, start=seg.start, end=seg.end)
        )
        last_text = text

    return cleaned


# --------------------------------------------------------------------------
# 3. Sliding-window chunking with overlap
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    speakers: list[str] = field(default_factory=list)
    start: float | None = None
    end: float | None = None


def chunk_transcript(
    segments: list[TranscriptSegment],
    chunk_size_words: int = 200,
    overlap_ratio: float = 0.20,
) -> list[Chunk]:
    """
    Sliding-window chunking over a cleaned transcript.

    Joins cleaned segments into a single word stream (keeping speaker/time
    metadata per word) and slides a window of `chunk_size_words`, advancing
    by chunk_size_words * (1 - overlap_ratio) each step. A ~20% overlap
    keeps a sentence or idea from being split across chunk boundaries,
    which was found empirically to improve retrieval quality alongside
    tuning chunk size and Pinecone's top-k. This replaces
    RecursiveCharacterTextSplitter's character-based splitting.
    """
    if not segments:
        return []

    words: list[tuple[str, str, float, float]] = []  # (word, speaker, start, end)
    for seg in segments:
        for w in seg.text.split():
            words.append((w, seg.speaker, seg.start, seg.end))

    step = max(1, int(chunk_size_words * (1 - overlap_ratio)))
    chunks: list[Chunk] = []

    for start_idx in range(0, len(words), step):
        window = words[start_idx : start_idx + chunk_size_words]
        if not window:
            break

        text = " ".join(w for w, _, _, _ in window)
        speakers = sorted({spk for _, spk, _, _ in window})
        chunks.append(
            Chunk(
                text=text,
                speakers=speakers,
                start=window[0][2],
                end=window[-1][3],
            )
        )

        if start_idx + chunk_size_words >= len(words):
            break

    return chunks


# --------------------------------------------------------------------------
# 4. Convert to LangChain Documents (drop-in for split_documents() output)
# --------------------------------------------------------------------------

def chunks_to_documents(chunks: list[Chunk], source: str = "transcription.txt") -> list[Document]:
    """
    Convert Chunk objects into LangChain Documents so they can be passed
    directly to PineconeVectorStore.from_documents(documents, embeddings, ...)
    exactly like the original text_splitter.split_documents(...) output.
    """
    return [
        Document(
            page_content=chunk.text,
            metadata={
                "source": source,
                "speakers": ", ".join(chunk.speakers),
                "start": chunk.start,
                "end": chunk.end,
            },
        )
        for chunk in chunks
    ]


# --------------------------------------------------------------------------
# End-to-end convenience wrapper
# --------------------------------------------------------------------------

def prepare_documents_for_embedding(
    audio_path: str,
    chunk_size_words: int = 200,
    overlap_ratio: float = 0.20,
    min_words: int = 3,
    source: str = "transcription.txt",
) -> list[Document]:
    """Run the full pipeline: WhisperX -> clean -> sliding-window chunk -> Documents."""
    raw_segments = transcribe_with_whisperx(audio_path)
    cleaned_segments = clean_transcript(raw_segments, min_words=min_words)
    chunks = chunk_transcript(
        cleaned_segments, chunk_size_words=chunk_size_words, overlap_ratio=overlap_ratio
    )
    return chunks_to_documents(chunks, source=source)
