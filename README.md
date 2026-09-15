# Simple RAG Application with Pinecone and OpenAI

This project is a Retrieval-Augmented Generation (RAG) application that enables users to ask questions about YouTube videos (including podcasts). The application transcribes the video, cleans and chunks the transcript, stores the chunks in a vector store, and retrieves relevant information to answer queries using OpenAI's GPT-3.5-turbo.

## Features

- **Video Transcription with Speaker Diarization**: Transcribes YouTube/podcast audio using **WhisperX**, which improves on base Whisper by separating individual speakers and producing more accurate word-level timestamps — useful for podcasts with crosstalk and background noise.
- **Custom Transcript Cleaning & Chunking**: Custom Python preprocessing that:
  - Strips filler words and low-signal/non-meaningful segments before embedding.
  - Splits the cleaned transcript using a **sliding-window chunking strategy with ~20% overlap**, so an idea or sentence isn't cut off at a chunk boundary.
  - Removes empty sections and repeated text left over from transcription.
- **Vector Store Integration**: Uses Pinecone to store and retrieve chunk embeddings based on similarity to the query.
- **Question Answering**: Generates answers to questions using retrieved chunks with OpenAI's GPT-3.5-turbo.

## Architecture

- **Language Model**: OpenAI's GPT-3.5-turbo powers the generation of responses.
- **Transcription**: WhisperX handles speech-to-text and speaker diarization.
- **Preprocessing**: Custom logic (`transcript_preprocessing.py`) cleans the transcript and produces overlapping chunks via a sliding window.
- **Embeddings**: OpenAI's embedding model vectorizes both transcript chunks and user questions — this is the only vectorization method used; Pinecone itself does not generate embeddings.
- **Vector Store**: Pinecone stores chunk embeddings and performs similarity search to find the nearest chunks to a query.
- **Chain**: Implements the logic to handle the question-answering process, from retrieval through to answer generation.

## Design notes (why it's built this way)

- **Vectorization**: OpenAI's embedding model is used for both transcript chunks and questions, with Pinecone purely as the similarity-search store. Retrieval quality was tuned by varying chunk size, chunk overlap, and the number of nearest neighbors (`k`) returned by Pinecone, rather than by swapping embedding models.
- **Transcript cleaning**: Raw WhisperX output still contains filler words and low-value segments, so a custom cleaning pass filters those out before chunking. Poor chunking (too small, no overlap) can hurt retrieval even with a strong embedding model, which is why overlap and chunk size were tuned experimentally.
- **What's custom vs. off-the-shelf**: WhisperX is used as-is for transcription and diarization (no point re-implementing speech separation). The cleaning, filtering, and sliding-window chunking logic on top of it is custom Python.
- **Known limitations / possible next steps**: Finer-grained speaker attribution in downstream chunks and background-speech removal are not implemented yet, but are natural next improvements.

## Prerequisites

- Python 3.9+
- OpenAI API Key
- Pinecone API Key

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Sai-Lohith-Panthangi/RAG-application-using-Pincone-and-OpenAI.git
   cd RAG-application-using-Pincone-and-OpenAI
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   Create a `.env` file in the root directory and add your API keys:

   ```
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   ```

4. **Run the Application**:

   ```bash
   python main.py
   ```

5. **Usage**:
   - Provide the URL of a YouTube video/podcast to download and transcribe its content.
   - The transcript is cleaned and split into overlapping chunks by `transcript_preprocessing.py`.
   - Chunks are embedded with OpenAI's embedding model and stored in Pinecone.
   - Ask questions via `chain.invoke`; the system retrieves the nearest chunks from Pinecone and generates an answer.

## Example

```python
question = "What is the video about?"
response = chain.invoke({"question": question})
print(response)
```

## Limitations

1. **Accuracy**: The quality of responses depends on transcription accuracy and the relevance of retrieved chunks.
2. **Speaker attribution**: Diarization from WhisperX is not yet propagated into per-chunk metadata.
3. **Background speech**: Removing background/non-primary-speaker audio is not yet implemented.
4. **Scalability**: Handling very large videos or multiple videos may require further optimization.
