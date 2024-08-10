Building a simple Retrieval Augmented Generation Application using Pincone and OpenAI's API. This application will allow us to ask questions about any Youtube video.

# Simple RAG Application with Pinecone and OpenAI
This project is a Retrieval-Augmented Generation (RAG) application that enables users to ask questions about YouTube videos. The application transcribes the video, splits the text into chunks, stores them in a vector store, and retrieves relevant information to answer queries using OpenAI's GPT-3.5-turbo.

## Features
- **Video Transcription**: Transcribes YouTube videos using OpenAI's Whisper.
- **Text Chunking**: Splits large transcripts into smaller chunks for efficient processing.
- **Vector Store Integration**: Uses Pinecone for storing and retrieving text chunks based on similarity to the query.
- **Question Answering**: Generates answers to questions using retrieved text chunks with OpenAI's GPT-3.5-turbo.

## Architecture
- **Language Model**: OpenAI's GPT-3.5-turbo powers the generation of responses.
- **Document Loader**: Transcribes video and splits the text.
- **Vector Store**: Pinecone is used to store and query chunks of text.
- **Chain**: Implements the logic to handle the question-answering process.

## Prerequisites
- Python 3.7+
- OpenAI API Key
- Pinecone API Key

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/simple-rag-app.git
   cd simple-rag-app 

2. **Install Dependencies**:
   Install the required Python packages using pip
   ```bash
   pip install -r requirements.txt ```

3. **Environment Variables**:
   -Create a .env file in the root directory and add your API keys:
   -OPENAI_API_KEY=your-openai-api-key
   -PINECONE_API_KEY=your-pinecone-api-key

4. **Run the Application**:
   Execute the main script to transcribe a video and interact with the model:
   ```bash
   python main.py

5. **Usage**:
   Transcribe a YouTube Video:
   -Provide the URL of a YouTube video to download and transcribe its content.

6. **Store Transcription in Pinecone**:
   -The transcription is split into chunks and stored in Pinecone for efficient retrieval.

7. **Ask Questions**:
   -Use the chain.invoke method to ask questions, and the system will retrieve relevant chunks and generate an answer.

## Example:
   1. **question = "What is the video about?"**
   2. **response = chain.invoke({"question": question})**
   3. **print(response)**

## Limitations:
   1. Accuracy: The quality of responses depends on the transcription accuracy and the relevance of retrieved chunks.
   2. Scalability: Handling very large videos or multiple videos may require optimization.

