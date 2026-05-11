# Project 15: Real-Time Voice Assistant

**Difficulty:** Expert  
**Module:** 13 (Multimodal)

## 📌 The Challenge
Create an end-to-end Voice-to-Voice pipeline (similar to advanced Siri) that accepts audio chunks, transcribes them, pipes the text through an LLM, and synthesizes the output back to streaming audio using WebSockets.

## 📖 The Approach
1. Use the Browser `MediaRecorder` API or python `pyaudio` to stream bytes to the backend.
2. Fast STT (Speech-To-Text) using a small model like Whisper-tiny local or Deepgram.
3. LLM streaming node to immediately begin parsing the response.
4. Fast TTS (Text-To-Speech) using models like ElevenLabs or XTTS to emit audio back rapidly.
