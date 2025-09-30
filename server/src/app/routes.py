from fastapi import APIRouter, File, Form, UploadFile, HTTPException
import asyncio
import numpy as np
from app.schemas import ConversationResponse, ConversationRequest, TranscriptionRequest, TranscriptionResponse
from models.llama import ConversationAnalizer, MODEL_GEMMA_3
from models.transcription import VoskTranscriber

router = APIRouter(prefix="/api", tags=["triage"])


# receive chunks of audio byte and return a string of text and a chat like conversation using pyannote
@router.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(...),
):
    try:
        audio_data = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {str(e)}")

    # Use asyncio subprocess to run FFmpeg for conversion
    try:
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-f', 'webm', '-i', 'pipe:0', '-f', 's16le', '-ac', '1', '-ar', '16000', 'pipe:1',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        out, err = await process.communicate(input=audio_data)

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {err.decode()}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

    try:
        audio_np_array = np.frombuffer(out, dtype=np.int16)
        sample_rate = 16000
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio data conversion failed: {str(e)}")

    # Transcribe
    try:
        transcriber = VoskTranscriber()
        transcript = transcriber.transcribe((sample_rate, audio_np_array))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    return TranscriptionResponse(transcript=transcript)


# return the most apropriate response based on red-flag words and conversation context
# @router.post("/v1/suggestions", response_model=SuggestionResponse)
# async def suggestions(session_id: str = Form(...)) -> SuggestionResponse:
#     pass


# @router.post("/v1/session/reset")
# async def reset_session(session_id: str = Form(...)) -> dict:


# @router.post("/v1/feedback")
# async def feedback(feedback: NurseFeedback) -> dict:


# given a transcription text, return an array of chat like conversation between the nurse and patient
# we have the option of either using the converstation_analizer based on llama3.2
@router.post("/v1/conversation", response_model=ConversationResponse)
async def converstaion(request: ConversationRequest) -> ConversationResponse:
    try:
        return ConversationAnalizer(MODEL_GEMMA_3).analyze(request.transcript)
    except Exception as e:
        print(e)
        return {'converstaion': []}
