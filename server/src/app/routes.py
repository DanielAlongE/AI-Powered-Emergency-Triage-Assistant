from fastapi import APIRouter, File, Form, UploadFile
import io
import numpy as np
import ffmpeg
from app.schemas import ConversationResponse, ConversationRequest, TranscriptionRequest, TranscriptionResponse
from models.llama import ConversationAnalizer, MODEL_GEMMA_3
from models.transcription import VoskTranscriber

router = APIRouter(prefix="/api", tags=["triage"])


# receive chunks of audio byte and return a string of text and a chat like conversation using pyannote
@router.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(...),
):
    audio_data = await audio.read()

    # Use FFmpeg to convert to 16-bit PCM mono at 16kHz
    process = ffmpeg.input('pipe:0').output('pipe:1', format='s16le', acodec='pcm_s16le', ac=1, ar=16000).run_async(
        pipe_stdin=True, pipe_stdout=True, pipe_stderr=True
    )

    process.stdin.write(audio_data)
    process.stdin.close()
    out = process.stdout.read()

    audio_np_array = np.frombuffer(out, dtype=np.int16)
    sample_rate = 16000

    result = VoskTranscriber().transcribe((sample_rate, audio_np_array))
    
    print(result)
    print(sample_rate, str(audio_np_array.dtype), audio_np_array.shape)
    return TranscriptionResponse(transcript=result)


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
