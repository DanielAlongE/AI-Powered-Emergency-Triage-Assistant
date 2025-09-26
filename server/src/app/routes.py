from fastapi import APIRouter, File, Form, UploadFile
import io
import numpy as np
from pydub import AudioSegment
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

    # Wrap the binary data in an in-memory file-like object
    audio_stream = io.BytesIO(audio_data)

    audio_segment = AudioSegment.from_file(audio_stream, format='webm')        
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
    audio_np_array = np.array(audio_segment.get_array_of_samples())
    sample_rate = audio_segment.frame_rate

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
