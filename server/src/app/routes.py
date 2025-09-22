from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter(prefix="/api", tags=["triage"])


# receive chunks of audio byte and return a string of text and a chat like conversation using pyannote
# @router.post("/v1/transcribe", response_model=TranscriptionResponse)
# async def transcribe():
#     pass


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
# @router.post("/conversation", response_model=ChatResponse)
# async def ConversationAnalizer().analyze(transcript: str) -> ChatResponse:
