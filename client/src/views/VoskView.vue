<template>
  <div>
    <v-card class="elevation-2 mt-4">
      <v-card-title>Suggestion</v-card-title>
      <v-card-text>
        <h1>Start by introducing yourself as the nurse!</h1>
      </v-card-text>
    </v-card>
    <v-row>
      <v-col cols="4">
        <v-card class="elevation-2 mt-4 card-min-height">
          <v-card-title>Speech Recognition</v-card-title>
          <v-card-text>
            <v-btn
              @click="isRecording ? stopRecording() : startRecording()"
              :color="isRecording ? 'red' : 'primary'"
              class="mb-4"
            >
              {{ isRecording ? 'Stop Recording' : 'Start Recording' }}
            </v-btn>
            <HighlightTextarea  v-model="transcript" label="Transcript" min-height="280px" :words-to-highlight="['ache', 'breathe', 'bleed', 'pain']"></HighlightTextarea>
          </v-card-text>
        </v-card>
      </v-col>
      <v-col cols="4">
        <ChatConversation :transcript="transcript" />
      </v-col>
      <v-col cols="4">
        
      </v-col>
    </v-row>
    <v-card class="elevation-2 mt-4">
      <v-card-title>Audit Log</v-card-title>
      <v-data-table :items="tableItems"></v-data-table>
    </v-card>

  </div>
</template>

<script setup>
import { ref, inject, markRaw, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import ChatConversation from '@/components/ChatConversation.vue'
import HighlightTextarea from '@/components/HighlightTextarea.vue'

const apiUrl = inject('$apiUrl')

const route = useRoute()
const transcript = ref('')
const isRecording = ref(false)
const mediaRecorder = ref(null)
const audioContext = ref(null)
const analyser = ref(null)
const silenceStart = ref(null)
const silenceThreshold = 0.01 // Adjust this threshold as needed
const isTranscribing = ref(false)

let audioChunks = markRaw([])

const tableItems = [
  {suggestion: "Sample suggestion 1", response: "Sample response 2", similarity: '50%'},
  {suggestion: "Sample suggestion 3", response: "Sample response 4", similarity: '50%'},
  {suggestion: "Sample suggestion 5", response: "Sample response 6", similarity: '50%'},
]


const sessionId = route.params.sessionId

// Cleanup audio context when component is unmounted
onUnmounted(() => {
  if (audioContext.value) {
    audioContext.value.close()
  }
})



const checkSilence = () => {
  if (!analyser.value || !isRecording.value) return

  const bufferLength = analyser.value.fftSize
  const dataArray = new Float32Array(bufferLength)
  analyser.value.getFloatTimeDomainData(dataArray)

  // Calculate RMS (Root Mean Square) from time domain data
  let sum = 0
  for (let i = 0; i < bufferLength; i++) {
    sum += dataArray[i] * dataArray[i]
  }
  const rms = Math.sqrt(sum / bufferLength)

  if (rms < silenceThreshold) {
    if (!silenceStart.value) {
      silenceStart.value = Date.now()
      console.log('Silence started at:', silenceStart.value)
    } else if (Date.now() - silenceStart.value >= 1500) {
      // Silence detected for 2 seconds
      onSilenceDetected()
      // if(!isRecording.value) 
      return
    }
  } else {
    if (silenceStart.value) {
      console.log('Silence reset, audio detected')
    }
    silenceStart.value = null
  }

  // Continue checking
  requestAnimationFrame(checkSilence)
}

const fetchTranscription = async () => {
  if (audioChunks.length === 0 || isTranscribing.value) return

  isTranscribing.value = true
  const chunksToSend = [...audioChunks] // Copy all current chunks

  const audioBlob = new Blob(chunksToSend, { type: 'audio/webm' })
  const formData = new FormData()
  formData.append('audio', audioBlob)

  try {
    const response = await fetch(`${apiUrl}/api/v1/transcribe?session_id=${sessionId}`, {
      method: 'POST',
      body: formData
    })
    const data = await response.json()
    const newTranscript = data?.transcript || ''
    if(newTranscript.length > 2){
      transcript.value += newTranscript
    }
    // Clear chunks after successful transcription
    audioChunks = []
  } catch (error) {
    console.error('Error transcribing:', error)
  } finally {
    isTranscribing.value = false
  }
}

const onSilenceDetected = () => {
  console.log('Silence detected for 2 seconds, fetching transcription')
  restartRecording()
}

const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

    // Set up Web Audio API for silence detection
    audioContext.value = new (window.AudioContext || window.webkitAudioContext)()
    analyser.value = audioContext.value.createAnalyser()
    analyser.value.fftSize = 256
    const source = audioContext.value.createMediaStreamSource(stream)
    source.connect(analyser.value)

    mediaRecorder.value = new MediaRecorder(stream)
    audioChunks = []
    silenceStart.value = null

    mediaRecorder.value.ondataavailable = (event) => {
      // console.log('silenceStart', silenceStart)
      audioChunks.push(event.data)
    }

    mediaRecorder.value.onstop = async () => {
      fetchTranscription()
    }

    mediaRecorder.value.start(5000)
    isRecording.value = true

    // Start silence detection
    checkSilence()

    // intervalTimerId.value = setInterval(fetchTranscription, 10000)

  } catch (error) {
    console.error('Error accessing microphone:', error)
  }
}

const stopRecording = () => {
  if (mediaRecorder.value && isRecording.value) {
    mediaRecorder.value.stop()
    isRecording.value = false
  }
  // Reset silence detection
  silenceStart.value = null
}

// This hack became necessary as sending chunks of audio data caused errors
const restartRecording = () => {
  mediaRecorder.value.stop()
  silenceStart.value = null
  startRecording()
}
</script>

<style scoped>
.card-min-height {
  min-height: 450px;
}
</style>
