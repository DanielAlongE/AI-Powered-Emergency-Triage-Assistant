<template>
  <div>
    <v-row>
      <v-col cols="8">
        <v-card class="elevation-5 mt-4">
          <v-card-title>Vosk Speech Recognition</v-card-title>
          <v-card-text>
            <v-btn
              @click="isRecording ? stopRecording() : startRecording()"
              :color="isRecording ? 'red' : 'primary'"
              class="mb-4"
            >
              {{ isRecording ? 'Stop Recording' : 'Start Recording' }}
            </v-btn>
            <HighlightTextarea  v-model="transcript" label="Transcript" min-height="500px" :words-to-highlight="['ache', 'breath', 'bleed']"></HighlightTextarea>
            <v-btn @click="sendTranscript" color="success" class="mt-2">Send Transcript</v-btn>
          </v-card-text>
        </v-card>
      </v-col>
      <v-col cols="4">
        <v-card class="elevation-4">
          <ChatBubble v-for="(msg, index) in messages" :key="index" :content="msg.content" :primary="['NURSE', 'assistant'].includes(msg.role)" />
        </v-card>
      </v-col>
    </v-row>
  </div>
</template>

<script setup>
import { ref, inject, markRaw, onUnmounted } from 'vue'
import ChatBubble from '@/components/ChatBubble.vue'
import HighlightTextarea from '@/components/HighlightTextarea.vue'

const apiUrl = inject('$apiUrl')

const transcript = ref('')
const isRecording = ref(false)
const mediaRecorder = ref(null)
// const audioChunks = ref([])
const messages = ref([])
const audioContext = ref(null)
const analyser = ref(null)
const silenceStart = ref(null)
const silenceThreshold = 0.01 // Adjust this threshold as needed

let audioChunks = markRaw([])


// console.log({audioChunks, v: audioChunks})


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
    } else if (Date.now() - silenceStart.value >= 2000) {
      // Silence detected for 2 seconds
      onSilenceDetected()
      // return
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
  if(audioChunks.length === 0) return

  const chunk = [audioChunks.shift()]

  const audioBlob = new Blob(chunk, { type: 'audio/webm' })
      const formData = new FormData()
      formData.append('audio', audioBlob)

      try {
        const response = await fetch(`${apiUrl}/api/v1/transcribe`, {
          method: 'POST',
          body: formData
        })
        const data = await response.json()
        transcript.value += data.transcript
      } catch (error) {
        audioChunks.unshift(chunk[0])
        console.error('Error transcribing:', error)
      }
}

const onSilenceDetected = () => {
  console.log('Silence detected for 2 seconds, fetching transcription')
  // stopRecording()
  fetchTranscription()
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

    mediaRecorder.value.start(10000)
    isRecording.value = true

    // Start silence detection
    checkSilence()
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

const sendTranscript = async () => {
  if (!transcript.value.trim()) return

  try {
    const response = await fetch(`${apiUrl}/api/v1/conversation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript: transcript.value })
    })
    const { conversation } = await response.json()

    messages.value = conversation
  } catch (error) {
    console.error('Error sending transcript:', error)
  }
}
</script>
