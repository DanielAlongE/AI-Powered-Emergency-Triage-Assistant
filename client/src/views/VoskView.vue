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
            <v-textarea v-model="transcript" label="Transcript" rows="4"></v-textarea>
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
import { ref } from 'vue'
import ChatBubble from '@/components/ChatBubble.vue'

const transcript = ref('')
const isRecording = ref(false)
const mediaRecorder = ref(null)
const audioChunks = ref([])
const messages = ref([])

const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder.value = new MediaRecorder(stream)
    audioChunks.value = []

    mediaRecorder.value.ondataavailable = (event) => {
      audioChunks.value.push(event.data)
    }

    mediaRecorder.value.onstop = async () => {
      const audioBlob = new Blob(audioChunks.value, { type: 'audio/webm' })
      const formData = new FormData()
      formData.append('audio', audioBlob)

      try {
        const response = await fetch('http://localhost:8000/api/v1/transcribe', {
          method: 'POST',
          body: formData
        })
        const data = await response.json()
        transcript.value = data.transcript
      } catch (error) {
        console.error('Error transcribing:', error)
      }
    }

    mediaRecorder.value.start()
    isRecording.value = true
  } catch (error) {
    console.error('Error accessing microphone:', error)
  }
}

const stopRecording = () => {
  if (mediaRecorder.value && isRecording.value) {
    mediaRecorder.value.stop()
    isRecording.value = false
  }
}

const sendTranscript = async () => {
  if (!transcript.value.trim()) return

  try {
    const response = await fetch('http://localhost:8000/api/conversation', {
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
