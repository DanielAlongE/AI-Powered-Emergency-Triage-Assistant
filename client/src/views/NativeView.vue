<script setup>
import { ref, inject, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import ChatBubble from '@/components/ChatBubble.vue'

const apiUrl = inject('$apiUrl')

const router = useRouter()
const route = useRoute()
const transcript = ref('')
const isListening = ref(false)
const recognition = ref(null)
const messages = ref([])

const sessionId = route.params.sessionId

onMounted(() => {
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    recognition.value = new SpeechRecognition()
    recognition.value.continuous = true
    recognition.value.interimResults = true
    recognition.value.lang = 'en-US'

    recognition.value.onstart = () => {
      isListening.value = true
    }

    recognition.value.onresult = (event) => {
      let finalTranscript = ''
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript
        }
      }
      if (finalTranscript) {
        transcript.value += finalTranscript + '\n'
      }
    }

    recognition.value.onend = () => {
      isListening.value = false
    }

    recognition.value.onerror = (event) => {
      console.error('Speech recognition error:', event.error)
      isListening.value = false
    }
  } else {
    router.push(`/vosk/${sessionId}`)
    console.warn('Speech recognition not supported in this browser')
  }
})

const startListening = () => {
  if (recognition.value && !isListening.value) {
    recognition.value.start()
  }
}

const stopListening = () => {
  if (recognition.value && isListening.value) {
    recognition.value.stop()
  }
}

const sendTranscript = async () => {
  if (!transcript.value.trim()) return

  try {
    const response = await fetch(`${apiUrl}/api/v1/conversation?session_id=${sessionId}`, {
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

<template>
  <div>
    <v-row>
      <v-col cols="8">
        <v-card class="elevation-5 mt-4">
          <v-card-title>Speech Recognition</v-card-title>
          <v-card-text>
            <v-btn
              @click="isListening ? stopListening() : startListening()"
              :color="isListening ? 'red' : 'primary'"
              class="mb-4"
            >
              {{ isListening ? 'Stop Listening' : 'Start Listening' }}
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
