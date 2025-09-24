<script setup>
import { ref, onMounted } from 'vue'
import ChatBubble from '@/components/ChatBubble.vue'

const transcript = ref('')
const isListening = ref(false)
const recognition = ref(null)

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
</script>

<template>
  <div>
    <v-container class="">
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
              <v-textarea v-model="transcript" label="Transcript" readonly rows="4"></v-textarea>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="4">
          <v-card class="elevation-5">
            <ChatBubble content="One" />
            <ChatBubble content="Two" primary />
          </v-card>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>
