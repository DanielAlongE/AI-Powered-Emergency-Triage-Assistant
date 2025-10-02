<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import HighlightTextarea from '@/components/HighlightTextarea.vue'


const { redFlags, transcriptBase } = defineProps({
  redFlags: { type: Array, default: () => [] },
  transcriptBase: { type: String, default: '' },
})

const emit = defineEmits(['update-transcript'])

const router = useRouter()
const route = useRoute()
const transcript = ref('')
const isListening = ref(false)
const recognition = ref(null)

const sessionId = route.params.sessionId

watch(transcript, () => {
  if(transcript.value === transcriptBase) return
  emit('update-transcript', transcript.value)
})

watch(() => transcriptBase, () => {
  transcript.value = transcriptBase
})


onMounted(() => {
  if (('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) && navigator.onLine) {
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
</script>

<template>
  <v-card class="elevation-2 mt-4" height="400px">
    <v-card-title>Speech Transcription</v-card-title>
    <v-card-text>
      <v-btn
        @click="isListening ? stopListening() : startListening()"
        :color="isListening ? 'red' : 'primary'"
        class="mb-4"
      >
        {{ isListening ? 'Stop Listening' : 'Start Listening' }}
      </v-btn>
      <HighlightTextarea
        v-model="transcript"
        label="Transcript"
        height="280px"
        :words-to-highlight="redFlags"
      ></HighlightTextarea>
    </v-card-text>
  </v-card>
</template>
