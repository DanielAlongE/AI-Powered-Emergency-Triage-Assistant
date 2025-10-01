<template>
  <v-card class="elevation-2 mt-4" height="400px">
    <v-card-title>Conversation</v-card-title>
    <v-card-text>
      <div ref="scrollContainer" class="scroll-container">
        <ChatBubble
          v-for="(msg, index) in messages"
          :key="index"
          :content="msg.content"
          :primary="['NURSE', 'assistant'].includes(msg.role)"
        />
        <v-skeleton-loader v-if="loading" type="text"></v-skeleton-loader>
      </div>
    </v-card-text>
  </v-card>
</template>

<script setup>
import { ref, inject, watch, nextTick } from 'vue'
import ChatBubble from '@/components/ChatBubble.vue'

const props = defineProps({
  transcript: {
    type: String,
    default: '',
  },
})

const apiUrl = inject('$apiUrl')
const emit = defineEmits(['update-conversations'])

const messages = ref([])
const loading = ref(false)
const scrollContainer = ref(null)

// Function to scroll to bottom
const scrollToBottom = () => {
  nextTick(() => {
    if (scrollContainer.value) {
      scrollContainer.value.scrollTop = scrollContainer.value.scrollHeight
    }
  })
}

// Debounce function
const debounce = (func, delay) => {
  let timeoutId
  return (...args) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => func.apply(null, args), delay)
  }
}

const sendTranscript = async () => {
  if (!props.transcript.trim()) return

  try {
    loading.value = true
    const response = await fetch(`${apiUrl}/api/v1/conversation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript: props.transcript }),
    })
    const { conversation } = await response.json()

    messages.value = conversation
    emit('update-conversations', messages.value)
    scrollToBottom()
  } catch (error) {
    console.error('Error sending transcript:', error)
  } finally {
    loading.value = false
  }
}

// Debounced version of sendTranscript with 2-second delay
const debouncedSendTranscript = debounce(sendTranscript, 2000)

// Watch for transcript changes and call debounced function
watch(() => props.transcript, debouncedSendTranscript)
</script>

<style>
.scroll-container {
  overflow-y: scroll;
  height: 330px;
}
</style>
