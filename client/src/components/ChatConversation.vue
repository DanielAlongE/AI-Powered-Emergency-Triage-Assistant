<template>
  <v-card class="elevation-2 mt-4" height="400px">
    <v-card-title>Conversation</v-card-title>
    <v-card-text>
      <div v-if="messages.length === 0 && !loading" class="text-center py-8">
        <v-icon size="64" color="grey">mdi-chat-outline</v-icon>
        <p class="mt-2">No conversation yet</p>
        <p class="text-caption">Start speaking to begin a conversation</p>
      </div>
      <div v-else ref="scrollContainer" class="scroll-container">
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

  <v-snackbar v-model="snackbar" :timeout="6000">
    {{ snackbarMessage }}
    <template v-slot:action="{ attrs }">
      <v-btn color="blue" text v-bind="attrs" @click="snackbar = false"> Close </v-btn>
    </template>
  </v-snackbar>
</template>

<script setup>
import { ref, inject, watch, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import ChatBubble from '@/components/ChatBubble.vue'

const props = defineProps({
  transcript: {
    type: String,
    default: '',
  },
  conversationBase: {
    type: Array,
    default: () => [],
  },
})

const apiUrl = inject('$apiUrl')
const emit = defineEmits(['update-conversations'])

const route = useRoute()
const messages = ref([])
const loading = ref(false)
const scrollContainer = ref(null)
const snackbar = ref(false)
const snackbarMessage = ref('')

const sessionId = route.params.sessionId

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
    timeoutId = setTimeout(() => func(...args), delay)
  }
}

const sendTranscript = async () => {
  if (!props.transcript.trim()) return

  try {
    scrollToBottom()
    loading.value = true
    const response = await fetch(`${apiUrl}/api/v1/conversation?session_id=${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript: props.transcript }),
    })
    const { conversation } = await response.json()

    messages.value = conversation
    emit('update-conversations', messages.value)
    scrollToBottom()
  } catch (error) {
    console.error('Error fetching conversation:', error)
    snackbarMessage.value = 'Error fetching conversation'
    snackbar.value = true
  } finally {
    loading.value = false
  }
}

// Debounced version of sendTranscript with 2-second delay
const debouncedSendTranscript = debounce(sendTranscript, 1000)

// Watch for transcript changes and call debounced function
watch(() => props.transcript, debouncedSendTranscript)

watch(
  () => props.conversationBase,
  () => {
    messages.value = props.conversationBase
  },
)
</script>

<style>
.scroll-container {
  overflow-y: scroll;
  height: 330px;
}
</style>
