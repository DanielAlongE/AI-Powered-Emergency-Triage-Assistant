<template>
  <div>
    <NextQuestionSuggestion :suggestion="suggestion" />
    <v-row>
      <v-col cols="4">
        <BrowserNativeSpeechRecognition
          v-if="isNativeModel"
          @update-transcript="updateTranscript"
          :redFlags="redFlagWords"
          :transcriptBase="transcript"
        />
        <VoskAudioTranscriber
          v-else
          @update-transcript="updateTranscript"
          :redFlags="redFlagWords"
          :transcriptBase="transcript"
        />
      </v-col>
      <v-col cols="4">
        <ChatConversation :transcript="transcript" :conversationBase="conversations" @update-conversations="updateConversations" />
      </v-col>
      <v-col cols="4">
        <TriageSummary
          @update-sugestions="updateSuggestions"
          @update-red-flag-terms="updateRedFlags"
          :conversations="conversations"
          :summaryBase="summary"
          :currentResponse="actualResponse"
        />
      </v-col>
    </v-row>
    <SuggestionAuditLog :suggestion="suggestion" :actualResponse="actualResponse" />
  </div>
</template>

<script setup>
import { computed, inject, ref, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import ChatConversation from '@/components/ChatConversation.vue'
import VoskAudioTranscriber from '@/components/VoskAudioTranscriber.vue'
import BrowserNativeSpeechRecognition from '@/components/BrowserNativeSpeechRecognition.vue'
import TriageSummary from '@/components/TriageSummary.vue'
import NextQuestionSuggestion from '@/components/NextQuestionSuggestion.vue'
import SuggestionAuditLog from '@/components/SuggestionAuditLog.vue'

const safeJsonParse = (str, defaultObject={}) => {
  if(!str) return defaultObject

  try {
    return JSON.parse(str);
  } catch (error) {
    console.error('Invalid JSON string:', error);
    return defaultObject;
  }
};

const route = useRoute()
const speechModel = route.params.speechModel
const isNativeModel = speechModel === 'native'

const sessionId = route.params.sessionId

const apiUrl = inject('$apiUrl')

const transcript = ref('')
const conversations = ref([])
const suggestion = ref('')
const redFlagTerms = ref({})
const summary = ref({})
const actualResponse = ref(null)

const redFlagWords = computed(() => Object.values(redFlagTerms.value))

watch(conversations, (convo) => {
  const lastMessage = convo.at(-1)
  const hasNurseResponse = lastMessage && ['NURSE', 'assistant'].includes(lastMessage.role)
  actualResponse.value = hasNurseResponse ? lastMessage.content : null
})

console.log({response: actualResponse.value})

const updateTranscript = (newTranscript) => {
  transcript.value = newTranscript
}

const updateConversations = (newConversations) => {
  conversations.value = newConversations
}

const updateSuggestions = (suggestions) => {
  suggestion.value = suggestions[0]
}

const updateRedFlags = (flags) => {
  flags.forEach((f) => {
    redFlagTerms.value[f] = f
  })
}

onMounted(async () => {
  if(!transcript.value){
    suggestion.value = 'Start by introducing yourself as the nurse!'
  }

  try {
    const response = await fetch(`${apiUrl}/api/v1/sessions/${sessionId}`)
    if (response.ok) {
      const sessionData = await response.json()
      transcript.value = sessionData.transcript || ''
      summary.value = safeJsonParse(sessionData.summary)
      conversations.value = safeJsonParse(sessionData.conversation)["conversation"] || []

      console.log(conversations.value, sessionData)
    } else {
      console.error('Failed to fetch session data')
    }
  } catch (error) {
    console.error('Error fetching session data:', error)
  }
})
</script>
