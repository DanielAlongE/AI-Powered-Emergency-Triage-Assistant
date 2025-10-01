<template>
  <div>
    <NextQuestionSuggestion :suggestion="suggestion" />
    <v-row>
      <v-col cols="4">
        <BrowserNativeSpeechRecognition v-if="isNativeModel" @update-transcript="updateTranscript" :redFlags="redFlagWords" />
        <VoskAudioTranscriber v-else @update-transcript="updateTranscript" :redFlags="redFlagWords" />
      </v-col>
      <v-col cols="4">
        <ChatConversation :transcript="transcript" @update-conversations="updateConversations" />
      </v-col>
      <v-col cols="4">
        <TriageSummary @update-sugestions="updateSuggestions"
        @update-red-flag-terms="updateRedFlags" :conversations="conversations" />
      </v-col>
    </v-row>
    <SuggestionAuditLog :suggestion="suggestion" :actualResponse="actualResponse" />
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRoute } from 'vue-router'
import ChatConversation from '@/components/ChatConversation.vue'
import VoskAudioTranscriber from '@/components/VoskAudioTranscriber.vue'
import BrowserNativeSpeechRecognition from '@/components/BrowserNativeSpeechRecognition.vue'
import TriageSummary from '@/components/TriageSummary.vue'
import NextQuestionSuggestion from '@/components/NextQuestionSuggestion.vue'
import SuggestionAuditLog from '@/components/SuggestionAuditLog.vue'



const route = useRoute()
const speechModel = route.params.speechModel
const isNativeModel = speechModel === 'native'


const transcript = ref('')
const conversations = ref([])
const suggestion = ref('Start by introducing yourself as the nurse!')
const redFlagTerms = ref({})

const redFlagWords = computed(() => Object.values(redFlagTerms.value))

const actualResponse = computed(() => { 
  const lastMessage = conversations.value.at(-1)

  if(lastMessage && ['NURSE', 'assistant'].includes(lastMessage.role)){
    return lastMessage.content
  }

  return null
})

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
  console.log({flags})
  flags.forEach((f) => {
    redFlagTerms.value[f] = f
  }) 
}

</script>
