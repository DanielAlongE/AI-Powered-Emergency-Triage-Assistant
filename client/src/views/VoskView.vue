<template>
  <div>
    <NextQuestionSuggestion :suggestion="suggestion" />
    <v-row>
      <v-col cols="4">
        <VoskAudioTranscriber @update-transcript="updateTranscript" :redFlags="redFlagWords" />
      </v-col>
      <v-col cols="4">
        <ChatConversation :transcript="transcript" @update-conversations="updateConversations" />
      </v-col>
      <v-col cols="4">
        <TriageSummary @update-sugestions="updateSuggestions"
        @update-red-flag-terms="updateRedFlags" :conversations="conversations" />
      </v-col>
    </v-row>
    <v-card class="elevation-2 mt-4">
      <v-card-title>Audit Log</v-card-title>
      <v-data-table :items="tableItems"></v-data-table>
    </v-card>

  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import ChatConversation from '@/components/ChatConversation.vue'
import VoskAudioTranscriber from '@/components/VoskAudioTranscriber.vue'
import TriageSummary from '@/components/TriageSummary.vue'
import NextQuestionSuggestion from '@/components/NextQuestionSuggestion.vue'

const transcript = ref('')
const conversations = ref([])
const suggestion = ref('Start by introducing yourself as the nurse!')
const redFlagTerms = ref({})

const redFlagWords = computed(() => Object.values(redFlagTerms.value))

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

const tableItems = [
  {suggestion: "Sample suggestion 1", response: "Sample response 2", similarity: '50%'},
  {suggestion: "Sample suggestion 3", response: "Sample response 4", similarity: '50%'},
  {suggestion: "Sample suggestion 5", response: "Sample response 6", similarity: '50%'},
]


</script>
