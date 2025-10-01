<template>
  <div>
    <v-card class="elevation-4 mt-4">
      <v-card-title>Suggestion</v-card-title>
      <v-card-text>
        <h1>Start by introducing yourself as the nurse!</h1>
      </v-card-text>
    </v-card>
    <v-row>
      <v-col cols="4">
        <VoskAudioTranscriber @update-transcript="updateTranscript" />
      </v-col>
      <v-col cols="4">
        <ChatConversation :transcript="transcript" @update-conversations="updateConversations" />
      </v-col>
      <v-col cols="4">
        <TriageSummary :conversations="conversations" />
      </v-col>
    </v-row>
    <v-card class="elevation-2 mt-4">
      <v-card-title>Audit Log</v-card-title>
      <v-data-table :items="tableItems"></v-data-table>
    </v-card>

  </div>
</template>

<script setup>
import { ref } from 'vue'
import ChatConversation from '@/components/ChatConversation.vue'
import VoskAudioTranscriber from '@/components/VoskAudioTranscriber.vue'
import TriageSummary from '@/components/TriageSummary.vue'

const transcript = ref('')
const conversations = ref([])

const updateTranscript = (newTranscript) => {
  transcript.value = newTranscript
}

const updateConversations = (newConversations) => {
  conversations.value = newConversations
}

const tableItems = [
  {suggestion: "Sample suggestion 1", response: "Sample response 2", similarity: '50%'},
  {suggestion: "Sample suggestion 3", response: "Sample response 4", similarity: '50%'},
  {suggestion: "Sample suggestion 5", response: "Sample response 6", similarity: '50%'},
]


</script>
