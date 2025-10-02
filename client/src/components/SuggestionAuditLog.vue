<!-- eslint-disable vue/valid-v-slot -->
<template>
  <!-- eslint-disable vue/v-slot-style -->
  <v-card class="elevation-2 mt-4">
    <v-card-title>Audit Log</v-card-title>
    <v-data-table :items="tableItems" :headers="headers" :loading="loading">
      <template v-slot:item.similarity="{ item }">
        {{ item.similarity }}%
      </template>
      <template v-slot:item.created_at="{ item }">
        {{ new Date(item.created_at).toLocaleString() }}
      </template>
    </v-data-table>
  </v-card>
  <!-- eslint-enable vue/v-slot-style -->
</template>

<script setup>
import { ref, onMounted, inject, watch } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const sessionId = route.params.sessionId

const { suggestion, actualResponse } = defineProps({
  suggestion: { type: String, default: '' },
  actualResponse: { type: String, default: '' },
})

const apiUrl = inject('$apiUrl')
const tableItems = ref([])
const loading = ref(false)
const suggestionsSet = ref(new Set())

const headers = [
  { title: 'Suggestion', key: 'suggestion' },
  { title: 'Actual Response', key: 'response' },
  { title: 'Similarity', key: 'similarity' },
  { title: 'Created At', key: 'created_at' },
]

const fetchAuditLogs = async () => {
  try {
    loading.value = true
    const response = await fetch(`${apiUrl}/api/v1/audit-logs`)
    const data = await response.json()
    tableItems.value = data
  } catch (error) {
    console.error('Error fetching audit logs:', error)
  } finally {
    loading.value = false
  }
}

const createAuditLog = async (suggestion, response, sessionId) => {
  if (!suggestion || !response || !sessionId) return

  try {
    const auditLogData = {
      session_id: sessionId,
      suggestion: suggestion,
      response: response,
      similarity: 0 // Default similarity, can be calculated later if needed
    }

    const res = await fetch(`${apiUrl}/api/v1/audit-logs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(auditLogData)
    })

    if (res.ok) {
      // Optionally refresh the table
      fetchAuditLogs()
    } else {
      console.error('Failed to create audit log')
    }
  } catch (error) {
    console.error('Error creating audit log:', error)
  }
}

watch(() => [suggestion, actualResponse], ([newSuggestion, actualResponse]) => {
  console.log({newSuggestion, actualResponse})
  if (newSuggestion && actualResponse && sessionId && !suggestionsSet.value.has(newSuggestion)) {
    suggestionsSet.value.add(newSuggestion)
    createAuditLog(newSuggestion, actualResponse, sessionId)
  }
})

onMounted(fetchAuditLogs)
</script>
