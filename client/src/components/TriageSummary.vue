<template>
    <v-card class="elevation-2 mt-4 card-min-height">
      <v-card-title>Triage Summary</v-card-title>
      <v-table>
        <tbody>
          <tr>
            <td>ESI Level</td>
            <td>{{ summary.esi_level }}</td>
          </tr>
          <tr>
            <td>Confidence</td>
            <td>{{ summary.confidence }}</td>
          </tr>
          <tr>
            <td>Rationale</td>
            <td>{{ summary.rationale }}</td>
          </tr>
          <tr>
            <td>Follow up questions</td>
            <td>
              <div v-for="value in summary.follow_up_questions" :key="value">{{ value }}</div>
            </td>
          </tr>
        </tbody>
      </v-table>
    </v-card>
</template>
<script setup>
import { computed, inject, ref, watch } from 'vue'

const { conversations } = defineProps({
  conversations: {
    type: Array,
    default: () => []
  }
})

const loading = ref(false)
const apiUrl = inject('$apiUrl')
const summary = ref({})

console.log(summary)

const tableItems = computed(() => {
  return conversations.map(conv => ({ speaker: ['NURSE', 'assistant'].includes(conv.role) ? 'nurse': 'patient', message: conv.content }))
})

const fetchSummary = async () => {
  if (!Array.isArray(conversations) || !conversations.length) return

  try {
    loading.value = true
    const response = await fetch(`${apiUrl}/api/v1/triage-summary`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ turns: tableItems.value })
    })
    const data = await response.json()

    summary.value = data
  } catch (error) {
    console.error('Error fetching summary:', error)
  } finally{
    loading.value = false
  }
}

watch(() => conversations, fetchSummary)
</script>
