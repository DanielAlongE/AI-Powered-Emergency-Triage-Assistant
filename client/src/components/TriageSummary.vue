<template>
    <v-card class="elevation-2 mt-4" height="400px" style="overflow-y: scroll;">
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
            <td colspan="2">
              <div class="font-weight-medium  mt-3 mb-2">Rationale</div>
              <div class="rationale-container">
                {{ summary.rationale }}
              </div>
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

const emit = defineEmits(['update-sugestions', 'update-red-flag-terms'])

const loading = ref(false)
const apiUrl = inject('$apiUrl')
const summary = ref({})

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

    if(data.rationale.includes("Error")) return

    summary.value = data
    emit('update-sugestions', data.follow_up_questions)
    emit('update-red-flag-terms', data.red_flag_terms)

  } catch (error) {
    console.error('Error fetching summary:', error)
  } finally{
    loading.value = false
  }
}

watch(() => conversations, fetchSummary)
</script>

<style scoped>
.rationale-container {
  height: 200px;
  overflow-y: scroll;
}
</style>