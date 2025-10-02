<template>
  <v-card class="elevation-2 mt-4" height="400px" style="overflow-y: scroll">
    <v-card-title>Triage Summary</v-card-title>
    <v-table>
      <tbody>
        <tr>
          <td>
            ESI Level
            <v-icon @click="showDialog = true" size="small" class="ml-1"
              >mdi-information-outline</v-icon
            >
          </td>
          <td>
            {{ summary.esi_level }}
            <SeverityLevelBadge v-if="summary.esi_level" :level="summary.esi_level" />
          </td>
        </tr>
        <tr>
          <td>Confidence</td>
          <td>{{ summary.confidence }}</td>
        </tr>
        <tr>
          <td colspan="2">
            <div class="font-weight-medium mt-3 mb-2">Rationale</div>
            <div class="rationale-container">
              {{ summary.rationale }}
            </div>
          </td>
        </tr>
      </tbody>
    </v-table>
  </v-card>

  <v-dialog v-model="showDialog" max-width="600px">
    <v-card>
      <v-card-title>About ESI Levels</v-card-title>
      <v-card-text>
        <p>
          The Emergency Severity Index (ESI) is a five-level emergency department triage algorithm
          that provides clinically relevant stratification of patients into five groups from 1 (most
          urgent) to 5 (least urgent) on the basis of acuity and resource needs.
        </p>
        <ul>
          <li>
            <strong>Level 1:</strong> Immediate, life-saving intervention required without delay.
          </li>
          <li>
            <strong>Level 2:</strong> High risk situation, requires immediate assessment and
            intervention.
          </li>
          <li>
            <strong>Level 3:</strong> Urgent, but stable; requires assessment and treatment within
            30 minutes.
          </li>
          <li>
            <strong>Level 4:</strong> Less urgent; requires assessment and treatment within 1 hour.
          </li>
          <li>
            <strong>Level 5:</strong> Least urgent; can wait to be seen in the treatment area.
          </li>
        </ul>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn color="primary" @click="showDialog = false">Close</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>

  <v-snackbar v-model="snackbar" :timeout="6000">
    {{ snackbarMessage }}
    <template v-slot:action="{ attrs }">
      <v-btn color="blue" text v-bind="attrs" @click="snackbar = false"> Close </v-btn>
    </template>
  </v-snackbar>
</template>
<script setup>
import { computed, inject, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import SeverityLevelBadge from '@/components/SeverityLevelBadge.vue'

const { conversations, summaryBase, currentResponse } = defineProps({
  conversations: {
    type: Array,
    default: () => [],
  },
  summaryBase: {
    type: Object,
    default: () => {},
  },
  currentResponse: {
    type: String,
    default: '',
  },
})

const emit = defineEmits(['update-sugestions', 'update-red-flag-terms'])

const route = useRoute()
const loading = ref(false)
const apiUrl = inject('$apiUrl')
const summary = ref({})
const showDialog = ref(false)
const snackbar = ref(false)
const snackbarMessage = ref('')

const sessionId = route.params.sessionId

const tableItems = computed(() => {
  return conversations.map((conv) => ({
    speaker: ['NURSE', 'assistant'].includes(conv.role) ? 'nurse' : 'patient',
    message: conv.content,
  }))
})

const fetchSummary = async () => {
  if (!Array.isArray(conversations) || !conversations.length) return

  try {
    loading.value = true
    const response = await fetch(`${apiUrl}/api/v1/triage-summary?session_id=${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ turns: tableItems.value }),
    })
    const data = await response.json()

    if (data.rationale.includes('Error')) return

    summary.value = data
    if (!currentResponse) {
      emit('update-sugestions', data.follow_up_questions)
    }
    emit('update-red-flag-terms', data.red_flag_terms)
  } catch (error) {
    console.error(error)
    snackbarMessage.value = 'Error fetching summary!'
    snackbar.value = true
  } finally {
    loading.value = false
  }
}

watch(
  () => summaryBase,
  () => {
    summary.value = summaryBase
  },
)

watch(() => conversations, fetchSummary)
</script>

<style scoped>
.rationale-container {
  height: 200px;
  overflow-y: scroll;
}
</style>
