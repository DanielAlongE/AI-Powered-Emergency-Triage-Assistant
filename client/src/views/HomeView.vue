<script setup>
import { ref, inject, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const apiUrl = inject('$apiUrl')

const router = useRouter()

const sessionName = ref('')
const isSubmitting = ref(false)
const showSnackbar = ref(false)
const snackbarMessage = ref('')
const snackbarColor = ref('error')
const sessions = ref([])

const submitSession = async () => {
  if (!sessionName.value.trim()) return

  isSubmitting.value = true
  try {
    const response = await fetch(`${apiUrl}/api/v1/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ name: sessionName.value }),
    })
    if (response.ok) {
      sessionName.value = ''

      const { id } = await response.json()
      router.push(`/native/${id}`)
    } else {
      snackbarMessage.value = 'Failed to create session'
      showSnackbar.value = true
      isSubmitting.value = false
    }
  } catch {
    snackbarMessage.value = 'Error submitting session'
    showSnackbar.value = true
    isSubmitting.value = false
  }
}

const fetchSessions = async () => {
  try {
    const response = await fetch(`${apiUrl}/api/v1/sessions`)
    if (response.ok) {
      sessions.value = await response.json()
    } else {
      console.error('Failed to fetch sessions')
    }
  } catch (error) {
    console.error('Error fetching sessions:', error)
  }
}

onMounted(() => {
  fetchSessions()
})
</script>

<template>
  <v-card class="pa-4">
    <v-card-title>Create new session</v-card-title>
    <v-card-text>
      <v-text-field
        v-model="sessionName"
        label="Session Name"
        :input-style="{ fontWeight: 'bold' }"
        required
      ></v-text-field>
      <v-btn
        @click="submitSession"
        :loading="isSubmitting"
        :disabled="!sessionName.trim()"
        color="primary"
      >
        Submit
      </v-btn>
    </v-card-text>
  </v-card>

  <v-card class="pa-4 mt-4">
    <v-card-title>Existing Sessions</v-card-title>
    <v-card-text>
      <v-list v-if="sessions.length > 0">
        <v-list-item
          v-for="session in sessions"
          :key="session.id"
          @click="router.push(`/native/${session.id}`)"
        >
          <v-list-item-title>{{ session.name }}</v-list-item-title>
          <v-list-item-subtitle
            >Created: {{ new Date(session.created_at).toLocaleString() }}</v-list-item-subtitle
          >
        </v-list-item>
      </v-list>
      <p v-else>No sessions found.</p>
    </v-card-text>
  </v-card>

  <v-snackbar v-model="showSnackbar" :timeout="2000" :color="snackbarColor">
    {{ snackbarMessage }}
  </v-snackbar>
</template>
