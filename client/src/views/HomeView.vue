<script setup>
import { ref, inject } from 'vue'
import { useRouter } from 'vue-router'

const apiUrl = inject('$apiUrl')

const router = useRouter()

const sessionName = ref('')
const isSubmitting = ref(false)
const showSnackbar = ref(false)
const snackbarMessage = ref('')
const snackbarColor = ref('error')

const submitSession = async () => {
  if (!sessionName.value.trim()) return

  isSubmitting.value = true
  try {
    const response = await fetch(`${apiUrl}/api/v1/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ name: sessionName.value })
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

  <v-snackbar
    v-model="showSnackbar"
    :timeout="2000"
    :color="snackbarColor"
  >
    {{ snackbarMessage }}
  </v-snackbar>
</template>
