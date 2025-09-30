<script setup>
import { ref } from 'vue'

const sessionName = ref('')
const isSubmitting = ref(false)

const submitSession = async () => {
  if (!sessionName.value.trim()) return

  isSubmitting.value = true
  try {
    const response = await fetch('/api/v1/sessions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ name: sessionName.value })
    })
    if (response.ok) {
      sessionName.value = ''
    } else {
      console.error('Failed to create session')
    }
  } catch (error) {
    console.error('Error submitting session:', error)
  } finally {
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
</template>
