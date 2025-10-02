<template>
  <v-card class="elevation-4 mt-4">
    <v-card-title>Suggestion</v-card-title>
    <v-card-text>
      <div :class="['text-h4', { 'text-grey': isGrey }]">{{ suggestion }}</div>
    </v-card-text>
  </v-card>
</template>
<script setup>
import { ref, watch } from 'vue'

const props = defineProps({ suggestion: String })

const isGrey = ref(false)
let timeoutId = null

watch(
  () => props.suggestion,
  () => {
    isGrey.value = false
    if (timeoutId) clearTimeout(timeoutId)
    timeoutId = setTimeout(() => {
      isGrey.value = true
    }, 60000)
  },
  { immediate: true },
)
</script>
