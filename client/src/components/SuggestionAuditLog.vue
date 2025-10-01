<template>
  <v-card class="elevation-2 mt-4">
    <v-card-title>Audit Log</v-card-title>
    <v-data-table :items="tableItems"></v-data-table>
  </v-card>
</template>

<script setup>
const { suggestion, actualResponse } = defineProps({
  suggestion: { type: String, default: '' },
  actualResponse: { type: String, default: '' },
})

console.log({ suggestion, actualResponse })
const tableItems = [
  { suggestion: 'Sample suggestion 1', response: 'Sample response 2', similarity: '50%' },
  { suggestion: 'Sample suggestion 3', response: 'Sample response 4', similarity: '50%' },
  { suggestion: 'Sample suggestion 5', response: 'Sample response 6', similarity: '50%' },
]
</script>
