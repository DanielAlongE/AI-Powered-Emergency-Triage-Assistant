<template>
    <v-card class="elevation-4 mt-4">
      <v-card-title>Suggestion</v-card-title>
      <v-card-text>
        <h1>{{ suggestion }}</h1>
      </v-card-text>
    </v-card>
</template>
<script setup>
defineProps({suggestion: String})
</script>