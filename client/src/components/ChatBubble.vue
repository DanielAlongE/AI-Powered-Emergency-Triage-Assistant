<script setup>
defineProps({ content: String, primary: Boolean })

</script>

<template>
  <div class="container">
    <v-card-text :class="['bubble', primary ? 'bg-primary primary' : 'bg-accent' ]">
      {{ content }}
    </v-card-text>
  </div>
</template>

<style scoped>
.container {
  display: flex;
  margin-bottom: 12px;
  align-items: flex-start;
}

.bubble {
  max-width: 80%;
}

.primary {
  overflow-x: hidden;
  max-width: 70%;
  margin-left: auto;
}
</style>
