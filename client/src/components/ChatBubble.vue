<script setup>
defineProps({content: String, primary: Boolean})
</script>

<template>
    <div class="container">
        <v-card-text :class="['bubble', {'primary':primary}]">
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
        background-color: #0b93f6;
        color: #000;
        max-width: 80%;
    }

    .primary {
        background-color: #f7f7f8;
        color: #000;
        overflow-x:hidden;
        max-width: 70%;
        margin-left: auto;
    }
</style>