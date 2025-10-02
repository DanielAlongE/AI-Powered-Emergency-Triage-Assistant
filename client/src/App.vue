<script setup>
import { RouterView, useRouter } from 'vue-router'
import { useTheme } from 'vuetify'

// RouterLink,

const router = useRouter()
const theme = useTheme()

const toggleTheme = () => {
  theme.global.name.value = theme.global.name.value === 'light' ? 'dark' : 'light'
}
</script>

<template>
  <v-app>
    <v-app-bar :elevation="2">
      <template v-slot:prepend>
        <v-app-bar-nav-icon></v-app-bar-nav-icon>
      </template>

      <v-app-bar-title @click="router.push('/')" style="cursor: pointer"
        >AI-Powered Emergency Triage Assistant</v-app-bar-title
      >

      <template v-slot:append>
        <v-btn icon @click="toggleTheme">
          <v-icon>mdi-theme-light-dark</v-icon>
        </v-btn>
      </template>
    </v-app-bar>
    <v-main>
      <v-container>
        <RouterView />
      </v-container>
    </v-main>
  </v-app>
</template>
