import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import vuetify from './plugins/vuetify'

const app = createApp(App)


app.provide('$apiUrl', import.meta.env.VITE_API_URL || "http://localhost:8000")

app.use(router)
app.use(vuetify)
app.mount('#app')
