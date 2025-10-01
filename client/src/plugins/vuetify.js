import { createVuetify } from 'vuetify'
import 'vuetify/styles' // Import Vuetify styles
import '@mdi/font/css/materialdesignicons.css' // Import Material Design Icons CSS
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'

export default createVuetify({
  components,
  directives,
  theme: {
    defaultTheme: 'light',
    themes: {
      light: {
        colors: {
          primary: '#1976D2',
          secondary: '#424242',
          accent: '#82B1FF',
          error: '#FF5252',
          info: '#2196F3',
          success: '#4CAF50',
          warning: '#FFC107',
        },
      },
      dark: {
        colors: {
          primary: '#BB86FC',
          secondary: '#03DAC6',
          accent: '#CF6679',
          error: '#CF6679',
          info: '#2196F3',
          success: '#03DAC6',
          warning: '#FBC02D',
          background: '#121212',
          surface: '#1E1E1E',
        },
      },
    },
  },
})
