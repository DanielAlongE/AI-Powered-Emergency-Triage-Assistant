import { createVuetify } from 'vuetify'
import 'vuetify/styles' // Import Vuetify styles
import '@mdi/font/css/materialdesignicons.css' // Import Material Design Icons CSS
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'

export default createVuetify({
  components,
  directives,
  // Optional: Add custom themes or other configurations here
  // theme: {
  //   defaultTheme: 'light',
  //   themes: {
  //     light: {
  //       colors: {
  //         primary: '#1976D2',
  //         secondary: '#424242',
  //         accent: '#82B1FF',
  //         error: '#FF5252',
  //         info: '#2196F3',
  //         success: '#4CAF50',
  //         warning: '#FFC107',
  //       },
  //     },
  //   },
  // },
})
