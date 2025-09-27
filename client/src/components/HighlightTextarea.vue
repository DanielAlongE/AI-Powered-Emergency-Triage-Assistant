<template>
  <div class="highlight-container" :class="containerClass" :style="containerStyle">
    <div 
      ref="editableDiv"
      class="highlight-textarea" 
      :style="textareaStyle"
      :contenteditable="!disabled && !readonly"
      @input="handleInput"
      @focus="handleFocus"
      @blur="handleBlur"
      v-html="displayText"
    ></div>
    <div v-if="showPlaceholder" class="placeholder">{{ placeholder }}</div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, watch } from 'vue'

const props = defineProps({
  modelValue: {
    type: String,
    default: ''
  },
  wordsToHighlight: {
    type: Array,
    default: () => []
  },
  placeholder: {
    type: String,
    default: ''
  },
  disabled: {
    type: Boolean,
    default: false
  },
  readonly: {
    type: Boolean,
    default: false
  },
  class: {
    type: String,
    default: ''
  },
  style: {
    type: Object,
    default: () => ({})
  },
  highlightClass: {
    type: String,
    default: 'highlight'
  },
  minHeight: {
    type: String,
    default: '200px'
  }
})

const emit = defineEmits(['update:modelValue', 'input', 'focus', 'blur'])

const editableDiv = ref(null)
const isFocused = ref(false)

const highlightedText = computed(() => {
  if (!props.modelValue) return ''
  const pattern = new RegExp(`\\b(${props.wordsToHighlight.join('|')})\\b`, 'gi')
  return props.modelValue.replace(pattern, `<span class="${props.highlightClass} sample">$1</span>`)
})

const displayText = computed(() => {
  return props.disabled || props.readonly ? props.modelValue : highlightedText.value
})

const showPlaceholder = computed(() => {
  return props.placeholder && !props.modelValue && !isFocused.value
})

const containerClass = computed(() => props.class)
const containerStyle = computed(() => ({ ...props.style }))
const textareaStyle = computed(() => ({ minHeight: props.minHeight }))

const handleInput = () => {
  const newValue = editableDiv.value.textContent
  emit('update:modelValue', newValue)
  emit('input', newValue)
}

const handleFocus = () => {
  isFocused.value = true
  emit('focus')
}

const handleBlur = () => {
  isFocused.value = false
  emit('blur')
}

const updateCursor = async () => {
  await nextTick()
  if (!editableDiv.value || props.disabled || props.readonly) return
  
  const div = editableDiv.value
  const range = document.createRange()
  const sel = window.getSelection()
  range.selectNodeContents(div)
  range.collapse(false)
  sel.removeAllRanges()
  sel.addRange(range)
}

// Watch for changes and update cursor position
watch(highlightedText, updateCursor)
</script>

<style>
.highlight {
  background-color: #dcff69;
  padding: 2px 0;
  border-radius: 2px;
}

.highlight-container {
  position: relative;
}

.highlight-textarea {
  border: 1px solid #ccc;
  padding: 8px;
  font-family: monospace;
  white-space: pre-wrap;
  word-wrap: break-word;
  outline: none;
  border-radius: 4px;
  background-color: white;
}

.highlight-textarea:focus {
  border-color: #007bff;
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

.highlight-textarea[contenteditable="false"] {
  background-color: #f8f9fa;
  cursor: not-allowed;
}

.placeholder {
  position: absolute;
  top: 8px;
  left: 8px;
  color: #6c757d;
  pointer-events: none;
  font-family: monospace;
}
</style>
