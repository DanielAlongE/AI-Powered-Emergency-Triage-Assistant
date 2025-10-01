<template>
  <div class="highlight-container"
  :class="containerClass"
  :style="containerStyle"
  >
    <div
      ref="editableDiv"
      class="highlight-textarea overflow-y-auto"
      :style="textareaStyle"
      :contenteditable="!disabled && !readonly"
      @input="handleInput"
      @focus="handleFocus"
      @blur="handleBlur"
      @keydown="handleKeyDown"
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
    default: 'bg-error'
  },
  minHeight: {
    type: String,
    default: '280px'
  }
})

const emit = defineEmits(['update:modelValue', 'input', 'focus', 'blur'])

const editableDiv = ref(null)
const isFocused = ref(false)
const savedPosition = ref(null)

const highlightedText = computed(() => {
  if (!props.modelValue) return ''
  // Convert newlines to <br> tags for HTML display
  let text = props.modelValue.replace(/\n/g, '<br>')
  // console.log(editableDiv.value.innerHTML, editableDiv.value.textContent)
  const pattern = new RegExp(`\\b(${props.wordsToHighlight.join('|')})\\b`, 'gi')
  return text.replace(pattern, `<span class="${props.highlightClass}">$1</span>`)
})

const prepareRawText = (text) => text.replace(/\n/g, '<br>')

const displayText = computed(() => {
  return props.disabled || props.readonly ? prepareRawText(props.modelValue) : prepareRawText(highlightedText.value)
})

const showPlaceholder = computed(() => {
  return props.placeholder && !props.modelValue && !isFocused.value
})

const containerClass = computed(() => props.class)
const containerStyle = computed(() => ({ ...props.style }))
const textareaStyle = computed(() => ({ minHeight: props.minHeight, height:'280px', whiteSpace: 'pre-wrap' }))

const saveCursorPosition = () => {
  const sel = window.getSelection()
  if (sel.rangeCount > 0) {
    const range = sel.getRangeAt(0)
    const preCaretRange = range.cloneRange()
    preCaretRange.selectNodeContents(editableDiv.value)
    preCaretRange.setEnd(range.startContainer, range.startOffset)
    const start = preCaretRange.toString().length
    preCaretRange.setEnd(range.endContainer, range.endOffset)
    const end = preCaretRange.toString().length
    return { start, end }
  }
  return null
}

const restoreCursorPosition = (pos) => {
  if (!pos) return
  const sel = window.getSelection()
  const range = document.createRange()
  const textNodes = getTextNodes(editableDiv.value)
  let charCount = 0
  let startNode = null
  let startOffset = 0
  for (let node of textNodes) {
    const nextCharCount = charCount + node.length
    if (pos.start <= nextCharCount) {
      startNode = node
      startOffset = pos.start - charCount
      break
    }
    charCount = nextCharCount
  }
  charCount = 0
  let endNode = null
  let endOffset = 0
  for (let node of textNodes) {
    const nextCharCount = charCount + node.length
    if (pos.end <= nextCharCount) {
      endNode = node
      endOffset = pos.end - charCount
      break
    }
    charCount = nextCharCount
  }
  if (startNode) range.setStart(startNode, startOffset)
  if (endNode) range.setEnd(endNode, endOffset)
  sel.removeAllRanges()
  sel.addRange(range)
}

const getTextNodes = (element) => {
  const textNodes = []
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null, false)
  let node
  while ((node = walker.nextNode())) {
    textNodes.push(node)
  }
  return textNodes
}

const handleInput = () => {
  savedPosition.value = saveCursorPosition()
  const newValue = editableDiv.value.innerText
  
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

  if (savedPosition.value) {
    restoreCursorPosition(savedPosition.value)
    savedPosition.value = null
  } else {
    const div = editableDiv.value
    const range = document.createRange()
    const sel = window.getSelection()
    range.selectNodeContents(div)
    range.collapse(false)
    sel.removeAllRanges()
    sel.addRange(range)
  }
}

const handleKeyDown = (e) => {
    // Check for the Enter key (keyCode 13) and prevent default behavior.
    if (e.keyCode === 13) {
        e.preventDefault();

        // Insert a line break element (<br>) instead.
        const selection = window.getSelection();
        const range = selection.getRangeAt(0);
        const br1 = document.createElement('br');
        const br2 = document.createElement('br');
        range.deleteContents();
        range.insertNode(br1);
        range.insertNode(br2);
        range.setStartAfter(br2);
        range.setEndAfter(br2);
        selection.removeAllRanges();
        selection.addRange(range);

    }
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
