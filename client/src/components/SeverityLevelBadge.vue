<template>
<v-chip :color="item.color" variant="outlined">
  {{ item.label }}
</v-chip>
    
</template>
<script setup>
import { computed } from 'vue';

const { level } = defineProps('level')

const items = [
    {key:'1', color:'red', label:'Immediate'},
    {key:'2', color:'red', label:'Emergency'},
    {key:'3', color:'yellow', label:'Urgent'},
    {key:'4', color:'yellow', label:'Semi-urgent'},
    {key:'5', color:'grey', label:'Nonurgent'},
]

const item = computed(() => {
    const currentItem = items.find((item) => item.key === level)
    if(currentItem) return currentItem

    return items.at(-1)
    })
</script>