.syntax unified 
.cpu cortex-m4 
.thumb 

// pull in the variables defined in the linker script
.extern _estack 
.extern _sidata 
.extern _sdata 
.extern _edata 
.extern _sbss 
.extern _ebss 
.extern SysTick_Handler

.global g_pfnVectors 
.global Reset_Handler 

// Put vector table at the very front of flash memory
.section .isr_vector,"a",%progbits 

// Cortex-M4 vector table — entries 0-15 are fixed ARM system exceptions.
// Entry index = exception number. SysTick is exception #15 (offset 0x3C).
// Any gap between UsageFault (#7) and SysTick (#15) must be filled with
// reserved words so the hardware reads the correct slot.
g_pfnVectors: 
    .word _estack            // Entry  0: Initial Stack Pointer value
    .word Reset_Handler      // Entry  1: Reset
    .word Default_Handler    // Entry  2: NMI
    .word Default_Handler    // Entry  3: HardFault
    .word Default_Handler    // Entry  4: MemManage
    .word Default_Handler    // Entry  5: BusFault
    .word Default_Handler    // Entry  6: UsageFault
    .word 0                  // Entry  7: Reserved
    .word 0                  // Entry  8: Reserved
    .word 0                  // Entry  9: Reserved
    .word 0                  // Entry 10: Reserved
    .word Default_Handler    // Entry 11: SVCall
    .word Default_Handler    // Entry 12: Debug Monitor
    .word 0                  // Entry 13: Reserved
    .word Default_Handler    // Entry 14: PendSV
    .word SysTick_Handler    // Entry 15: SysTick  <-- correct slot

.section .text.Reset_Handler,"ax",%progbits  
.type Reset_Handler, %function  // Ensures the linker tags this as a Thumb function (sets the LSB to 1)

Reset_Handler: 
    /* copy .data */ 
    ldr r0, =_sidata         // source address in flash
    ldr r1, =_sdata          // destination start in ram
    ldr r2, =_edata          // destination end in ram
1:  
    cmp r1, r2               // check if we reached the end
    bcc 2f 
    b 3f 
2:  
    /* increment the pointer in data section by 4 bytes */
    ldr r3, [r0], #4         // load word and move forward
    str r3, [r1], #4         // store word and move forward
    b 1b 
3: 
    /* zero .bss */ 
    ldr r1, =_sbss           // start of bss in ram
    ldr r2, =_ebss           // end of bss
    movs r3, #0              // prepare our zero value
4:  
    cmp r1, r2 
    bcc 5f 
    b 6f 
5:  
    /* increment the pointer in bss section by 4 bytes */
    str r3, [r1], #4         // write zero and move forward
    b 4b 
6: 
    bl main                  // jump to our c code
7:  
    b 7b                     // trap just in case main returns

/* --- The Safety Net --- */
.section .text.Default_Handler,"ax",%progbits
.type Default_Handler, %function

Default_Handler:
    b Default_Handler    // Branch back to the Default_Handler label forever