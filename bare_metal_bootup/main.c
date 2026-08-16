#include "semihosting.h" // include custom semihosting library

// systick register memory addresses
#define SYSTICK_CTRL  (*(volatile unsigned int*)0xE000E010)
#define SYSTICK_LOAD  (*(volatile unsigned int*)0xE000E014)
#define SYSTICK_VAL   (*(volatile unsigned int*)0xE000E018)

// --- Runtime Initialization Demo Variables ---
// 'initialized' goes into .data: stored in FLASH, copied to RAM by Reset_Handler.
// GDB should show its address in RAM (0x20000000) holding the value 123.
int initialized = 123;

// 'uninitialized' goes into .bss: allocated in RAM, zeroed by Reset_Handler.
// GDB should show its address in RAM (0x20000004) holding the value 0.
int uninitialized;

// --- SysTick counter ---
// Also in .bss (zero-initialized). Incremented by the interrupt handler.
volatile int tick = 0;

// this runs when the timer reaches zero
void SysTick_Handler(void)
{
    tick++; // increase the tick count
}

int main(void)
{
    sh_puts("--- STM32 Boot Sequence Started ---\n");
    sh_puts("Memory Initialization: SUCCESS\n");

    // set the reload value for the timer
    // 16 MHz / 1600000 = 10 Hz -> fires every 100 ms
    SYSTICK_LOAD = 1600000;
    
    // reset the current timer value to zero
    SYSTICK_VAL = 0;
    
    // enable systick, interrupt, and processor clock (7 is 111 in binary)
    SYSTICK_CTRL = 7;

    sh_puts("Reached main() successfully. SysTick Enabled. System Running.\n");

    // Track the last tick we reacted to
    int last_printed_tick = -1;

    while (1)
    {
        // 1. Ensure tick > 0 so we don't print instantly on boot
        // 2. Check if tick is a multiple of 100 
        // 3. Ensure we haven't already printed for this specific tick
        if (tick > 0 && (tick % 100) == 0 && tick != last_printed_tick)
        {
            last_printed_tick = tick; // Update our tracker
            sh_puts("INTERRUPT WORKING: 100 Ticks Passed!\n"); // Print once
        }
    }
}