#pragma once // prevent this file from being included multiple times

// the operation number for writing a string to the debugger console
#define SEMIHOSTING_SYS_WRITE0 0x04

// core function to trigger the semihosting interrupt
static inline int semihosting_call(int reason, void *arg)
{
    int value;
    
    // volatile tells the compiler not to optimize this assembly away
    __asm volatile (
        "mov r0, %1\n"          // put the operation reason code into register r0
        "mov r1, %2\n"          // put our argument (the string pointer) into r1
        "bkpt 0xAB\n"           // trigger the specific semihosting software breakpoint
        "mov %0, r0\n"          // save the debugger's return value from r0 into our variable
        : "=r"(value)           // output operand
        : "r"(reason), "r"(arg) // input operands 
        : "r0", "r1", "memory"  // tell compiler these registers were modified
    );
    
    return value;
}

// simple wrapper function so we don't have to type the raw assembly call every time
static inline void sh_puts(const char *s)
{
    semihosting_call(SEMIHOSTING_SYS_WRITE0, (void*)s); // trigger the write call
}
