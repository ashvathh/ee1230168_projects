ENTRY(Reset_Handler)
MEMORY
{
  FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 1024K
  RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 128K
}
_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  /* vector table stays completely in flash(perm memory) */
  .isr_vector : { *(.isr_vector) } > FLASH
  .text : { *(.text*) } > FLASH
  .rodata : { *(.rodata*) } > FLASH

  /* data section is stored in flash but lives in RAM */
  /* takes care of initialized variables stored in memory */
  .data : {
    _sdata = .;        
    *(.data*)      /* local initialized variables */
    _edata = .;        
  } > RAM AT > FLASH   

  _sidata = LOADADDR(.data);

  /* bss section lives in RAM */
  .bss : {
    _sbss = .;         
    *(.bss*)       /* local uninitialized variables */  
    *(COMMON)      /* standard C uninitialized globals */
    _ebss = .;        
  } > RAM
}
