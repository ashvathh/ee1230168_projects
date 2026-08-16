# Force make to use Windows command prompt
SHELL := cmd.exe

# Toolchain Definitions
CC      = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
SIZE    = arm-none-eabi-size

# Project Parameters
TARGET    = firmware
C_SRCS    = src/main.c 
S_SRCS    = startup/startup_stm32f4.s 
LD_SCRIPT = ld/linker.ld 

# Object file list (generated from source list)
OBJS = $(C_SRCS:.c=.o) $(S_SRCS:.s=.o) 

# Compiler and Linker Flags
CFLAGS = -mcpu=cortex-m4 -mthumb -O0 -g \
-Wall -ffreestanding -nostdlib \
-mfpu=fpv4-sp-d16 -mfloat-abi=hard

ASFLAGS = $(CFLAGS)

LDFLAGS = -T $(LD_SCRIPT) -nostdlib -Wl,-Map=$(TARGET).map

# --- Build Rules ---

all: $(TARGET).bin

# Rule for C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule for Assembly files
%.o: %.s
	$(CC) $(ASFLAGS) -c $< -o $@

# Link to ELF
$(TARGET).elf: $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@
	$(SIZE) $@

# Convert ELF to Binary
$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

# Cleanup (Modified for Windows compatibility)
clean:
	del /f /q $(OBJS) $(TARGET).elf $(TARGET).bin $(TARGET).map