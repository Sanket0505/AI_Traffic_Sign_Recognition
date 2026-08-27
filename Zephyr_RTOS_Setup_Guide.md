# Zephyr RTOS Setup Guide (Windows)

This guide explains how to:

1. Install Zephyr RTOS from scratch
2. Install QEMU for Windows
3. Build the Zephyr Blinky sample
4. Run Blinky on QEMU
5. Build and flash Blinky on real hardware

---

# Prerequisites

Before starting, ensure you have:

- Windows 10 or Windows 11
- Internet connection
- Administrator privileges for software installation
- Git installed

Git Download:

https://git-scm.com/download/win

---

# 1. Install Zephyr RTOS

The Zephyr Project provides a comprehensive and continuously updated Getting Started Guide. Always follow the official documentation for the latest dependency versions and installation requirements.

## Official Zephyr Getting Started Guide

https://docs.zephyrproject.org/latest/develop/getting_started/index.html

The guide covers:

- Installing Python
- Installing CMake
- Installing Ninja
- Installing Device Tree Compiler (DTC)
- Installing West
- Installing Zephyr SDK
- Creating a Zephyr workspace
- Downloading all Zephyr modules

Follow the **Windows** instructions from the official guide.

---

# 2. Verify Zephyr Installation

After completing the official setup, open a Zephyr Command Prompt, PowerShell, or terminal.

Verify West installation:

```bash
west --version
```

Expected output:

```text
West version: x.x.x
```

Verify your workspace:

```bash
cd zephyrproject
west list
```

You should see Zephyr and its modules listed.

---

# 3. Install QEMU on Windows

QEMU allows Zephyr applications to run on an emulator without requiring physical hardware.

## Official QEMU Downloads

https://www.qemu.org/download/

---

## Option 1: Windows Installer (Recommended)

Download the latest Windows installer from:

https://qemu.eu/

### Installation Steps

1. Download the latest 64-bit installer.
2. Run the installer.
3. Install QEMU to:

```text
C:\Program Files\qemu
```

4. Add the installation directory to your Windows PATH.

Example:

```text
C:\Program Files\qemu
```

5. Open a new Command Prompt.

Verify the installation:

```bash
qemu-system-x86_64 --version
```

Expected output:

```text
QEMU emulator version x.x.x
```

If a version is displayed, QEMU is installed correctly.

---

# 4. Build the Blinky Sample

The Blinky sample is Zephyr's "Hello World" application.

Navigate to the Zephyr repository:

```bash
cd zephyrproject\zephyr
```

Build for the QEMU x86 target:

```bash
west build -b qemu_x86 samples/basic/blinky
```

Successful output will contain:

```text
Build files have been written to:
build
```

---

# 5. Run Blinky on QEMU

After a successful build, launch the emulator:

```bash
west build -t run
```

A QEMU window will open.

You should see output similar to:

```text
*** Booting Zephyr OS ***
LED state: ON
LED state: OFF
LED state: ON
LED state: OFF
```
