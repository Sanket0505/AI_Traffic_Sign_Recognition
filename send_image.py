#!/usr/bin/env python3
"""Send a 48x48x3 RGB image to the board over UART and print the prediction.

Protocol (matches src/qemu_app/src/main.c):
  1. Board boots and prints "Ready. Waiting for images on UART ...".
  2. Host sends the 2-byte sync marker 0xAA 0x55.
  3. Host sends exactly 48*48*3 = 6912 raw RGB bytes (row-major, uint8).
  4. Board runs inference and replies with the predicted class id as ASCII
     followed by a newline (e.g. "43\n").

UART is fixed at 115200 baud, 8N1 (submission requirement).

Usage:
    python send_image.py COM7 sign.png
    python send_image.py /dev/ttyACM0 sign.raw --raw
"""

import argparse
import sys
import time

import serial  # pyserial

WIDTH = 48
HEIGHT = 48
CHANNELS = 3
NUM_BYTES = WIDTH * HEIGHT * CHANNELS  # 6912

BAUDRATE = 115200
SYNC_MARKER = b"\xAA\x55"

READY_TIMEOUT_S = 30.0
RESULT_TIMEOUT_S = 30.0


def load_payload(path: str, raw: bool) -> bytes:
    if raw:
        with open(path, "rb") as f:
            data = f.read()
    else:
        from PIL import Image  # pillow

        img = Image.open(path).convert("RGB").resize((WIDTH, HEIGHT))
        data = img.tobytes()

    if len(data) != NUM_BYTES:
        raise SystemExit(f"expected {NUM_BYTES} bytes (48x48x3), got {len(data)}")
    return data


def read_and_echo_line(port: serial.Serial) -> str | None:
    line = port.readline().decode(errors="ignore").strip()
    if line:
        print(line)
        return line
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("image")
    parser.add_argument("--baud", type=int, default=BAUDRATE,
                        help="UART baud rate (default: 115200)")
    parser.add_argument("--raw", action="store_true",
                        help="image file is already 48x48x3 raw RGB")
    parser.add_argument("--no-wait", action="store_true",
                        help="do not wait for the board 'Ready' banner")
    args = parser.parse_args()

    payload = load_payload(args.image, args.raw)

    # 115200 baud, 8 data bits, no parity, 1 stop bit (8N1).
    with serial.Serial(args.port, args.baud, bytesize=serial.EIGHTBITS,
                       parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                       timeout=0.1) as port:
        # Wait for the board to announce it is ready.
        if not args.no_wait:
            ready_deadline = time.monotonic() + READY_TIMEOUT_S
            while time.monotonic() < ready_deadline:
                line = read_and_echo_line(port)
                if line and "Ready" in line:
                    break
            else:
                print("warning: no 'Ready' banner seen; sending anyway")

        # Send sync marker + raw image.
        send_start = time.monotonic()
        port.write(SYNC_MARKER)
        port.write(payload)
        port.flush()
        print(f"Sent {len(payload)} bytes in {time.monotonic() - send_start:.3f}s")

        # Read the predicted class id (first integer-only line).
        result_deadline = time.monotonic() + RESULT_TIMEOUT_S
        while time.monotonic() < result_deadline:
            line = read_and_echo_line(port)
            if line is None:
                continue
            token = line.strip().lstrip("-")
            if token.isdigit():
                print(f"PREDICTION: class_id={line.strip()}")
                return 0

        raise SystemExit("timed out waiting for prediction")


if __name__ == "__main__":
    sys.exit(main())
