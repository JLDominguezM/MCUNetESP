#!/usr/bin/env python3
"""
Monitor serial no-interactivo para capturar logs del firmware durante N segundos.
Sustituye `idf.py monitor` cuando se quiere capturar output programáticamente.

Uso:
    python monitor.py /dev/ttyACM0 45      # 45 segundos
"""

import sys
import time
import serial

def main(port: str, seconds: float) -> int:
    # Open without touching DTR/RTS first (some USB-CDC chips reset on assertion).
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = 115200
    ser.timeout = 0.5
    ser.dtr = False
    ser.rts = False
    ser.open()
    # Pulse reset via RTS (typical esptool sequence): RTS high → low → high
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False
    time.sleep(0.1)
    print(f"# monitoring {port} for {seconds}s @ 115200 (after RTS reset)", flush=True)
    t_end = time.time() + seconds
    buf = b""
    while time.time() < t_end:
        chunk = ser.read(2048)
        if chunk:
            buf += chunk
            # Print newline-delimited lines as they arrive
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                print(line.decode("utf-8", "replace").rstrip("\r"), flush=True)
    if buf:
        print(buf.decode("utf-8", "replace"), flush=True)
    ser.close()
    return 0

if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    sys.exit(main(port, secs))
