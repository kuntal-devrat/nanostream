#!/usr/bin/env python3
"""NanoStream-OD: Live Webcam & Microcontroller Simulation Demo.

Run with:
    python demo_webcam.py             # Auto-connects to webcam (or synthetic if no camera)
    python demo_webcam.py --synthetic # Run synthetic animated shapes benchmark
    python demo_webcam.py --int-mode  # Run bit-exact integer mode matching MCU kernel
"""

from nanostream.demo import main

if __name__ == "__main__":
    main()
