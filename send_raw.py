#!/usr/bin/env python3
"""
send_raw.py — 单向强制发送（不做握手，不等 ACK）。

用法：
    python send_raw.py <文件路径>

适用于单根音频线、耳机口接收听信号但无法回传 ACK 的场景。
"""

import argparse, sys, os, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import sounddevice as sd
from wiretransmit.constants import SAMPLE_RATE, MAX_FILE_SIZE, PREAMBLE_BITS, SYNC_WORD, END_MARKER
from wiretransmit.modulator import FSKModulator
from wiretransmit.framing import build_frame_bits
from wiretransmit.ecc import encode as rs_encode


def main():
    parser = argparse.ArgumentParser(description="WireTransmit 单向强制发送")
    parser.add_argument("file", help="要发送的文件")
    parser.add_argument("--mode", default="4fsk", choices=["2fsk","4fsk"])
    parser.add_argument("--baud", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=3, help="重复发送次数 (默认3)")
    args = parser.parse_args()

    with open(args.file, "rb") as f:
        data = f.read()
    if len(data) > MAX_FILE_SIZE:
        print(f"错误: 文件过大 {len(data)} > {MAX_FILE_SIZE}")
        sys.exit(1)

    encoded = rs_encode(data)

    mod = FSKModulator(args.mode, args.baud)
    bits = build_frame_bits(encoded)
    signal = mod.modulate(bits)
    dur = len(signal) / SAMPLE_RATE
    bps = len(encoded) * 8 / dur

    print("=" * 52)
    print("  单向强制发送模式")
    print("=" * 52)
    print(f"  文件 : {os.path.basename(args.file)}")
    print(f"  大小 : {len(data):,d} B  ({len(data)/1024:.1f} KiB)")
    print(f"  RS编码后 : {len(encoded):,d} B")
    print(f"  时长 : {dur:.1f}s  速率 : ~{bps:.0f} bps")
    print(f"  重复 : {args.repeat} 次")
    print("=" * 52)

    for i in range(args.repeat):
        print(f"\n  [{i+1}/{args.repeat}] 发送中 ...")
        sd.play(signal.astype(np.float32), SAMPLE_RATE)
        sd.wait()
        if i < args.repeat - 1:
            time.sleep(0.5)

    print("\n  发送完成。")


if __name__ == "__main__":
    main()
