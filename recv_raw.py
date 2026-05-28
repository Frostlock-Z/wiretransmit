#!/usr/bin/env python3
"""
recv_raw.py — 单向强制接收（不做握手，不发 ACK）。

用法：
    python recv_raw.py <输出文件>

适用于单根音频线、接收端只能被动接收的场景。
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import sounddevice as sd
from wiretransmit.constants import SAMPLE_RATE, PREAMBLE_LEN
from wiretransmit.demodulator import FSKDemodulator
from wiretransmit.framing import parse_frame
from wiretransmit.ecc import decode as rs_decode


def main():
    parser = argparse.ArgumentParser(description="WireTransmit 单向强制接收")
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--mode", default="4fsk", choices=["2fsk","4fsk"])
    parser.add_argument("--baud", type=int, default=300)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    if args.device is not None:
        sd.default.device[0] = args.device

    dem = FSKDemodulator(args.mode, args.baud)
    print(f"  单向接收模式 — 监听 {args.timeout}s ...  (请启动发送端)")
    sys.stdout.flush()

    raw = sd.rec(int(args.timeout * SAMPLE_RATE),
                 samplerate=SAMPLE_RATE, channels=1,
                 dtype="float32")
    sd.wait()
    raw = raw.flatten()
    raw -= np.mean(raw)

    coarse = dem.find_sync(raw)
    if coarse is None:
        print("  失败 — 未检测到有效信号。")
        sys.exit(1)

    fine = dem.fine_sync(raw, coarse)
    print(f"  同步点: 样本 {fine} ({fine/SAMPLE_RATE:.2f}s)")

    bits, confs, _ = dem.demodulate(raw, start=fine)
    frame_bits = bits[PREAMBLE_LEN:]
    payload, err = parse_frame(frame_bits)
    if payload is None:
        print(f"  解析失败: {err}")
        sys.exit(1)

    qual = np.mean([abs(c - 0.5) * 2 for c in confs[:len(frame_bits)]])
    print(f"  信号质量: {qual * 100:.1f}%")
    print(f"  接收 {len(payload)} 字节 (RS 编码后)")

    # RS 解码
    try:
        decoded = rs_decode(payload)
        # 找原始文件末尾（去除补零）
        # 简单做法：找最后一个非零字节位置
        end = len(decoded)
        while end > 0 and decoded[end - 1] == 0:
            end -= 1
        decoded = decoded[:max(end, 1)]
    except Exception as e:
        print(f"  RS 解码失败: {e}，保存原始数据")
        decoded = payload

    with open(args.output, "wb") as f:
        f.write(decoded)
    print(f"  已保存: {args.output} ({len(decoded)} 字节)")
    print("  接收完成。")


if __name__ == "__main__":
    main()
