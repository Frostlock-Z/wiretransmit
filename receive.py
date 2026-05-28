#!/usr/bin/env python3
"""
receive.py — 通过 3.5mm 音频输入口接收文件。

用法：
    python receive.py <输出文件>                  # 双向模式 (带握手ACK)
    python receive.py <输出文件> --raw --timeout N # 单向模式 (纯录音)
    python receive.py --list-devices              # 列出音频设备

务必先启动接收端，再启动发送端。
"""

import argparse, sys, os
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wiretransmit.constants import BAUD_RATES, SAMPLE_RATE, PREAMBLE_LEN
from wiretransmit.utils import list_devices, find_device, set_device


def run_raw(args) -> None:
    """单向模式：纯录音+解调，不发 ACK/NACK。"""
    from wiretransmit.demodulator import FSKDemodulator
    from wiretransmit.framing import parse_frame
    from wiretransmit.ecc import decode as rs_decode

    dem = FSKDemodulator(args.mode, args.baud)
    print(f"  单向接收模式 — 录音 {args.timeout}s ...  (请启动发送端)")
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

    try:
        decoded = rs_decode(payload)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WireTransmit — 通过 3.5mm 音频线接收文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python receive.py received.png\n"
               "  python receive.py output.dat --raw --timeout 60\n"
               "  python receive.py --list-devices\n\n"
               "务必先启动接收端，再启动发送端！",
    )
    parser.add_argument("output", nargs="?", metavar="输出文件", help="输出文件路径")
    parser.add_argument("--mode", choices=["2fsk","4fsk"], default="4fsk", help="调制方案 (需与发送端一致)")
    parser.add_argument("--baud", type=int, default=300, choices=BAUD_RATES, help=f"波特率 (需与发送端一致)")
    parser.add_argument("--timeout", type=float, default=120.0, help="最长监听时间（秒），默认: 120")
    parser.add_argument("--device", type=str, default=None, help="音频输入设备名称或索引")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备并退出")
    parser.add_argument("--raw", action="store_true", help="单向模式：不发 ACK/NACK，纯录音解调")
    args = parser.parse_args()

    if args.list_devices:
        list_devices(); return
    if args.output is None:
        parser.error("需要指定输出文件 (或使用 --list-devices)")

    if args.device:
        dev_id = find_device(args.device, "input")
        if dev_id is None:
            try: dev_id = int(args.device)
            except ValueError:
                print(f"错误: 未找到设备: {args.device}", file=sys.stderr); list_devices(); sys.exit(1)
        set_device(dev_id, "input")
        print(f"使用输入设备 #{dev_id}")

    if args.raw:
        run_raw(args)
    else:
        from wiretransmit.receiver import Receiver
        ok = Receiver(args.output, args.mode, args.baud, args.timeout).run()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
