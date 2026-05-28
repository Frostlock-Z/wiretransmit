#!/usr/bin/env python3
"""
send.py — 通过 3.5mm 音频输出口发送文件。

用法：
    python send.py <文件路径>                     # 双向模式 (带握手)
    python send.py <文件路径> --raw               # 单向模式 (无握手)
    python send.py --list-devices                 # 列出音频设备
    python send.py --test-tone                    # 播放测试音

双向模式需要两端都有扬声器+麦克风，单向模式适用于单根 TRRS 线场景。
"""

import argparse, sys, os, time
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wiretransmit.constants import BAUD_RATES, SAMPLE_RATE, MAX_FILE_SIZE
from wiretransmit.utils import list_devices, find_device, set_device


def run_raw(args) -> None:
    """单向模式：纯粹调制播放，不做握手不等 ACK。"""
    from wiretransmit.modulator import FSKModulator
    from wiretransmit.framing import build_frame_bits
    from wiretransmit.ecc import encode as rs_encode

    with open(args.file, "rb") as f:
        data = f.read()
    if len(data) > MAX_FILE_SIZE:
        print(f"错误: 文件过大 {len(data)} > {MAX_FILE_SIZE}", file=sys.stderr)
        sys.exit(1)

    encoded = rs_encode(data)
    mod = FSKModulator(args.mode, args.baud)
    bits = build_frame_bits(encoded)
    signal = mod.modulate(bits)
    dur = len(signal) / SAMPLE_RATE
    bps = len(encoded) * 8 / dur
    repeat = getattr(args, 'repeat', 3)
    total_dur = dur * repeat + 0.5 * (repeat - 1)

    print("=" * 52)
    print("  单向强制发送模式")
    print("=" * 52)
    print(f"  文件 : {os.path.basename(args.file)}")
    print(f"  大小 : {len(data):,d} B  ({len(data)/1024:.1f} KiB)")
    print(f"  RS编码后 : {len(encoded):,d} B")
    print(f"  单次时长 : {dur:.1f}s  x{repeat} = 总计 {total_dur:.0f}s")
    print(f"  速率 : ~{bps:.0f} bps")
    print("=" * 52)
    print(f"\n  >>> 接收端请设置: --raw --timeout {int(total_dur + 5)} <<<\n")

    for i in range(repeat):
        print(f"  [{i+1}/{repeat}] 发送中 ...")
        sd.play(signal.astype(np.float32), SAMPLE_RATE)
        sd.wait()
        if i < repeat - 1:
            time.sleep(0.5)
    print("\n  发送完成。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WireTransmit — 通过 3.5mm 音频线传输文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python send.py document.pdf\n"
               "  python send.py image.png --raw --repeat 3\n"
               "  python send.py --list-devices\n"
               "  python send.py --test-tone",
    )
    parser.add_argument("file", nargs="?", metavar="文件", help="要发送的文件")
    parser.add_argument("--mode", choices=["2fsk","4fsk"], default="4fsk", help="调制方案 (默认: 4fsk)")
    parser.add_argument("--baud", type=int, default=300, choices=BAUD_RATES, help=f"波特率 (默认: 300)")
    parser.add_argument("--packet-size", type=int, default=512, help="每包载荷字节数 (双向模式)")
    parser.add_argument("--device", type=str, default=None, help="音频输出设备名称或索引")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备并退出")
    parser.add_argument("--test-tone", action="store_true", help="播放测试音验证输出")
    parser.add_argument("--raw", action="store_true", help="单向模式：不做握手，不等 ACK")
    parser.add_argument("--repeat", type=int, default=3, help="单向模式重复发送次数 (默认3)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices(); return
    if args.test_tone:
        from wiretransmit.modulator import FSKModulator
        mod = FSKModulator(args.mode, args.baud)
        tpl = mod.template()
        print(f"播放 {args.mode.upper()} 前导码+同步字测试音 @{args.baud} 波特 ({len(tpl)/SAMPLE_RATE:.1f}s)...")
        sd.play(tpl.astype(np.float32), SAMPLE_RATE); sd.wait()
        print("完成。"); return
    if args.file is None:
        parser.error("需要指定文件 (或使用 --list-devices / --test-tone)")
    if not os.path.isfile(args.file):
        print(f"错误: 文件不存在: {args.file}", file=sys.stderr); sys.exit(1)

    if args.device:
        dev_id = find_device(args.device, "output")
        if dev_id is None:
            try: dev_id = int(args.device)
            except ValueError:
                print(f"错误: 未找到设备: {args.device}", file=sys.stderr); list_devices(); sys.exit(1)
        set_device(dev_id, "output")
        print(f"使用输出设备 #{dev_id}")

    if args.raw:
        run_raw(args)
    else:
        from wiretransmit.transmitter import Transmitter
        ok = Transmitter(args.file, args.mode, args.baud, args.packet_size).run()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
