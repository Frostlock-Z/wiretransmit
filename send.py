#!/usr/bin/env python3
"""
send.py — 通过 3.5mm 音频输出口发送文件。

用法：
    python send.py <文件路径>                      # 默认: 4FSK @ 300 波特
    python send.py <文件路径> --mode 2fsk --baud 100
    python send.py --list-devices                 # 列出音频设备
    python send.py --test-tone                    # 播放测试音

请用 3.5mm 音频线将本机的耳机/扬声器插孔连接至接收端电脑的麦克风/线路输入插孔。
"""

import argparse
import sys
import os

# 确保 wiretransmit 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wiretransmit.constants import BAUD_RATES
from wiretransmit.utils import list_devices, find_device, set_device
from wiretransmit.transmitter import Transmitter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WireTransmit — 通过 3.5mm 音频线传输文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python send.py document.pdf
  python send.py image.png --baud 300 --mode 4fsk --packet-size 512
  python send.py --list-devices
  python send.py --test-tone
        """,
    )

    parser.add_argument(
        "file", nargs="?", metavar="文件",
        help="要发送的文件",
    )
    parser.add_argument(
        "--mode", choices=["2fsk", "4fsk"], default="4fsk",
        help="调制方案 (默认: 4fsk)",
    )
    parser.add_argument(
        "--baud", type=int, default=300, choices=BAUD_RATES,
        help=f"波特率（符号/秒），可选: {BAUD_RATES} (默认: 300)",
    )
    parser.add_argument(
        "--packet-size", type=int, default=512,
        help="每个 DATA 包的载荷字节数 (默认: 512)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="音频输出设备名称或索引",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="列出可用音频设备并退出",
    )
    parser.add_argument(
        "--test-tone", action="store_true",
        help="播放短测试音以验证输出设备",
    )

    args = parser.parse_args()

    # ---- 特殊命令 ----
    if args.list_devices:
        list_devices()
        return

    if args.test_tone:
        from wiretransmit.modulator import FSKModulator
        import numpy as np
        import sounddevice as sd
        from wiretransmit.constants import SAMPLE_RATE

        mod = FSKModulator(args.mode, args.baud)
        tpl = mod.template()
        print(f"播放 {args.mode.upper()} 前导码+同步字测试音 "
              f"@{args.baud} 波特 ({len(tpl)/SAMPLE_RATE:.1f}s)...")
        sd.play(tpl.astype(np.float32), SAMPLE_RATE)
        sd.wait()
        print("完成。")
        return

    # ---- 正常操作 ----
    if args.file is None:
        parser.error("需要指定文件 (或使用 --list-devices / --test-tone)")

    if not os.path.isfile(args.file):
        print(f"错误: 文件不存在: {args.file}", file=sys.stderr)
        sys.exit(1)

    # 设备选择
    if args.device:
        dev_id = find_device(args.device, "output")
        if dev_id is None:
            try:
                dev_id = int(args.device)
            except ValueError:
                print(f"错误: 未找到设备: {args.device}", file=sys.stderr)
                list_devices()
                sys.exit(1)
        set_device(dev_id, "output")
        print(f"使用输出设备 #{dev_id}")

    tx = Transmitter(
        filepath=args.file,
        mode=args.mode,
        baud=args.baud,
        packet_size=args.packet_size,
    )
    ok = tx.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
