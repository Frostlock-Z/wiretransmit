#!/usr/bin/env python3
"""
receive.py — 通过 3.5mm 音频输入口接收文件。

用法：
    python receive.py <输出文件路径>                  # 默认参数
    python receive.py <输出文件路径> --baud 300 --timeout 120
    python receive.py --list-devices                  # 列出音频设备

请用 3.5mm 音频线将发送端电脑的耳机/扬声器插孔连接至本机的麦克风/线路输入插孔。

务必先启动接收端，再启动发送端。
"""

import argparse
import sys
import os

# 确保 wiretransmit 包可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wiretransmit.constants import BAUD_RATES
from wiretransmit.utils import list_devices, find_device, set_device
from wiretransmit.receiver import Receiver


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WireTransmit — 通过 3.5mm 音频线接收文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python receive.py received.png
  python receive.py output.dat --baud 300 --mode 4fsk --timeout 180
  python receive.py --list-devices

务必先启动接收端，再启动发送端！
        """,
    )

    parser.add_argument(
        "output", nargs="?", metavar="输出文件",
        help="输出文件路径（接收到的数据将保存至此）",
    )
    parser.add_argument(
        "--mode", choices=["2fsk", "4fsk"], default="4fsk",
        help="调制方案（需与发送端一致）",
    )
    parser.add_argument(
        "--baud", type=int, default=300, choices=BAUD_RATES,
        help=f"波特率（需与发送端一致），可选: {BAUD_RATES} (默认: 300)",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="最长监听时间（秒），默认: 120",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="音频输入设备名称或索引",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="列出可用音频设备并退出",
    )

    args = parser.parse_args()

    # ---- 特殊命令 ----
    if args.list_devices:
        list_devices()
        return

    # ---- 正常操作 ----
    if args.output is None:
        parser.error("需要指定输出文件 (或使用 --list-devices)")

    # 设备选择
    if args.device:
        dev_id = find_device(args.device, "input")
        if dev_id is None:
            try:
                dev_id = int(args.device)
            except ValueError:
                print(f"错误: 未找到设备: {args.device}", file=sys.stderr)
                list_devices()
                sys.exit(1)
        set_device(dev_id, "input")
        print(f"使用输入设备 #{dev_id}")

    rx = Receiver(
        output_path=args.output,
        mode=args.mode,
        baud=args.baud,
        timeout=args.timeout,
    )
    ok = rx.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
