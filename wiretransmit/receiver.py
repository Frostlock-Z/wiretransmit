"""
接收端 — WireTransmit 的文件接收侧。

职责：
  1. 检测 HELLO，发送 ACK
  2. 收集 DATA 包
  3. 检测批次结束，发送 NACK（缺失包列表）
  4. 处理重传包
  5. RS 解码、SHA-256 校验、发送 DONE
"""

import os
import sys
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from wiretransmit.constants import (
    SAMPLE_RATE,
    FLAG_DATA, FLAG_HELLO, FLAG_ACK, FLAG_NACK, FLAG_DONE, FLAG_LAST,
    PREAMBLE_LEN,
)
from wiretransmit.modulator import FSKModulator
from wiretransmit.demodulator import FSKDemodulator
from wiretransmit.framing import parse_packet_bits
from wiretransmit.ecc import decode as rs_decode
from wiretransmit.protocol import (
    HelloPacket, DataPacket, FileAssembler, NackPacket,
    build_ack_bits, build_nack_bits, build_done_bits,
)
from wiretransmit.utils import print_progress, print_done


# ===================================================================
class Receiver:
    """管理文件接收会话。"""

    def __init__(self, output_path: str, mode: str = "4fsk",
                 baud: int = 300, timeout: float = 120.0) -> None:
        self.output_path = output_path
        self.mode = mode
        self.baud = baud
        self.global_timeout = timeout

        self.mod = FSKModulator(mode, baud)
        self.dem = FSKDemodulator(mode, baud)

        self._hello: Optional[HelloPacket] = None
        self._assembler: Optional[FileAssembler] = None

    # ------------------------------------------------------------------
    def run(self) -> bool:
        """执行完整接收协议。成功返回 True。"""
        print("=" * 52)
        print("  音频调制解调器 - 接收端")
        print("=" * 52)
        print(f"  输出 : {self.output_path}")
        print(f"  模式 : {self.mode.upper()} @ {self.baud} 波特")
        print(f"  监听 : 最长 {self.global_timeout:.0f} 秒")
        print("=" * 52)
        print("\n  正在监听 ...  (请启动发送端)")

        deadline = time.time() + self.global_timeout

        # ---- 阶段 1：等待 HELLO ----
        print("\n[1/4] 等待 HELLO ...")
        hello = self._wait_hello(deadline)
        if hello is None:
            print("  失败 — 在超时时间内未收到 HELLO。")
            return False

        self._hello = hello
        self._assembler = FileAssembler(hello)
        self._send_ack()

        print(f"\n  文件 : {hello.filename}")
        print(f"  大小 : {hello.file_size:,d} 字节  "
              f"({hello.file_size / 1024:.1f} KiB)")
        print(f"  数据包 : {hello.total_packets}")

        # ---- 阶段 2：收集 DATA ----
        print("\n[2/4] 接收数据 ...")
        self._collect_data(deadline)

        # ---- 阶段 3：NACK + 重传 ----
        print("\n[3/4] 校验 ...")
        nack_cycles = 0
        max_retries = 3
        while nack_cycles < max_retries and not self._assembler.is_complete:
            missing = self._assembler.missing
            print(f"  缺失: {len(missing)} 个包")
            self._send_nack(missing)
            time.sleep(0.3)
            self._collect_data(deadline, phase="重传")
            nack_cycles += 1

        # ---- 阶段 4：RS 解码、校验、DONE ----
        print("\n[4/4] 解码 ...")
        if not self._assembler.is_complete:
            miss = self._assembler.missing
            print(f"  警告: 仍有 {len(miss)} 个包缺失")
            if len(miss) > 0:
                print(f"  缺失序号: {miss[:10]}...")

        data = self._assembler.assemble()
        # 去除 RS 补零
        decoded = rs_decode(data)[:self._hello.file_size]

        print(f"  已解码 : {len(decoded):,d} 字节")
        if self._assembler.verify():
            print("  SHA-256 : 校验通过")
            self._send_done()
            with open(self.output_path, "wb") as f:
                f.write(decoded)
            print(f"  已保存 : {self.output_path}")
            print("\n  传输完成！")
            return True
        else:
            print("  SHA-256 : 不匹配 — 文件可能已损坏")
            self._send_done()
            # 仍然保存以便检查
            with open(self.output_path, "wb") as f:
                f.write(decoded)
            return False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _record(self, duration: float) -> np.ndarray:
        """录制指定时长的音频，返回展平并去直流分量的数组。"""
        raw = sd.rec(int(duration * SAMPLE_RATE),
                     samplerate=SAMPLE_RATE, channels=1,
                     dtype="float32")
        sd.wait()
        raw = raw.flatten()
        raw -= np.mean(raw)
        return raw

    def _demod(self, raw: np.ndarray, skip_preamble: bool = True):
        """将原始音频解调为（比特序列, 置信度列表），未找到同步时返回 None。"""
        coarse = self.dem.find_sync(raw)
        if coarse is None:
            return None

        fine = self.dem.fine_sync(raw, coarse)
        bits, confs, _ = self.dem.demodulate(raw, start=fine)

        if skip_preamble:
            bits = bits[PREAMBLE_LEN:]
            confs = confs[PREAMBLE_LEN:]

        return bits, confs

    def _wait_hello(self, deadline: float) -> Optional[HelloPacket]:
        """持续录音并搜索 HELLO 包，直到超时。"""
        chunk_dur = 5.0                    # 每 5 秒一块
        while time.time() < deadline:
            remaining = deadline - time.time()
            dur = min(chunk_dur, remaining)
            if dur < 1.0:
                break

            raw = self._record(dur)
            result = self._demod(raw)
            if result is None:
                continue

            bits, confs = result
            pkt, err = parse_packet_bits(bits)
            if pkt is None:
                continue
            if pkt["flags"] == FLAG_HELLO and pkt["crc_ok"]:
                return HelloPacket.decode(pkt["payload"])

        return None

    def _send_ack(self) -> None:
        """发送 ACK 应答信号。"""
        bits = build_ack_bits()
        signal = self.mod.modulate(bits)
        sd.play(signal.astype(np.float32), SAMPLE_RATE)
        sd.wait()

    def _send_nack(self, missing: list[int]) -> None:
        """发送 NACK（缺失包列表）信号。"""
        np_ = NackPacket(missing=missing)
        bits = build_nack_bits(np_)
        signal = self.mod.modulate(bits)
        sd.play(signal.astype(np.float32), SAMPLE_RATE)
        sd.wait()

    def _send_done(self) -> None:
        """发送 DONE 完成信号。"""
        bits = build_done_bits()
        signal = self.mod.modulate(bits)
        sd.play(signal.astype(np.float32), SAMPLE_RATE)
        sd.wait()

    def _collect_data(self, deadline: float,
                      phase: str = "数据") -> None:
        """分块录音并收集 DATA 包。"""
        idle_count = 0
        max_idle = 6                     # 约 18 秒无新包则停止

        while time.time() < deadline and idle_count < max_idle:
            dur = 3.0
            if time.time() + dur > deadline:
                dur = deadline - time.time()
            if dur < 1.0:
                break

            raw = self._record(dur)
            result = self._demod(raw)
            if result is None:
                idle_count += 1
                continue

            bits, confs = result
            pkt, err = parse_packet_bits(bits)
            if pkt is None:
                idle_count += 1
                continue

            if not pkt["crc_ok"]:
                idle_count += 1
                continue

            flags = pkt["flags"]

            if flags & FLAG_DATA:
                dp = DataPacket(seq=pkt["seq"], payload=pkt["payload"],
                                flags=flags)
                self._assembler.add(dp)
                idle_count = 0            # 收到有效包，重置空闲计数器

                total = self._hello.total_packets
                got = len(self._assembler.buckets)
                print_progress(got, total, f"  接收({phase})")

                if flags & FLAG_LAST or self._assembler.is_complete:
                    print_done(f"  批次接收完毕 ({got}/{total})")
                    return
            elif flags == FLAG_DONE:
                print_done("  收到 DONE。")
                return

        print_done(
            f"  收集结束 ({len(self._assembler.buckets)}"
            f"/{self._hello.total_packets})"
        )
