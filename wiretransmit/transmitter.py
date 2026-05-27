"""
发送端 — WireTransmit 的文件发送侧。

职责：
  1. HELLO 握手
  2. DATA 批量传输
  3. 处理 NACK + 选择性重传
  4. 等待 DONE 确认
"""

import os
import sys
import time
import hashlib
from typing import Optional

import numpy as np
import sounddevice as sd

from wiretransmit.constants import (
    SAMPLE_RATE, MAX_FILE_SIZE,
    HELLO_TIMEOUT, NACK_TIMEOUT, DONE_TIMEOUT,
    FLAG_DATA, FLAG_HELLO, FLAG_ACK, FLAG_NACK, FLAG_DONE, FLAG_LAST,
    PREAMBLE_LEN,
)
from wiretransmit.modulator import FSKModulator
from wiretransmit.demodulator import FSKDemodulator
from wiretransmit.framing import (
    build_packet_bits, parse_packet_bits,
)
from wiretransmit.ecc import encode as rs_encode
from wiretransmit.protocol import (
    HelloPacket, DataPacket, NackPacket,
    build_hello_bits, build_data_bits,
)
from wiretransmit.utils import (
    sha256_file, print_progress, print_done,
)


# ===================================================================
class Transmitter:
    """管理文件发送会话。"""

    def __init__(self, filepath: str, mode: str = "4fsk",
                 baud: int = 300, packet_size: int = 512) -> None:
        self.filepath = filepath
        self.mode = mode
        self.baud = baud
        self.packet_size = packet_size

        self.mod = FSKModulator(mode, baud)
        self.dem = FSKDemodulator(mode, baud)

        # 内部状态
        self._file_data: bytes = b''
        self._rs_data: bytes = b''
        self._packets: list[bytes] = []
        self._hello: Optional[HelloPacket] = None

    # ------------------------------------------------------------------
    def prepare(self) -> HelloPacket:
        """读取文件、RS 编码、拆分为数据包、构建 HELLO。"""
        filename = os.path.basename(self.filepath)

        # 读取文件
        with open(self.filepath, "rb") as f:
            self._file_data = f.read()

        if len(self._file_data) > MAX_FILE_SIZE:
            raise ValueError(
                f"文件过大: {len(self._file_data)} > {MAX_FILE_SIZE}"
            )

        file_sha = hashlib.sha256(self._file_data).digest()

        # RS 编码（不足 k 字节的块补零对齐）
        self._rs_data = rs_encode(self._file_data)

        # 拆分为数据包
        self._packets = []
        total_len = len(self._rs_data)
        total_pkts = (total_len + self.packet_size - 1) // self.packet_size
        for i in range(total_pkts):
            start = i * self.packet_size
            end = min(start + self.packet_size, total_len)
            self._packets.append(self._rs_data[start:end])

        self._hello = HelloPacket(
            filename=filename,
            file_size=len(self._file_data),
            total_packets=total_pkts,
            payload_size=self.packet_size,
            sha256=file_sha,
        )
        return self._hello

    # ------------------------------------------------------------------
    def send_hello(self) -> bool:
        """发送 HELLO 并等待 ACK。成功返回 True。"""
        if self._hello is None:
            self.prepare()

        # 发送 HELLO
        bits = build_hello_bits(self._hello)
        signal = self.mod.modulate(bits)
        self._play(signal)

        # 监听 ACK
        ack = self._listen_for(FLAG_ACK, HELLO_TIMEOUT)
        if ack is not None:
            print("  收到 ACK — 接收端已就绪。")
        return ack is not None

    # ------------------------------------------------------------------
    def send_data_batch(self) -> None:
        """按顺序发送所有数据包。"""
        total = len(self._packets)
        print(f"\n  发送 {total} 个数据包 ...")

        for i, payload in enumerate(self._packets):
            flags = FLAG_DATA | (FLAG_LAST if i == total - 1 else 0)
            dp = DataPacket(seq=i, payload=payload, flags=flags)
            bits = build_data_bits(dp)
            signal = self.mod.modulate(bits)
            self._play(signal)
            print_progress(i + 1, total, "  发送")

        print_done("  所有数据包已发送。")
        # 短暂暂停，让接收端处理最后一个包
        time.sleep(0.5)

    # ------------------------------------------------------------------
    def handle_nack(self) -> tuple[int, int]:
        """监听 NACK，重传缺失的包。

        返回 (缺失数量, 重传数量)。
        """
        nack_pkt = self._listen_for(FLAG_NACK, NACK_TIMEOUT)
        if nack_pkt is None:
            print("  未收到 NACK — 假设所有包均已成功接收。")
            return (0, 0)

        np_ = NackPacket.decode(nack_pkt["payload"])
        missing = np_.missing
        if not missing:
            print("  NACK 未列出缺失包 — 传输正常。")
            return (0, 0)

        print(f"  NACK: {len(missing)} 个包缺失。")
        retx = 0
        for seq in missing:
            if seq >= len(self._packets):
                continue
            dp = DataPacket(seq=seq, payload=self._packets[seq])
            bits = build_data_bits(dp)
            signal = self.mod.modulate(bits)
            self._play(signal)
            retx += 1
            print_progress(retx, len(missing), "  重传")
        print_done(f"  已重传 {retx} 个包。")
        return (len(missing), retx)

    # ------------------------------------------------------------------
    def wait_done(self) -> bool:
        """等待 DONE 确认。收到返回 True。"""
        done = self._listen_for(FLAG_DONE, DONE_TIMEOUT)
        if done is not None:
            print("  收到 DONE — 传输完成。")
            return True
        print("  警告: 未收到 DONE，但传输可能已完成。")
        return False

    # ------------------------------------------------------------------
    def run(self) -> bool:
        """执行完整传输协议。成功返回 True。"""
        self.prepare()

        print("=" * 52)
        print("  音频调制解调器 - 发送端")
        print("=" * 52)
        print(f"  文件 : {self._hello.filename}")
        print(f"  大小 : {self._hello.file_size:,d} 字节  "
              f"({self._hello.file_size / 1024:.1f} KiB)")
        print(f"  模式 : {self.mode.upper()} @ {self.baud} 波特")
        print(f"  数据包 : {self._hello.total_packets}")
        print("=" * 52)

        # 阶段 1：握手
        print("\n[1/4] 握手 ...")
        if not self.send_hello():
            print("  失败 — 未收到接收端 ACK。")
            return False

        # 阶段 2：数据
        print("\n[2/4] 数据传输 ...")
        self.send_data_batch()

        # 阶段 3：NACK + 重传
        print("\n[3/4] 校验 ...")
        missing, retx = self.handle_nack()

        # 阶段 4：完成
        print("\n[4/4] 完成确认 ...")
        ok = self.wait_done()
        return ok

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _play(self, signal: np.ndarray) -> None:
        """播放音频信号。"""
        sd.play(signal.astype(np.float32), SAMPLE_RATE)
        sd.wait()

    def _listen_for(self, expected_flag: int,
                    timeout: float) -> Optional[dict]:
        """录音并搜索指定标志类型的包。

        返回解析后的包字典，未找到时返回 None。
        """
        dur = max(timeout, self.mod.signal_duration_bytes(200))
        try:
            raw = sd.rec(int(dur * SAMPLE_RATE),
                         samplerate=SAMPLE_RATE, channels=1,
                         dtype="float32")
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            return None

        raw = raw.flatten()
        raw -= np.mean(raw)                # 去除直流分量

        coarse = self.dem.find_sync(raw)
        if coarse is None:
            return None

        fine = self.dem.fine_sync(raw, coarse)
        bits, _, _ = self.dem.demodulate(raw, start=fine)

        # 跳过前导码比特
        frame_bits = bits[PREAMBLE_LEN:]

        pkt, err = parse_packet_bits(frame_bits)
        if pkt is None:
            return None

        if not pkt["crc_ok"]:
            return None

        if pkt["flags"] == expected_flag:
            return pkt
        return None
