"""
双向数据包协议：通过音频线实现可靠文件传输。

协议阶段：
  1. HELLO  （发送方 → 接收方）：文件元数据
  2. ACK    （接收方 → 发送方）：就绪确认
  3. DATA   （发送方 → 接收方）：载荷数据包（批量）
  4. NACK   （接收方 → 发送方）：缺失包位掩码
  5. 重传   （发送方 → 接收方）：仅发送缺失的包
  6. DONE   （接收方 → 发送方）：文件完整性验证通过
"""

import struct
import hashlib
from dataclasses import dataclass
from typing import List

from wiretransmit.constants import (
    PACKET_PAYLOAD_DEFAULT as PAYLOAD_SIZE,
    FLAG_DATA, FLAG_HELLO, FLAG_ACK, FLAG_NACK, FLAG_DONE, FLAG_LAST,
    MAX_FILE_SIZE,
)
from wiretransmit.framing import build_packet_bits


# =====================================================================
@dataclass
class HelloPacket:
    """传输开始前发送的元数据包。"""
    filename: str          # 原始文件名（UTF-8 编码）
    file_size: int         # 文件总大小（字节）
    total_packets: int     # DATA 包总数
    payload_size: int      # 每包载荷字节数
    sha256: bytes          # 完整文件的 SHA-256 摘要

    def encode(self) -> bytes:
        """编码为 HELLO 包体。

        布局：[文件名长度(1B)][文件名(UTF-8)][文件大小(4B)]
              [总包数(2B)][载荷大小(2B)][SHA-256(32B)]
        """
        fn_bytes = self.filename.encode("utf-8")
        if len(fn_bytes) > 255:
            fn_bytes = fn_bytes[:255]
        return struct.pack(
            f"<B{len(fn_bytes)}sIHH32s",
            len(fn_bytes), fn_bytes,
            self.file_size,
            self.total_packets,
            self.payload_size,
            self.sha256,
        )

    @staticmethod
    def decode(data: bytes) -> "HelloPacket":
        """从字节串解码 HELLO 包。"""
        fn_len = data[0]
        fn_bytes = data[1:1 + fn_len]
        offset = 1 + fn_len
        file_size, total_pkts, payload_sz = struct.unpack(
            "<IHH", data[offset:offset + 8]
        )
        offset += 8
        sha256 = data[offset:offset + 32]
        return HelloPacket(
            filename=fn_bytes.decode("utf-8"),
            file_size=file_size,
            total_packets=total_pkts,
            payload_size=payload_sz,
            sha256=sha256,
        )


# =====================================================================
@dataclass
class DataPacket:
    """单个载荷数据块。"""
    seq: int               # 序列号（0..N-1）
    payload: bytes         # 载荷数据
    flags: int = FLAG_DATA # 包标志

    def encode(self) -> bytes:
        return self.payload


# =====================================================================
@dataclass
class NackPacket:
    """请求重传指定编号的包。"""
    missing: List[int]     # 缺失包的序列号列表

    def encode(self) -> bytes:
        """编码为位掩码（大端序位序）。"""
        if not self.missing:
            return b'\x00\x00'
        max_seq = max(self.missing)
        num_bytes = (max_seq + 8) // 8
        mask = bytearray(num_bytes)
        for seq in self.missing:
            byte_idx = seq // 8
            bit_idx = 7 - (seq % 8)
            mask[byte_idx] |= (1 << bit_idx)
        return bytes(mask)

    @staticmethod
    def decode(data: bytes) -> "NackPacket":
        """从字节串解码 NACK 包。"""
        missing: List[int] = []
        for byte_idx, byte_val in enumerate(data):
            for bit_idx in range(8):
                if byte_val & (1 << (7 - bit_idx)):
                    missing.append(byte_idx * 8 + bit_idx)
        return NackPacket(missing=missing)


# =====================================================================
class FileAssembler:
    """从接收到的数据包拼接完整文件。"""

    def __init__(self, hello: HelloPacket) -> None:
        self.hello = hello
        self.buckets: dict[int, bytes] = {}              # seq → payload
        self.total = hello.total_packets

    def add(self, pkt: DataPacket) -> None:
        """存入一个数据包。"""
        self.buckets[pkt.seq] = pkt.payload

    @property
    def missing(self) -> List[int]:
        """返回尚未收到的包序列号列表。"""
        return [i for i in range(self.total) if i not in self.buckets]

    @property
    def is_complete(self) -> bool:
        """是否已收齐所有数据包。"""
        return len(self.buckets) == self.total

    def assemble(self) -> bytes:
        """按序列号顺序拼接所有已收到的数据包。"""
        result = bytearray()
        for i in range(self.total):
            if i in self.buckets:
                result.extend(self.buckets[i])
        return bytes(result)

    def verify(self) -> bool:
        """校验拼接后文件的 SHA-256 摘要。"""
        if not self.is_complete:
            return False
        data = self.assemble()
        return hashlib.sha256(data).digest() == self.hello.sha256


# =====================================================================
# 各类型包的比特序列构建函数
# =====================================================================
def build_hello_bits(hp: HelloPacket) -> List[int]:
    return build_packet_bits(0, FLAG_HELLO, hp.encode())


def build_ack_bits() -> List[int]:
    return build_packet_bits(0, FLAG_ACK, b'')


def build_data_bits(dp: DataPacket) -> List[int]:
    return build_packet_bits(dp.seq, dp.flags, dp.payload)


def build_nack_bits(np_: NackPacket) -> List[int]:
    return build_packet_bits(0, FLAG_NACK, np_.encode())


def build_done_bits() -> List[int]:
    return build_packet_bits(0, FLAG_DONE, b'')
