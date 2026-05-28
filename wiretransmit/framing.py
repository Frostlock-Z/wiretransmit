"""
WireTransmit 物理层帧构建与解析。

协议帧格式：
  [前导码 64bit] [同步字 16bit] [长度 16bit] [载荷 N×8bit] [CRC-32 32bit] [结束标记 16bit]
支持数据包级别的构建/解析（用于双向协议）。
"""

import zlib
from typing import List, Tuple, Optional

from wiretransmit.constants import (
    PREAMBLE_BITS, SYNC_WORD, END_MARKER,
    MAX_FILE_SIZE,
)


# ---------------------------------------------------------------------------
# 比特级辅助函数
# ---------------------------------------------------------------------------
def int_to_bits(val: int, n: int) -> List[int]:
    """整数转 n 位比特列表，高位在前。"""
    return [(val >> i) & 1 for i in range(n - 1, -1, -1)]


def bits_to_int(bits: List[int]) -> int:
    """比特列表（高位在前）转整数。"""
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def bytes_to_bits(data: bytes) -> List[int]:
    """将字节串展平为比特列表，每字节高位在前。"""
    bits: List[int] = []
    for b in data:
        bits.extend(int_to_bits(b, 8))
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """将比特列表（长度需为 8 的倍数）打包为字节串。"""
    assert len(bits) % 8 == 0, "比特数必须是 8 的倍数"
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(bits_to_int(bits[i:i + 8]))
    return bytes(out)


# ---------------------------------------------------------------------------
# CRC-16-CCITT（用于数据包级别校验）
# ---------------------------------------------------------------------------
def crc16(data: bytes) -> int:
    """CRC-16-CCITT（多项式 0x1021）。"""
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# 帧构建
# ---------------------------------------------------------------------------
def build_frame_bits(data: bytes) -> List[int]:
    """构建完整的物理层帧比特序列。"""
    bits: List[int] = []
    bits += PREAMBLE_BITS                                       # 前导码
    bits += int_to_bits(SYNC_WORD, 16)                          # 同步字
    bits += int_to_bits(len(data), 16)                          # 载荷长度
    bits += bytes_to_bits(data)                                 # 载荷
    bits += int_to_bits(zlib.crc32(data) & 0xFFFFFFFF, 32)      # CRC-32
    bits += int_to_bits(END_MARKER, 16)                         # 结束标记
    return bits


# ---------------------------------------------------------------------------
# 帧解析
# ---------------------------------------------------------------------------
def parse_frame(bits: List[int]) -> Tuple[Optional[bytes], str]:
    """解析接收到的比特流，返回（载荷, 错误信息）。

    *bits* 必须从同步字位置开始（前导码已消耗）。
    解析失败时返回 (None, 错误原因)。
    """
    if len(bits) < 16:
        return None, "比特数不足，无法解析同步字"

    sync = bits_to_int(bits[:16])
    if sync != SYNC_WORD:
        return None, f"同步字不匹配 0x{sync:04X}"
    pos = 16

    if len(bits) < pos + 16:
        return None, "比特数不足，无法解析长度字段"
    length = bits_to_int(bits[pos:pos + 16])
    pos += 16

    if length == 0:
        return None, "载荷长度为零"
    if length > MAX_FILE_SIZE:
        return None, f"声明长度 {length} 超过最大限制 {MAX_FILE_SIZE}"

    need = length * 8
    if len(bits) < pos + need:
        return None, f"需要 {need} 载荷比特，实际只有 {len(bits) - pos}"

    payload_bits = bits[pos:pos + need]
    pos += need
    payload = bits_to_bytes(payload_bits)

    if len(bits) < pos + 32:
        return payload, "缺少 CRC 字段（帧不完整）"
    rx_crc = bits_to_int(bits[pos:pos + 32])
    pos += 32
    ex_crc = zlib.crc32(payload) & 0xFFFFFFFF

    err = ""
    if rx_crc != ex_crc:
        err = f"CRC 校验失败: 期望 0x{ex_crc:08X}, 收到 0x{rx_crc:08X}"

    if len(bits) >= pos + 16:
        end = bits_to_int(bits[pos:pos + 16])
        if end != END_MARKER:
            err += f"（结束标记异常 0x{end:04X}）"
    else:
        err += "（缺少结束标记）"

    return payload, err


# ---------------------------------------------------------------------------
# 数据包级辅助函数（供 protocol.py 使用）
# ---------------------------------------------------------------------------
def build_packet_bits(seq: int, flags: int, payload: bytes) -> List[int]:
    """构建数据包比特序列。

    包结构：[前导码][同步字][头部=4B][载荷][CRC-16][结束标记]
    头部: 序列号(2B) | 标志(1B) | 保留(1B)
    """
    header = bytes([(seq >> 8) & 0xFF, seq & 0xFF,
                     flags & 0xFF, 0x00])
    data = header + payload
    crc = crc16(data)

    bits: List[int] = []
    bits += PREAMBLE_BITS
    bits += int_to_bits(SYNC_WORD, 16)
    bits += bytes_to_bits(data)
    bits += int_to_bits(crc, 16)
    bits += int_to_bits(END_MARKER, 16)
    return bits


def parse_packet_bits(bits: List[int]) -> Tuple[Optional[dict], str]:
    """解析接收到的数据包比特，返回（包信息字典, 错误信息）。

    成功时字典包含: seq, flags, payload, crc_ok
    """
    if len(bits) < 16:
        return None, "比特数不足"

    sync = bits_to_int(bits[:16])
    if sync != SYNC_WORD:
        return None, f"同步字不匹配 0x{sync:04X}"
    pos = 16

    # 头部 = 4 字节 = 32 比特
    if len(bits) < pos + 32:
        return None, "比特数不足，无法解析包头部"
    header_bits = bits[pos:pos + 32]
    pos += 32
    header = bits_to_bytes(header_bits)
    seq   = (header[0] << 8) | header[1]
    flags = header[2]

    # 剩余比特 = 载荷 + CRC-16 + 结束标记
    remaining = len(bits) - pos - 16 - 16   # -CRC16 -END
    if remaining < 0:
        return None, "数据包过短"
    # 对齐到字节边界（解调后的比特可能有尾部噪声）
    remaining = (remaining // 8) * 8

    payload_bits = bits[pos:pos + remaining]
    pos += remaining
    payload = bits_to_bytes(payload_bits)

    rx_crc = bits_to_int(bits[pos:pos + 16])
    pos += 16
    ex_crc = crc16(header + payload)
    crc_ok = (rx_crc == ex_crc)

    return {
        "seq": seq,
        "flags": flags,
        "payload": payload,
        "crc_ok": crc_ok,
    }, ""
