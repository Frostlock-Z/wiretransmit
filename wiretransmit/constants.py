"""
WireTransmit 物理层与协议层常量集中管理。

所有可调参数均集中于此，便于在不同调制方案和波特率之间切换实验。
"""

from typing import Dict, List, Tuple

# ============================================================================
# 物理层参数（采样率、频率映射、波特率）
# ============================================================================

SAMPLE_RATE:       int = 44100           # Hz（CD 音质）
AMPLITUDE:       float = 0.8             # 输出振幅 0.0..1.0

# ---- 波特率 ----
BAUD_RATE_DEFAULT: int = 300             # 默认符号速率（符号/秒）
BAUD_RATES: List[int] = [100, 200, 300, 400, 600]

# ---- 2-FSK 频率表 ----
FREQ_2FSK: Tuple[int, int] = (2000, 4000)

# ---- 4-FSK 频率表（Hz） ----
# S0=00, S1=01, S2=10, S3=11
FREQ_4FSK: Tuple[int, int, int, int] = (2000, 2667, 3333, 4000)


def get_frequencies(mode: str) -> Tuple[int, ...]:
    """返回指定模式的频率元组（'2fsk' | '4fsk'）。"""
    if mode == "2fsk":
        return FREQ_2FSK
    if mode == "4fsk":
        return FREQ_4FSK
    raise ValueError(f"未知的调制模式: {mode}")


def get_bits_per_symbol(mode: str) -> int:
    """每个符号携带的比特数（= log2(频率数)）。"""
    return len(get_frequencies(mode)).bit_length() - 1


def get_samples_per_symbol(baud: int) -> int:
    """给定波特率下每个符号的采样点数。"""
    return SAMPLE_RATE // baud


# ============================================================================
# 协议常量（帧结构）
# ============================================================================

PREAMBLE_LEN: int = 64                # 前导码比特数
SYNC_WORD:    int = 0x1A2B            # 16 位帧同步字
END_MARKER:   int = 0xBEEF            # 16 位传输结束标记

MAX_FILE_SIZE: int = 4 * 1024 * 1024  # 最大文件 4 MB

# 预计算的前导码比特模式 [1,0,1,0,...]
PREAMBLE_BITS: List[int] = [1, 0] * (PREAMBLE_LEN // 2)

# ============================================================================
# 数据包协议（双向可靠传输）
# ============================================================================

PACKET_PAYLOAD_DEFAULT: int = 512     # 默认每包载荷字节数
PACKET_HEADER_SIZE:     int = 4       # [序列号(2B) | 标志(1B) | 保留(1B)]
PACKET_CRC_SIZE:        int = 2       # CRC-16

# 包类型标志位
FLAG_DATA:  int = 0x01   # 数据包
FLAG_HELLO: int = 0x02   # 握手请求
FLAG_ACK:   int = 0x04   # 确认应答
FLAG_NACK:  int = 0x08   # 否定应答（请求重传）
FLAG_DONE:  int = 0x10   # 传输完成
FLAG_LAST:  int = 0x80   # 批次的最后一个数据包

# CRC-16-CCITT 多项式
CRC16_POLY: int = 0x1021

# ACK/NACK 信号模式
ACK_SIGNAL:  int = 0xAAAA             # ACK 16 位模式
NACK_PREFIX: int = 0x5555             # NACK 16 位前缀

# 超时设置（秒）
HELLO_TIMEOUT:  float = 10.0          # 发送 HELLO 后等待 ACK 的超时
PACKET_TIMEOUT: float =  5.0          # 数据包之间的最大间隔
NACK_TIMEOUT:   float = 10.0          # 发送最后包后等待 NACK 的超时
DONE_TIMEOUT:   float = 10.0          # 等待 DONE 确认的超时

# ============================================================================
# Reed-Solomon 前向纠错参数
# ============================================================================

RS_N: int = 255    # 码字长度
RS_K: int = 239    # 数据长度
RS_T: int = 8      # 纠错能力 = (RS_N - RS_K) // 2，即可纠正 8 字节错误
