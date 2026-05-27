"""
FSK 调制器：相位连续合成 + 升余弦包络整形。

支持 2-FSK（1 比特/符号）和 4-FSK（2 比特/符号）两种调制方式。
"""

import numpy as np
from typing import List, Tuple

from wiretransmit.constants import (
    SAMPLE_RATE, AMPLITUDE,
    FREQ_2FSK, FREQ_4FSK,
    get_samples_per_symbol, get_frequencies, get_bits_per_symbol,
    PREAMBLE_BITS, SYNC_WORD,
)


class FSKModulator:
    """相位连续的 FSK / 4-FSK 调制器。"""

    def __init__(self, mode: str = "4fsk", baud: int = 300) -> None:
        self.mode = mode
        self.baud = baud
        self.freqs = get_frequencies(mode)          # 频率表
        self.bps = get_bits_per_symbol(mode)         # 每符号比特数（1 或 2）
        self.sps = get_samples_per_symbol(baud)      # 每符号采样点数
        self._phase = 0.0

        # 升余弦包络窗口（平滑 0→1→0）
        t = np.linspace(0, 1, self.sps)
        self._envelope = 0.5 - 0.5 * np.cos(2 * np.pi * t)

    def _frequency_for_symbol(self, symbol: int) -> int:
        """将符号索引（0..N-1）映射为频率值。"""
        return self.freqs[symbol]

    def _bits_to_symbols(self, bits: List[int]) -> List[int]:
        """将比特序列分组为符号（2FSK: 1 比特→1 符号, 4FSK: 2 比特→1 符号）。"""
        symbols: List[int] = []
        step = self.bps
        for i in range(0, len(bits), step):
            chunk = bits[i:i + step]
            # 不足时补零
            while len(chunk) < step:
                chunk.append(0)
            # 二进制块 → 整数符号
            sym = 0
            for b in chunk:
                sym = (sym << 1) | b
            symbols.append(sym)
        return symbols

    def _modulate_symbol(self, freq: int) -> np.ndarray:
        """针对给定频率生成一个符号的音频采样。"""
        dphi = 2 * np.pi * freq / SAMPLE_RATE
        phases = self._phase + dphi * np.arange(1, self.sps + 1)
        self._phase = phases[-1] % (2 * np.pi)
        return (np.sin(phases) * AMPLITUDE * self._envelope).astype(np.float32)

    def modulate(self, bits: List[int]) -> np.ndarray:
        """将比特列表调制成音频信号。

        在 4-FSK 模式下，每连续两个比特映射为一个符号。
        """
        self._phase = 0.0
        symbols = self._bits_to_symbols(bits)
        segments = [self._modulate_symbol(self._frequency_for_symbol(s))
                     for s in symbols]
        return np.concatenate(segments)

    def template(self) -> np.ndarray:
        """生成前导码+同步字模板，用于基于互相关的同步检测。"""
        sync_bits = [(SYNC_WORD >> i) & 1 for i in range(15, -1, -1)]
        return self.modulate(PREAMBLE_BITS + sync_bits)

    # ------------------------------------------------------------------
    # 便捷方法：直接调制原始字节
    # ------------------------------------------------------------------
    def modulate_bytes(self, data: bytes) -> np.ndarray:
        """将原始字节直接调制成音频信号。"""
        bits: List[int] = []
        for b in data:
            bits.extend((b >> i) & 1 for i in range(7, -1, -1))
        return self.modulate(bits)

    def signal_duration(self, n_bits: int) -> float:
        """承载 n_bits 个比特的信号时长（秒）。"""
        n_symbols = (n_bits + self.bps - 1) // self.bps
        return n_symbols * self.sps / SAMPLE_RATE

    def signal_duration_bytes(self, n_bytes: int) -> float:
        """承载 n_bytes 个字节的信号时长（秒）。"""
        return self.signal_duration(n_bytes * 8)
