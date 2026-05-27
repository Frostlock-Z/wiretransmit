"""
FSK 解调器：基于非相干正交匹配滤波器检测。

支持 2-FSK 和 4-FSK 两种模式。
同步通过归一化互相关匹配存储的前导码+同步模板完成。
"""

import numpy as np
from typing import List, Tuple, Optional

from wiretransmit.constants import (
    SAMPLE_RATE,
    FREQ_2FSK, FREQ_4FSK,
    get_samples_per_symbol, get_frequencies, get_bits_per_symbol,
    PREAMBLE_BITS, PREAMBLE_LEN,
)
from wiretransmit.modulator import FSKModulator


class FSKDemodulator:
    """非相干 FSK / 4-FSK 解调器。"""

    def __init__(self, mode: str = "4fsk", baud: int = 300) -> None:
        self.mode = mode
        self.baud = baud
        self.freqs = get_frequencies(mode)           # 频率表
        self.bps = get_bits_per_symbol(mode)          # 每符号比特数
        self.sps = get_samples_per_symbol(baud)       # 每符号采样点数

        # ---- 为每个频率预构建正交匹配滤波器 ----
        t = np.arange(self.sps) / SAMPLE_RATE
        win = np.hanning(self.sps)                    # 汉宁窗
        self._s_filters: List[Tuple[np.ndarray, np.ndarray]] = []
        for freq in self.freqs:
            s = np.sin(2 * np.pi * freq * t) * win    # 正弦分量
            c = np.cos(2 * np.pi * freq * t) * win    # 余弦分量
            self._s_filters.append((s, c))

        # 保留一个调制器实例用于生成同步模板
        self._mod = FSKModulator(mode, baud)

    # ------------------------------------------------------------------
    def _energies(self, seg: np.ndarray) -> List[float]:
        """计算每个频率通道的正交能量。"""
        if len(seg) != self.sps:
            return [0.0] * len(self.freqs)
        energies: List[float] = []
        for s_filt, c_filt in self._s_filters:
            e = np.dot(seg, s_filt) ** 2 + np.dot(seg, c_filt) ** 2
            energies.append(float(e))
        return energies

    # ------------------------------------------------------------------
    def _symbol_from_energies(self, energies: List[float]) -> int:
        """选择能量最高的通道对应的符号索引（0..N-1）。"""
        return int(np.argmax(energies))

    def _confidence(self, energies: List[float], symbol: int) -> float:
        """每个符号的置信度 [0..1]。"""
        total = sum(energies)
        if total < 1e-10:
            return 0.5
        return energies[symbol] / total

    # ------------------------------------------------------------------
    def demodulate(self, signal: np.ndarray,
                   start: int = 0) -> Tuple[List[int], List[float], List[int]]:
        """将信号解调为（比特序列, 置信度列表, 符号列表）。

        返回值：
            bits:         解调后的原始比特
            confidences:  每比特置信度 [0..1]
            symbols:      每符号判决结果（0..N-1）
        """
        symbols: List[int] = []
        confs:   List[float] = []

        n = (len(signal) - start) // self.sps
        for i in range(n):
            beg = start + i * self.sps
            end = beg + self.sps
            energies = self._energies(signal[beg:end])
            sym = self._symbol_from_energies(energies)
            symbols.append(sym)
            confs.append(self._confidence(energies, sym))

        # 符号 → 比特转换
        bits:      List[int] = []
        bit_confs: List[float] = []
        for sym, c in zip(symbols, confs):
            for shift in range(self.bps - 1, -1, -1):
                bits.append((sym >> shift) & 1)
                bit_confs.append(c)

        return bits, bit_confs, symbols

    # ------------------------------------------------------------------
    # 同步检测
    # ------------------------------------------------------------------
    def find_sync(self, signal: np.ndarray) -> Optional[int]:
        """通过归一化互相关定位前导码。

        返回最佳匹配的样本索引，找不到时返回 ``None``。
        """
        tpl = self._mod.template()
        tpl_energy = float(np.dot(tpl, tpl))

        corr = np.correlate(signal, tpl, mode="valid")
        win_energy = np.convolve(signal ** 2, np.ones(len(tpl)),
                                 mode="valid")
        safe = win_energy > 1e-10
        ncorr = np.zeros_like(corr)
        ncorr[safe] = corr[safe] / np.sqrt(win_energy[safe] * tpl_energy)
        ab = np.abs(ncorr)

        peak = int(np.argmax(ab))
        bg_mean = float(np.mean(ab))
        bg_std  = float(np.std(ab))
        thresh = max(bg_mean + 5 * bg_std, 0.25)

        if ab[peak] < thresh:
            return None
        return peak

    def fine_sync(self, signal: np.ndarray, coarse: int) -> int:
        """在粗同步点附近微调符号边界对齐。

        在 ±sps/4 样本范围内扫描，返回使前导码匹配数最大化的偏移量。
        """
        best_off, best_match = 0, 0
        rng = self.sps // 4
        for off in range(-rng, rng + 1):
            pos = coarse + off
            if pos < 0:
                continue
            bits, _, _ = self.demodulate(signal, start=pos)
            m = sum(1 for a, b in zip(bits[:PREAMBLE_LEN], PREAMBLE_BITS)
                    if a == b)
            if m > best_match:
                best_match, best_off = m, off
        return coarse + best_off
