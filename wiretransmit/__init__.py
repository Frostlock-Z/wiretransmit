"""
WireTransmit — 通过 3.5mm 音频线传输文件。

支持 2-FSK / 4-FSK 调制、Reed-Solomon 前向纠错和双向数据包协议，
实现可靠的文件传输。

Author:  Frostlock Zhou
License: MIT
"""

from wiretransmit.constants import (
    SAMPLE_RATE, BAUD_RATE_DEFAULT, BAUD_RATES,
    get_frequencies, get_samples_per_symbol,
)

__version__ = "0.2.0"
__author__  = "Frostlock Zhou"
__all__ = [
    "SAMPLE_RATE",
    "BAUD_RATE_DEFAULT",
    "BAUD_RATES",
    "get_frequencies",
    "get_samples_per_symbol",
]
