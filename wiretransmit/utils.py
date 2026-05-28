"""
辅助工具：音频设备枚举、信号质量评估、SHA-256 校验、格式化进度显示。
"""

import hashlib
import sys
from typing import Optional

import numpy as np
import sounddevice as sd

from wiretransmit.constants import SAMPLE_RATE


# ===================================================================
def sha256_file(path: str) -> bytes:
    """计算文件的 SHA-256 摘要。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.digest()


# ===================================================================
# 音频设备辅助函数
# ===================================================================
def list_devices() -> None:
    """列出所有可用的音频设备并输出到 stdout。"""
    devices = sd.query_devices()
    print(f"{'#':>3}  {'设备名称':<40} {'输入':>6} {'输出':>6}  {'默认':>8}")
    print("-" * 75)
    for i, d in enumerate(devices):
        ch_in = d.get("max_input_channels", 0)
        ch_out = d.get("max_output_channels", 0)
        is_default = ""
        if i == sd.default.device[0]:
            is_default = " 输入默认"
        elif i == sd.default.device[1]:
            is_default = " 输出默认"
        print(f"{i:>3}  {d['name']:<40} {ch_in:>6} {ch_out:>6}  {is_default}")


def find_device(name: str, kind: str = "output") -> Optional[int]:
    """通过（部分）名称查找设备。*kind*: 'input' | 'output'。"""
    devices = sd.query_devices()
    name_lower = name.lower()
    candidates = []
    for i, d in enumerate(devices):
        if name_lower not in d["name"].lower():
            continue
        if kind == "input" and d.get("max_input_channels", 0) <= 0:
            continue
        if kind == "output" and d.get("max_output_channels", 0) <= 0:
            continue
        # 验证设备真的可用
        if not _probe_device(i, kind):
            continue
        candidates.append(i)
    return candidates[0] if candidates else None


def set_device(device_id: Optional[int], kind: str) -> None:
    """设置默认输入或输出设备。"""
    if device_id is not None:
        if kind == "input":
            sd.default.device[0] = device_id
        else:
            sd.default.device[1] = device_id


def _probe_device(device_id: int, kind: str) -> bool:
    """快速验证设备是否真的可用（部分驱动上报可用但实际打不开）。"""
    try:
        if kind == "input":
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1,
                dtype="float32", device=device_id,
            )
        else:
            stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=1,
                dtype="float32", device=device_id,
            )
        stream.close()
        return True
    except Exception:
        return False


# ===================================================================
def signal_quality(bits: list, confs: list) -> float:
    """解调后比特的平均置信度评分（0..100%）。"""
    if not confs:
        return 0.0
    return float(np.mean([abs(c - 0.5) * 2 for c in confs]) * 100)


# ===================================================================
# 进度显示
# ===================================================================
def progress_bar(current: int, total: int, label: str = "",
                 width: int = 30) -> str:
    """返回一行进度条字符串。"""
    pct = min(current / total, 1.0) if total > 0 else 0.0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r{label} [{bar}] {pct * 100:5.1f}%  ({current}/{total})"


def print_progress(current: int, total: int, label: str = "",
                   width: int = 30) -> None:
    """向 stderr 输出一行进度条（回车刷新）。"""
    sys.stderr.write(progress_bar(current, total, label, width))
    sys.stderr.flush()


def print_done(label: str = "完成") -> None:
    """换行输出完成信息。"""
    sys.stderr.write(f"\n{label}\n")
    sys.stderr.flush()
