"""
Audio Modem -- File transfer over 3.5mm audio cable via FSK modulation.

Protocol:
  [Preamble 64b] [Sync 16b] [Length 16b] [Payload N*8b] [CRC-32 32b] [End 16b]

Physical layer:
  - Sample rate:  44100 Hz
  - f0 (bit 0):   2000 Hz
  - f1 (bit 1):   4000 Hz
  - Baud rate:    100 symbols/s
  - Modulation:   Phase-continuous FSK with raised-cosine envelope shaping
  - Demodulation: Non-coherent quadrature matched-filter detection
"""

import numpy as np
import sounddevice as sd
import zlib
import os
import sys
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
SAMPLE_RATE     = 44100        # Hz
FREQ_0          = 2000         # Hz for bit 0
FREQ_1          = 4000         # Hz for bit 1
BAUD_RATE       = 100          # symbols per second
AMPLITUDE       = 0.8          # 0.0 .. 1.0

SAMPLES_PER_BIT = SAMPLE_RATE // BAUD_RATE   # 441

PREAMBLE_LEN    = 64           # bits
SYNC_WORD       = 0x1A2B       # 16-bit frame sync
END_MARKER      = 0xBEEF       # 16-bit end-of-transmission

MAX_FILE_SIZE   = 4 * 1024 * 1024   # 4 MB safety limit

# Pre-computed bit patterns
PREAMBLE_BITS: List[int] = [1, 0] * (PREAMBLE_LEN // 2)
SYNC_BITS:      List[int] = [(SYNC_WORD  >> i) & 1 for i in range(15, -1, -1)]
END_BITS:       List[int] = [(END_MARKER >> i) & 1 for i in range(15, -1, -1)]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _bits_to_int(bits: List[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def _int_to_bits(val: int, n: int) -> List[int]:
    return [(val >> i) & 1 for i in range(n - 1, -1, -1)]


# ---------------------------------------------------------------------------
# FSK Modulator  (phase-continuous, with raised-cosine envelope)
# ---------------------------------------------------------------------------

class FSKModulator:
    """Phase-continuous FSK modulator."""

    def __init__(self) -> None:
        self._phase = 0.0
        t = np.linspace(0, 1, SAMPLES_PER_BIT)
        self._envelope = 0.5 - 0.5 * np.cos(2 * np.pi * t)   # raised cosine

    def _modulate_bit(self, bit: int) -> np.ndarray:
        freq = FREQ_1 if bit else FREQ_0
        dphi = 2 * np.pi * freq / SAMPLE_RATE
        phases = self._phase + dphi * np.arange(1, SAMPLES_PER_BIT + 1)
        self._phase = phases[-1] % (2 * np.pi)
        return (np.sin(phases) * AMPLITUDE * self._envelope).astype(np.float32)

    def modulate(self, bits: List[int]) -> np.ndarray:
        self._phase = 0.0
        return np.concatenate([self._modulate_bit(b) for b in bits])

    def template(self) -> np.ndarray:
        """Preamble + Sync template used by correlator-based sync detection."""
        return self.modulate(PREAMBLE_BITS + SYNC_BITS)


# ---------------------------------------------------------------------------
# FSK Demodulator  (non-coherent quadrature matched filter)
# ---------------------------------------------------------------------------

class FSKDemodulator:
    """Non-coherent FSK demodulator using quadrature matched filtering."""

    def __init__(self) -> None:
        t = np.arange(SAMPLES_PER_BIT) / SAMPLE_RATE
        win = np.hanning(SAMPLES_PER_BIT)
        self._s0 = np.sin(2 * np.pi * FREQ_0 * t) * win
        self._c0 = np.cos(2 * np.pi * FREQ_0 * t) * win
        self._s1 = np.sin(2 * np.pi * FREQ_1 * t) * win
        self._c1 = np.cos(2 * np.pi * FREQ_1 * t) * win

    def _energy(self, seg: np.ndarray) -> Tuple[float, float]:
        if len(seg) != SAMPLES_PER_BIT:
            return 0.0, 0.0
        e0 = np.dot(seg, self._s0) ** 2 + np.dot(seg, self._c0) ** 2
        e1 = np.dot(seg, self._s1) ** 2 + np.dot(seg, self._c1) ** 2
        return e0, e1

    def _confidence(self, e0: float, e1: float) -> float:
        total = e0 + e1
        return 0.5 if total < 1e-10 else e1 / total

    def demodulate(self, signal: np.ndarray,
                   start: int = 0) -> Tuple[List[int], List[float]]:
        """Demodulate *signal* into bits + per-bit confidence [0..1]."""
        bits: List[int] = []
        confs: List[float] = []

        n = (len(signal) - start) // SAMPLES_PER_BIT
        for i in range(n):
            beg = start + i * SAMPLES_PER_BIT
            end = beg + SAMPLES_PER_BIT
            e0, e1 = self._energy(signal[beg:end])
            c = self._confidence(e0, e1)
            bits.append(1 if c > 0.5 else 0)
            confs.append(c)

        return bits, confs

    def find_sync(self, signal: np.ndarray) -> Optional[int]:
        """Locate preamble via normalised cross-correlation.

        Returns the sample index where the preamble+sync template
        best matches, or ``None`` if no reliable match is found.
        """
        mod = FSKModulator()
        tpl = mod.template()
        tpl_energy = float(np.dot(tpl, tpl))

        # ---- normalised cross-correlation ----
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
        """Fine-tune bit-boundary alignment around coarse estimate."""
        best_off, best_match = 0, 0
        rng = SAMPLES_PER_BIT // 4
        for off in range(-rng, rng + 1):
            pos = coarse + off
            if pos < 0:
                continue
            bits, _ = self.demodulate(signal, start=pos)
            m = sum(1 for a, b in zip(bits[:PREAMBLE_LEN], PREAMBLE_BITS) if a == b)
            if m > best_match:
                best_match, best_off = m, off
        return coarse + best_off


# ---------------------------------------------------------------------------
# Frame builder / parser
# ---------------------------------------------------------------------------

def _build_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    bits += PREAMBLE_BITS
    bits += SYNC_BITS
    bits += _int_to_bits(len(data), 16)
    for b in data:
        bits += _int_to_bits(b, 8)
    bits += _int_to_bits(zlib.crc32(data) & 0xFFFFFFFF, 32)
    bits += END_BITS
    return bits


def _parse(bits: List[int]) -> Tuple[Optional[bytes], Optional[int], str]:
    """Return (payload, crc, error_msg).  *bits* must start at Sync word."""
    if len(bits) < 16:
        return None, None, "too few bits for sync"
    sync = _bits_to_int(bits[:16])
    if sync != SYNC_WORD:
        return None, None, f"sync mismatch 0x{sync:04X}"
    pos = 16

    if len(bits) < pos + 16:
        return None, None, "too few bits for length"
    length = _bits_to_int(bits[pos:pos + 16])
    pos += 16

    if length > MAX_FILE_SIZE:
        return None, None, f"length {length} > max"
    need = length * 8
    if len(bits) < pos + need:
        return None, None, f"need {need} payload bits, have {len(bits) - pos}"

    payload = bytearray()
    for _ in range(length):
        payload.append(_bits_to_int(bits[pos:pos + 8]))
        pos += 8
    payload = bytes(payload)

    if len(bits) < pos + 32:
        return payload, zlib.crc32(payload), "missing CRC field"
    rx_crc = _bits_to_int(bits[pos:pos + 32])
    pos += 32
    ex_crc = zlib.crc32(payload) & 0xFFFFFFFF

    err = "" if rx_crc == ex_crc else \
          f"CRC fail: expected 0x{ex_crc:08X}, got 0x{rx_crc:08X}"

    if len(bits) >= pos + 16:
        end = _bits_to_int(bits[pos:pos + 16])
        if end != END_MARKER:
            err += f" (end marker 0x{end:04X})"

    return payload, ex_crc, err


# ---------------------------------------------------------------------------
# Public high-level API
# ---------------------------------------------------------------------------

def transmit(filepath: str) -> np.ndarray:
    """Read *filepath*, modulate, play through default audio output."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    with open(filepath, "rb") as fh:
        data = fh.read()

    if len(data) > MAX_FILE_SIZE:
        raise ValueError(f"file too large ({len(data)} > {MAX_FILE_SIZE})")

    bits = _build_bits(data)
    signal = FSKModulator().modulate(bits)

    dur   = len(signal) / SAMPLE_RATE
    bps   = len(data) * 8 / dur

    print("=" * 52)
    print("  AUDIO MODEM - TRANSMITTER")
    print("=" * 52)
    print(f"  file  : {os.path.basename(filepath)}")
    print(f"  size  : {len(data):,d} bytes ({len(data)/1024:.1f} KiB)")
    print(f"  dur   : {dur:.1f} s")
    print(f"  baud  : {BAUD_RATE} sym/s  |  net rate : {bps:.0f} bps")
    print("=" * 52)
    print()
    print("  Playing  ...  (Ctrl+C to abort)")
    print("  Make sure the 3.5 mm cable is connected.")
    print()

    try:
        sd.play(signal, SAMPLE_RATE)
        sd.wait()
        print("  Done - transmission finished.")
    except KeyboardInterrupt:
        sd.stop()
        print("\n  Aborted.")
    return signal


def receive(output: str, timeout: float = 120.0) -> bool:
    """Record from default audio input, demodulate, save to *output*."""
    print("=" * 52)
    print("  AUDIO MODEM - RECEIVER")
    print("=" * 52)
    print(f"  output : {output}")
    print(f"  listen : up to {timeout:.0f} s")
    print("=" * 52)
    print()
    print("  Recording  ...  (start TRANSMITTER now, Ctrl+C to abort)")
    print()

    try:
        raw = sd.rec(int(timeout * SAMPLE_RATE),
                     samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("\n  Aborted.")
        return False

    raw = raw.flatten()
    raw -= np.mean(raw)                     # remove DC

    print("  Searching preamble  ...")
    dem = FSKDemodulator()
    coarse = dem.find_sync(raw)
    if coarse is None:
        print("  FAIL - no valid signal found.")
        print("  Check cable, input device, volume level.")
        return False

    fine = dem.fine_sync(raw, coarse)
    print(f"  Sync at sample {fine}  ({fine / SAMPLE_RATE:.2f} s)")

    print("  Demodulating  ...")
    bits, confs = dem.demodulate(raw, start=fine)

    # skip preamble bits (already consumed)
    frame = bits[PREAMBLE_LEN:]
    payload, _, err = _parse(frame)

    if payload is None:
        print(f"  FAIL - frame parse error: {err}")
        return False

    qual = np.mean([abs(c - 0.5) * 2 for c in confs[:len(frame)]])
    print(f"  quality : {qual * 100:.1f} %")
    print(f"  payload : {len(payload):,d} bytes")

    if err:
        print(f"  WARNING : {err}")
    else:
        print("  CRC     : OK")

    with open(output, "wb") as fh:
        fh.write(payload)
    print(f"  saved   : {output}")
    print("  Done.")
    return True
