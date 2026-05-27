"""
Reed-Solomon RS(255, 239) 前向纠错编解码器，基于 GF(2^8) 有限域。

纯 Python 实现——无外部依赖。

编码惯例（校验前置）：
  码字多项式：c(x) = c_0 + c_1·x + c_2·x^2 + ... + c_{n-1}·x^{n-1}
  码字布局：    [校验字节 = nroots 字节] || [消息字节 = k 字节]
  索引 j 的字节对应 x^j 的系数。

功能：
  - 编码：在每个 239 字节消息块前添加 16 字节校验
  - 解码：每 255 字节码字最多纠正 8 字节错误
"""

from typing import List, Optional


# =====================================================================
#  GF(2^8) 有限域运算  （本原元 α = 2，本原多项式 0x11D）
# =====================================================================
GF_SIZE: int = 256

# exp[i] = α^i,  log[a] = i  ⇔  α^i = a
_exp: List[int] = [1] * (GF_SIZE * 2)
_log: List[int] = [0] * GF_SIZE


def _init_gf() -> None:
    """初始化指数/对数查找表。"""
    x = 1
    for i in range(GF_SIZE - 1):
        _exp[i] = x
        _log[x] = i
        x <<= 1
        if x & GF_SIZE:
            x ^= 0x11D
    _exp[GF_SIZE - 1] = 1
    # 复制前半部分便于乘法中的索引回绕
    for i in range(GF_SIZE - 1, len(_exp)):
        _exp[i] = _exp[i - (GF_SIZE - 1)]


_init_gf()


def gf_add(a: int, b: int) -> int:
    """GF(2^8) 加法 = 异或。"""
    return a ^ b


def gf_mul(a: int, b: int) -> int:
    """GF(2^8) 乘法（查表法）。"""
    if a == 0 or b == 0:
        return 0
    return _exp[_log[a] + _log[b]]


def gf_pow(a: int, n: int) -> int:
    """a^n（n 可为负，使用模逆）。"""
    if a == 0:
        return 0 if n > 0 else 1
    exp = (_log[a] * n) % (GF_SIZE - 1)
    return _exp[exp]


def gf_inv(a: int) -> int:
    """求乘法逆元。"""
    if a == 0:
        raise ZeroDivisionError("0 没有乘法逆元")
    return _exp[(GF_SIZE - 1) - _log[a]]


# =====================================================================
#  多项式辅助函数（系数：索引 → 次数）
# =====================================================================
def poly_eval(p: List[int], x: int) -> int:
    """Horner 法求多项式在 x 处的值。p[0] = 常数项。"""
    y = 0
    for coef in reversed(p):
        y = gf_mul(y, x) ^ coef
    return y


def poly_mul(p: List[int], q: List[int]) -> List[int]:
    """多项式乘法。"""
    r = [0] * (len(p) + len(q) - 1)
    for i, pc in enumerate(p):
        for j, qc in enumerate(q):
            r[i + j] ^= gf_mul(pc, qc)
    return r


def poly_scale(p: List[int], s: int) -> List[int]:
    """多项式缩放（乘以标量）。"""
    return [gf_mul(c, s) for c in p]


def poly_strip(p: List[int]) -> List[int]:
    """移除多项式末尾的零系数。"""
    while p and p[-1] == 0:
        p.pop()
    return p


# =====================================================================
#  生成多项式
#  g(x) = ∏_{i=0}^{nroots-1} (x - α^i)  =  ∏ (x + α^i)
# =====================================================================
def rs_generator_poly(nroots: int) -> List[int]:
    """构建 RS 生成多项式。"""
    g = [1]
    for i in range(nroots):
        # g ← g · (x + α^i) → 系数顺序 [α^i, 1]（低→高：常数项 + 1·x）
        g = poly_mul(g, [gf_pow(2, i), 1])
    return g


# =====================================================================
#  RS 编解码器
# =====================================================================
class RSCodec:
    """Reed-Solomon (n, k) 系统码编解码器。

    码字 = 校验字节（nroots 字节）+ 消息字节（k 字节）。
    """

    def __init__(self, n: int = 255, k: int = 239) -> None:
        self.n = n               # 码字长度
        self.k = k               # 消息长度
        self.nroots = n - k      # 校验字节数（通常为 16）
        self.generator = rs_generator_poly(self.nroots)

    # ------------------------------------------------------------------
    def encode_block(self, msg: bytes) -> bytes:
        """编码一个 k 字节消息块 → n 字节码字（校验前缀格式）。

        原理：c(x) = m(x)·x^{nroots} - r(x)，其中 r = m·x^{nroots} mod g
        """
        assert len(msg) == self.k

        # 构建 m(x)·x^{nroots} → n 个系数
        dividend = [0] * self.nroots + list(msg)

        # 被 g(x) 做多项式除法
        for i in range(self.n - 1, self.nroots - 1, -1):
            coef = dividend[i]
            if coef != 0:
                # 减去 coef · g(x) · x^{i - nroots}
                for j in range(len(self.generator)):
                    idx = i - self.nroots + j
                    dividend[idx] ^= gf_mul(self.generator[j], coef)

        # 余数在 dividend[0..nroots-1]
        parity = bytes(dividend[:self.nroots])
        return parity + msg

    def encode(self, data: bytes) -> bytes:
        """将数据按 k 字节分块编码（最后一块不足则补零）。"""
        result = bytearray()
        for i in range(0, len(data), self.k):
            block = data[i:i + self.k]
            if len(block) < self.k:
                block = block + b'\x00' * (self.k - len(block))
            result.extend(self.encode_block(block))
        return bytes(result)

    # ------------------------------------------------------------------
    def _syndromes(self, codeword: bytes) -> List[int]:
        """计算症候 S_i = c(α^i)，i = 0..nroots-1。全零表示无误码。"""
        syn = []
        for i in range(self.nroots):
            alpha = gf_pow(2, i)
            syn.append(poly_eval(list(codeword), alpha))
        return syn

    # ------------------------------------------------------------------
    def decode_block(self, codeword: bytes) -> bytes:
        """解码一个 n 字节码字 → k 字节消息。

        可纠正最多 (nroots // 2) 个字节错误。
        """
        assert len(codeword) == self.n

        # ---- 症候计算 ----
        syn = self._syndromes(codeword)
        if all(s == 0 for s in syn):
            return bytes(codeword[self.nroots:])           # 无误码

        # ---- Berlekamp-Massey 算法：求错误位置多项式 σ(x) ----
        sigma:  List[int] = [1]       # 错误位置多项式
        prev_s: List[int] = [1]       # 上一次的 σ
        L: int = 0                     # 当前推测的错误数
        m: int = 1                     # x 偏移乘数
        prev_delta: int = 1            # 上一次的非零差异值

        for i in range(self.nroots):
            # 计算差异 Δ_i
            delta = syn[i]
            for j in range(1, L + 1):
                delta ^= gf_mul(sigma[j], syn[i - j])

            if delta == 0:
                m += 1
                continue

            scale = gf_mul(delta, gf_inv(prev_delta))
            T = sigma[:]

            for j in range(len(prev_s)):
                idx = j + m
                while len(sigma) <= idx:
                    sigma.append(0)
                sigma[idx] ^= gf_mul(scale, prev_s[j])

            if 2 * L <= i:
                L_new = i + 1 - L
                prev_s = T
                prev_delta = delta
                m = 1
                L = L_new
            else:
                m += 1

        sigma = sigma[:L + 1]
        poly_strip(sigma)

        error_count = len(sigma) - 1
        if error_count <= 0:
            return bytes(codeword[self.nroots:])

        # ---- Chien 搜索：求 σ(x) 的根 → 错误位置 ----
        # σ(x) 的根在 α^{-pos}，其中 pos 是错误字节的索引
        error_positions: List[int] = []
        for pos in range(self.n):
            # 计算 σ(α^{-pos})
            a_neg = gf_inv(gf_pow(2, pos))                # α^{-pos}
            if poly_eval(sigma, a_neg) == 0:
                error_positions.append(pos)

        if not error_positions or len(error_positions) > self.nroots:
            return bytes(codeword[self.nroots:])

        # ---- Forney 算法：计算错误幅值 ----
        # 错误求值多项式 Ω(x) = S(x)·σ(x) mod x^{nroots}
        s_poly = syn[:]                                    # S(x) = S_0 + S_1·x + ...
        omega = poly_mul(sigma, s_poly)
        omega = omega[:self.nroots]                        # mod x^{nroots}

        corrected = bytearray(codeword)
        for pos in error_positions:
            x = gf_pow(2, pos)                             # X_j = α^{pos}
            x_inv = gf_inv(x)                              # X_j^{-1} = α^{-pos}

            # σ'(X_j^{-1}) — 形式导数在 X_j^{-1} 处的值
            # σ'(x) = σ_1 + σ_3·x^2 + σ_5·x^4 + ...（偶次项在 GF(2) 中消零）
            sigma_deriv_val = 0
            xp = 1
            for j in range(1, len(sigma), 2):
                sigma_deriv_val ^= gf_mul(sigma[j], xp)
                xp = gf_mul(gf_mul(xp, x_inv), x_inv)      # 乘以 (X_j^{-1})^2

            if sigma_deriv_val != 0:
                # e_j = X_j · Ω(X_j^{-1}) / σ'(X_j^{-1})
                omega_val = poly_eval(omega, x_inv)
                magnitude = gf_mul(gf_mul(x, omega_val),
                                   gf_inv(sigma_deriv_val))
                corrected[pos] ^= magnitude

        return bytes(corrected[self.nroots:])

    def decode(self, data: bytes) -> bytes:
        """将数据按 n 字节码字分块解码。"""
        assert len(data) % self.n == 0
        result = bytearray()
        for i in range(0, len(data), self.n):
            result.extend(self.decode_block(data[i:i + self.n]))
        return bytes(result)


# =====================================================================
#  默认编解码器实例
# =====================================================================
_default = RSCodec(n=255, k=239)
encode = _default.encode
decode = _default.decode
