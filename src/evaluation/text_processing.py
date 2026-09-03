"""
Text Processing — Vietnamese NLP
=================================
Chuẩn hóa text và tokenize tiếng Việt sử dụng Underthesea.
Được sử dụng bởi cả metrics module và evaluator.
"""

import re

# ── Vietnamese tokenizer ──────────────────────────────────────────────
# Underthesea cung cấp word segmentation tiếng Việt chính xác,
# ví dụ: "Thành phố Hồ Chí Minh" → ["Thành_phố", "Hồ_Chí_Minh"]
try:
    from underthesea import word_tokenize as _underthesea_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False


def normalize_text(text: str) -> str:
    """Chuẩn hóa text để so sánh: lowercase, bỏ dấu câu thừa, gộp whitespace."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_vi(text: str) -> list[str]:
    """Tokenize tiếng Việt sử dụng Underthesea word segmentation.

    Underthesea tách từ ghép tiếng Việt chính xác hơn, ví dụ:
      "quyền sử dụng đất" → ["quyền", "sử_dụng", "đất"]
    thay vì split thô: ["quyền", "sử", "dụng", "đất"]

    Fallback về simple split nếu underthesea chưa được cài đặt.
    """
    text = text.strip()
    if not text:
        return []

    if HAS_UNDERTHESEA:
        # word_tokenize trả về chuỗi đã tách từ, vd: "quyền sử_dụng đất"
        segmented = _underthesea_tokenize(text, format="text")
        return normalize_text(segmented).split()
    else:
        return normalize_text(text).split()


def safe_print(*args, **kwargs):
    """In ra stdout an toàn, tự động thay thế ký tự không hỗ trợ thay vì crash."""
    import sys
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    flush = kwargs.get('flush', False)
    file = kwargs.get('file', sys.stdout)
    text = sep.join(str(arg) for arg in args) + end
    try:
        file.write(text)
    except (UnicodeEncodeError, UnicodeError):
        enc = getattr(file, 'encoding', None) or 'utf-8'
        try:
            if hasattr(file, 'buffer'):
                file.buffer.write(text.encode('utf-8', errors='replace'))
            else:
                sanitized = text.encode(enc, errors='replace').decode(enc, errors='replace')
                file.write(sanitized)
        except Exception:
            try:
                file.write(text.encode('ascii', errors='replace').decode('ascii'))
            except Exception:
                pass
    if flush:
        try:
            file.flush()
        except Exception:
            pass
