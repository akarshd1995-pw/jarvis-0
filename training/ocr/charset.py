"""The characters the reader can output. Index 0 is the CTC blank."""
CHARS = [chr(c) for c in range(32, 127)] + list("₹•–—‘’“”…©®°×÷€£")
BLANK = 0
IDX = {c: i + 1 for i, c in enumerate(CHARS)}
N_CLASSES = len(CHARS) + 1
NORMALIZE = {"\t": " ", "\u00a0": " ", "‐": "-", "‑": "-", "−": "-"}


def encode(text: str) -> list[int]:
    out = []
    for ch in text:
        ch = NORMALIZE.get(ch, ch)
        if ch in IDX:
            out.append(IDX[ch])
    return out


def decode_greedy(ids: list[int]) -> str:
    """CTC: collapse repeats, drop blanks."""
    out, prev = [], BLANK
    for i in ids:
        if i != prev and i != BLANK:
            out.append(CHARS[i - 1])
        prev = i
    return "".join(out)
