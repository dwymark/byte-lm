#!/usr/bin/env python3
"""Draw the README's figures as SVG, each in a light and a dark variant.

Every figure is 360 units wide, the width of a phone screen, so its text
stays legible when GitHub scales it to a narrow display. Numbers shown in the
figures come from running the model in onebyte.py, never from constants
typed here.

Usage:
    make_figures.py        write figures/NAME-light.svg and NAME-dark.svg
"""

import math
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import onebyte  # noqa: E402

W = 360
BYTE = onebyte.parse_weights(str(onebyte.DEFAULT_WEIGHTS))
WEIGHTS = onebyte.unpack(BYTE)

THEMES = {
    "light": dict(
        bg="#fcfcfb", card="#f1f0ec", ink="#0b0b0b", ink2="#52514e",
        line="#b9b8b2", zero="#2a78d6", one="#eb6834", on="#ffffff",
    ),
    "dark": dict(
        bg="#1a1a19", card="#272725", ink="#f0efec", ink2="#c3c2b7",
        line="#5a5955", zero="#3987e5", one="#d95926", on="#ffffff",
    ),
}
SANS = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


class Figure:
    def __init__(self, height, theme, title, desc):
        self.h = height
        self.c = THEMES[theme]
        self.parts = [
            f"<title>{escape(title)}</title>",
            f"<desc>{escape(desc)}</desc>",
            f'<rect width="{W}" height="{height}" rx="12" fill="{self.c["bg"]}"/>',
        ]

    def rect(self, x, y, w, h, fill="none", stroke="none", rx=6, sw=1.5):
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def text(self, x, y, s, size=13, fill=None, anchor="start", weight=400, mono=False):
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'font-family="{MONO if mono else SANS}" text-anchor="{anchor}" '
            f'fill="{fill or self.c["ink"]}">{escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke=None, sw=1.5, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke or self.c["line"]}" stroke-width="{sw}" '
            f'stroke-linecap="round"{extra}/>'
        )

    def arrow_down(self, x, y1, y2):
        self.line(x, y1, x, y2 - 2, stroke=self.c["ink2"])
        self.parts.append(
            f'<path d="M{x - 5:.1f} {y2 - 7:.1f} L{x:.1f} {y2:.1f} L{x + 5:.1f} {y2 - 7:.1f}" '
            f'fill="none" stroke="{self.c["ink2"]}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    def token(self, x, y, bit, size=28):
        """A bit of the text being read: 0 and 1 differ in color and digit."""
        self.rect(x, y, size, size, fill=self.c["one" if bit else "zero"], rx=6)
        self.text(x + size / 2, y + size * 0.69, str(bit), size=size * 0.6,
                  fill=self.c["on"], anchor="middle", weight=700, mono=True)

    def verdict(self, x, y, w, bit, label):
        """A rounded box naming the predicted bit."""
        self.rect(x, y, w, 40, fill=self.c["card"],
                  stroke=self.c["one" if bit else "zero"], rx=10, sw=2)
        self.token(x + 8, y + 7, bit, size=26)
        self.text(x + 42, y + 25, label, size=13, weight=700)

    def svg(self):
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {self.h}" '
            f'width="{W}" height="{self.h}" role="img">\n{body}\n</svg>\n'
        )


def fig_byte(theme):
    """The eight weights, one row per bit of the byte."""
    groups = [
        ("EMBEDDINGS", "how each bit of text is written down", [
            ("token 0", "a 0 in the text is written as +1"),
            ("token 1", "a 1 in the text is written as −1"),
        ]),
        ("ATTENTION", "how the model looks back at history", [
            ("Q  query", "what the latest bit looks for"),
            ("K  key", "how each earlier bit answers"),
            ("V  value", "what a bit says when listened to"),
            ("O  output", "how the result is written back"),
        ]),
        ("MLP", "a single on-or-off switch", [
            ("up", "feeds the switch"),
            ("down", "what the switch subtracts when on"),
        ]),
    ]
    values = list(WEIGHTS.values())
    f = Figure(574, theme, "The eight weights of the model",
               "The byte 0xBE written as eight bits, each bit labelled with the weight it stores.")
    c = f.c
    f.text(16, 30, "The whole model:", size=14, weight=700)
    for i, v in enumerate(values):
        x = 16 + i * 30
        on = v > 0
        f.rect(x, 42, 26, 30, fill=c["ink"] if on else "none", stroke=c["ink"], rx=5)
        f.text(x + 13, 63, "1" if on else "0", size=17, anchor="middle", weight=700,
               mono=True, fill=c["bg"] if on else c["ink"])
    f.text(268, 63, f"= 0x{BYTE:02X}", size=17, weight=700, mono=True)
    f.text(16, 94, "Bit 1 stores the weight +1, bit 0 stores −1.", size=12.5, fill=c["ink2"])

    y = 112
    i = 0
    for name, purpose, rows in groups:
        f.text(16, y + 16, name, size=12, weight=700)
        f.text(16 + 8.2 * len(name) + 8, y + 16, purpose, size=12, fill=c["ink2"])
        y += 26
        for label, job in rows:
            on = values[i] > 0
            f.rect(16, y, W - 32, 40, fill=c["card"], rx=8)
            f.rect(24, y + 6, 26, 28, fill=c["ink"] if on else "none", stroke=c["ink"], rx=5)
            f.text(37, y + 26, "1" if on else "0", size=16, anchor="middle", weight=700,
                   mono=True, fill=c["bg"] if on else c["ink"])
            f.text(60, y + 25, "+1" if on else "−1", size=14, weight=700)
            f.text(92, y + 17, label, size=13, weight=700)
            f.text(92, y + 33, job, size=12, fill=c["ink2"])
            y += 44
            i += 1
        y += 8
    return f


def fig_pipeline(theme):
    """One forward pass on the bits 1 1 0, top to bottom."""
    tokens = [1, 1, 0]
    t = onebyte.forward(tokens, WEIGHTS)
    answer = 0 if t["probs"][0] > t["probs"][1] else 1
    signed = lambda v, d=2: f"{v:+.{d}f}".replace("-", "−")
    steps = [
        ("Read", ["The text arrives as bits. The", "latest bit is the one on the right."], None, None),
        ("Embed", ["Write each 0 as +1", "and each 1 as −1."],
         "  ".join(signed(v, 0) for v in t["embeddings"]), None),
        ("Attention", ["Average the history. Bits that match", "the latest bit count about 7 times more."],
         signed(t["attended"]), "norm: keep only the sign"),
        ("Add", ["Add that average to the", "latest bit's own number."],
         signed(t["after_attention"]), None),
        ("MLP switch", ["If the number is positive, subtract 1.", "If it is negative, leave it alone."],
         signed(t["after_mlp"]), "norm: keep only the sign"),
        ("Output", ["Positive means the next bit is 0,", "negative means 1. Confidence: 88%."],
         None, "norm: keep only the sign"),
    ]
    card_h, gap = 76, 34
    f = Figure(20 + len(steps) * card_h + (len(steps) - 1) * gap + 16, theme,
               "One forward pass, step by step",
               "The bits 1 1 0 pass through embedding, attention, a residual add, the MLP "
               "and the output, which predicts 0.")
    c = f.c
    y = 16
    for n, (name, lines, value, pill) in enumerate(steps):
        if n:
            f.arrow_down(40, y - gap + 4, y - 4)
            if pill:
                f.rect(58, y - gap + 6, 172, 22, fill="none", stroke=c["line"], rx=11, sw=1)
                f.text(144, y - gap + 21, pill, size=11.5, fill=c["ink2"], anchor="middle")
        f.rect(16, y, W - 32, card_h, fill=c["card"], rx=10)
        f.parts.append(f'<circle cx="40" cy="{y + 24}" r="12" fill="{c["ink"]}"/>')
        f.text(40, y + 29, str(n + 1), size=13, fill=c["bg"], anchor="middle", weight=700)
        f.text(62, y + 29, name, size=15, weight=700)
        for k, s in enumerate(lines):
            f.text(28, y + 50 + k * 16, s, size=12.5, fill=c["ink2"])
        if value:
            f.text(W - 28, y + 29, value, size=14, weight=700, anchor="end")
        if name == "Read":
            for k, bit in enumerate(tokens):
                f.token(W - 28 - (len(tokens) - k) * 30 + 4, y + 10, bit, size=26)
        if name == "Output":
            f.text(W - 62, y + 29, "next:", size=13, fill=c["ink2"], anchor="end")
            f.token(W - 54, y + 10, answer, size=26)
        y += card_h + gap
    return f


def fig_vote(theme):
    """The tipping point: seven 1s then a 0, against eight 1s then a 0."""
    match, differ = math.e, 1 / math.e
    scale = 96
    f = Figure(452, theme, "The tipping point",
               "After seven 1s and a 0 the single 0 outweighs the 1s and the model predicts 0. "
               "After eight 1s and a 0 the 1s outweigh it and the model predicts 1.")
    c = f.c
    y = 14
    for ones in (7, 8):
        tokens = [1] * ones + [0]
        answer = onebyte.predict(tokens, WEIGHTS)
        word = {7: "Seven", 8: "Eight"}[ones]
        f.text(16, y + 16, f"{word} 1s, then a 0", size=14, weight=700)
        for k, bit in enumerate(tokens):
            f.token(16 + k * 31, y + 26, bit, size=27)
        pull0 = match
        pull1 = ones * differ
        by = y + 86
        f.text(16, by - 6, f"the 0 pulls   1 × {match:.2f} = {pull0:.2f}", size=12.5, fill=c["ink2"])
        f.rect(16, by, pull0 * scale, 14, fill=c["zero"], rx=4)
        by += 40
        f.text(16, by - 6, f"the 1s pull   {ones} × {differ:.2f} = {pull1:.2f}", size=12.5, fill=c["ink2"])
        seg = differ * scale
        for k in range(ones):
            f.rect(16 + k * seg, by, seg - 2, 14, fill=c["one"], rx=4)
        edge = 16 + pull0 * scale
        f.line(edge, y + 80, edge, by + 20, stroke=c["ink"], sw=1, dash="3 3")
        f.verdict(16, by + 30, 190, answer,
                  f"the {'1s win' if answer else '0 wins'}: predict {answer}")
        y += 222
    return f


def fig_rule(theme):
    """The whole behaviour as two questions."""
    threshold = math.e**2
    f = Figure(330, theme, "The rule the model follows",
               "If the latest bit is 1, predict 1. If it is 0, predict 0 unless the 1s so far "
               "outnumber the 0s by more than 7.39 to 1, in which case predict 1.")
    c = f.c
    f.rect(70, 14, 220, 44, fill=c["card"], stroke=c["ink"], rx=10)
    f.text(180, 41, "What is the latest bit?", size=14, weight=700, anchor="middle")

    f.line(120, 58, 76, 92, stroke=c["ink2"])
    f.arrow_down(76, 92, 122)
    f.text(64, 84, "it is a 1", size=12.5, fill=c["ink2"], anchor="end")
    f.verdict(16, 124, 120, 1, "predict 1")

    f.line(240, 58, 252, 80, stroke=c["ink2"])
    f.arrow_down(252, 80, 122)
    f.text(264, 100, "it is a 0", size=12.5, fill=c["ink2"])
    f.rect(160, 124, 184, 76, fill=c["card"], stroke=c["ink"], rx=10)
    for k, s in enumerate(["Do the 1s so far", "outnumber the 0s by more", f"than {threshold:.2f} to 1?"]):
        f.text(252, 146 + k * 19, s, size=13.5, weight=700, anchor="middle")

    f.arrow_down(204, 200, 244)
    f.text(196, 226, "no", size=12.5, fill=c["ink2"], anchor="end")
    f.arrow_down(300, 200, 244)
    f.text(308, 226, "yes", size=12.5, fill=c["ink2"])
    f.verdict(118, 246, 120, 0, "predict 0")
    f.verdict(246, 246, 98, 1, "predict 1")
    f.text(180, 312, "Whatever it predicts, it is 88% sure.", size=12.5, fill=c["ink2"], anchor="middle")
    return f


def fig_flip(theme):
    """Where the prediction after a 0 flips, as the count of 1s grows."""
    most = 32
    cell = 9.5
    left = 28
    f = Figure(334, theme, "How many 1s it takes to override a 0",
               "With one 0 in the history the prediction after a 0 flips to 1 at 8 ones; "
               "with two 0s at 15; with three at 23; with four at 30.")
    c = f.c
    f.text(16, 26, "The latest bit is a 0. The model predicts:", size=13, weight=700)
    f.token(16, 38, 0, size=20)
    f.text(42, 53, "a 0", size=12.5, fill=c["ink2"])
    f.token(84, 38, 1, size=20)
    f.text(110, 53, "a 1", size=12.5, fill=c["ink2"])
    y = 76
    words = ["one 0", "two 0s", "three 0s", "four 0s"]
    for zeros in range(1, 5):
        flip = next(n for n in range(most * 2)
                    if onebyte.predict([1] * n + [0] * zeros, WEIGHTS) == 1)
        f.text(16, y + 12, f"{words[zeros - 1]} so far", size=12.5, weight=700)
        f.text(W - 16, y + 12, f"flips at {flip} ones", size=12.5, fill=c["ink2"], anchor="end")
        f.rect(left, y + 20, flip * cell - 2, 14, fill=c["zero"], rx=4)
        f.rect(left + flip * cell, y + 20, (most + 1 - flip) * cell, 14, fill=c["one"], rx=4)
        y += 50
    f.line(left, y + 2, left + (most + 1) * cell, y + 2)
    for n in range(0, most + 1, 8):
        x = left + (n + 0.5) * cell
        f.line(x, y + 2, x, y + 7)
        f.text(x, y + 21, str(n), size=11.5, fill=c["ink2"], anchor="middle")
    f.text(left + (most + 1) * cell / 2, y + 38, "number of 1s so far", size=12, fill=c["ink2"], anchor="middle")
    return f


FIGURES = {
    "byte": fig_byte,
    "pipeline": fig_pipeline,
    "tipping-point": fig_vote,
    "rule": fig_rule,
    "flip": fig_flip,
}


def main():
    if len(sys.argv) > 1:
        sys.exit(__doc__)
    for name, draw in FIGURES.items():
        for theme in THEMES:
            (HERE / f"{name}-{theme}.svg").write_text(draw(theme).svg(), encoding="utf-8")
    print(f"wrote {len(FIGURES) * len(THEMES)} files to {HERE}")


if __name__ == "__main__":
    main()
