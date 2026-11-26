#!/usr/bin/env python3
"""A complete transformer language model whose weights are one byte.

The model reads bits and predicts the next bit. It has tied input and output
embeddings, one causal self-attention head, a one-unit ReLU MLP, a residual
stream one number wide, and RMSNorm before attention, before the MLP and
before the output. Every weight is +1 or -1, stored as one bit (1 means +1),
most significant bit first, in this order:

    emb0 emb1 Q K V O up down

Usage:
    onebyte.py predict BITS      probability of each next bit
    onebyte.py trace BITS        every intermediate value of the forward pass
    onebyte.py generate BITS     continue a bit string (--n, --seed, --greedy)
    onebyte.py text STRING       show a string as bits and predict the next bit
    onebyte.py check             compare the model with the closed-form rule
                                 on every bit string up to --max-len (12)
    onebyte.py survey            group all 256 weight bytes by what they compute
                                 on every bit string up to --max-len (10)

Every command accepts --weights, a byte such as 0xBE or a path to a one-byte
file; the default is weights.bin beside this script.
"""

import argparse
import itertools
import math
import random
import sys
from pathlib import Path

WEIGHT_NAMES = ("emb0", "emb1", "Q", "K", "V", "O", "up", "down")
DEFAULT_WEIGHTS = Path(__file__).with_name("weights.bin")


def unpack(byte):
    """Map a byte to the eight weights, each +1.0 or -1.0."""
    bits = [(byte >> (7 - i)) & 1 for i in range(8)]
    return {name: 1.0 if bit else -1.0 for name, bit in zip(WEIGHT_NAMES, bits)}


def rms_norm(x):
    """RMSNorm of a width-1 vector with no learned gain: x / sqrt(mean(x^2)).

    At width 1 this is the sign of x. A zero input, which the 0xBE weights
    never produce, stays zero.
    """
    return 0.0 if x == 0 else x / math.sqrt(x * x)


def forward(tokens, w):
    """Run the model on a non-empty bit sequence and trace the last position.

    Returns a dict of every intermediate value; "probs" holds the predicted
    probabilities of next bit 0 and next bit 1.
    """
    emb = (w["emb0"], w["emb1"])
    stream = [emb[t] for t in tokens]

    normed = [rms_norm(x) for x in stream]
    query = w["Q"] * normed[-1]
    scores = [query * w["K"] * h for h in normed]
    peak = max(scores)
    unnormalized = [math.exp(s - peak) for s in scores]
    total = sum(unnormalized)
    attention = [u / total for u in unnormalized]
    attended = w["O"] * sum(a * w["V"] * h for a, h in zip(attention, normed))

    after_attention = stream[-1] + attended
    hidden = max(0.0, w["up"] * rms_norm(after_attention))
    after_mlp = after_attention + w["down"] * hidden

    y = rms_norm(after_mlp)
    logits = (y * emb[0], y * emb[1])
    exps = [math.exp(v) for v in logits]
    probs = tuple(e / sum(exps) for e in exps)

    return {
        "embeddings": stream,
        "scores": scores,
        "attention": attention,
        "attended": attended,
        "after_attention": after_attention,
        "hidden": hidden,
        "after_mlp": after_mlp,
        "logits": logits,
        "probs": probs,
    }


def predict(tokens, w):
    """Return the more probable next bit."""
    p0, p1 = forward(tokens, w)["probs"]
    return 0 if p0 > p1 else 1


def rule(tokens):
    """The closed form of the 0xBE model: the next bit it prefers.

    After a 1 it predicts 1. After a 0 it predicts 0 unless the 1s so far
    outnumber the 0s by more than e^2 to 1.
    """
    if tokens[-1] == 1:
        return 1
    zeros = tokens.count(0)
    ones = len(tokens) - zeros
    return 1 if ones > math.e**2 * zeros else 0


def text_to_bits(text):
    """Tokenize: UTF-8 bytes, each as eight bits, most significant first."""
    return [(b >> (7 - i)) & 1 for b in text.encode("utf-8") for i in range(8)]


def parse_bits(s):
    bits = [c for c in s if c not in " _"]
    if not bits or any(c not in "01" for c in bits):
        raise argparse.ArgumentTypeError("expected a non-empty string of 0s and 1s")
    return [int(c) for c in bits]


def parse_weights(s):
    path = Path(s)
    if path.is_file():
        data = path.read_bytes()
        if len(data) != 1:
            raise argparse.ArgumentTypeError(f"{s} holds {len(data)} bytes, not 1")
        return data[0]
    try:
        byte = int(s, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s} is neither a file nor a byte")
    if not 0 <= byte <= 255:
        raise argparse.ArgumentTypeError("a weight byte is between 0 and 255")
    return byte


def show(bits):
    return "".join(map(str, bits))


def cmd_predict(args, w):
    p0, p1 = forward(args.bits, w)["probs"]
    print(f"P(next = 0) = {p0:.4f}")
    print(f"P(next = 1) = {p1:.4f}")


def cmd_trace(args, w):
    t = forward(args.bits, w)
    row = lambda values: " ".join(f"{v:+.3f}" for v in values)
    print(f"tokens            {'      '.join(map(str, args.bits))}")
    print(f"embeddings        {row(t['embeddings'])}")
    print(f"attention scores  {row(t['scores'])}")
    print(f"attention weights {row(t['attention'])}")
    print(f"attention output  m = {t['attended']:+.4f}")
    print(f"residual add      x = {t['after_attention']:+.4f}")
    print(f"MLP hidden unit   {t['hidden']:.0f}")
    print(f"after MLP         x = {t['after_mlp']:+.4f}")
    print(f"logits (0, 1)     {row(t['logits'])}")
    print(f"P(next = 0), P(next = 1) = {t['probs'][0]:.4f}, {t['probs'][1]:.4f}")


def cmd_generate(args, w):
    rng = random.Random(args.seed)
    bits = list(args.bits)
    for _ in range(args.n):
        p0, p1 = forward(bits, w)["probs"]
        if args.greedy:
            bits.append(0 if p0 > p1 else 1)
        else:
            bits.append(0 if rng.random() < p0 else 1)
    prompt = len(args.bits)
    print(f"{show(bits[:prompt])} {show(bits[prompt:])}")


def cmd_text(args, w):
    bits = text_to_bits(args.string)
    if not bits:
        sys.exit("the string is empty")
    print(" ".join(show(bits[i : i + 8]) for i in range(0, len(bits), 8)))
    args.bits = bits
    cmd_predict(args, w)


def cmd_check(args, w):
    confidence = 1 / (1 + math.exp(-2))
    checked = 0
    for length in range(1, args.max_len + 1):
        for tokens in itertools.product((0, 1), repeat=length):
            probs = forward(tokens, w)["probs"]
            got = 0 if probs[0] > probs[1] else 1
            if got != rule(tokens) or abs(max(probs) - confidence) > 1e-12:
                sys.exit(f"mismatch on {show(tokens)}: model {got}, rule {rule(tokens)}")
            checked += 1
    print(f"{checked} bit strings up to length {args.max_len}: the model matches")
    print(f"the rule on every one, each time with confidence {confidence:.4f}")


def cmd_survey(args, w):
    strings = [
        tokens
        for length in range(1, args.max_len + 1)
        for tokens in itertools.product((0, 1), repeat=length)
    ]
    groups = {}
    for byte in range(256):
        weights = unpack(byte)
        behaviour = tuple(round(forward(t, weights)["probs"][0], 9) for t in strings)
        groups.setdefault(behaviour, []).append(byte)
    print(f"{len(groups)} distinct behaviours among the 256 weight bytes,")
    print(f"compared on every bit string up to length {args.max_len}:")
    for members in sorted(groups.values(), key=len):
        shown = " ".join(f"0x{b:02X}" for b in members[:8])
        more = " ..." if len(members) > 8 else ""
        print(f"{len(members):4d} bytes  {shown}{more}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", type=parse_weights, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("predict", "trace"):
        sub.add_parser(name).add_argument("bits", type=parse_bits)
    gen = sub.add_parser("generate")
    gen.add_argument("bits", type=parse_bits)
    gen.add_argument("--n", type=int, default=48)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--greedy", action="store_true")
    sub.add_parser("text").add_argument("string")
    sub.add_parser("check").add_argument("--max-len", type=int, default=12)
    sub.add_parser("survey").add_argument("--max-len", type=int, default=10)
    args = parser.parse_args()

    byte = parse_weights(str(DEFAULT_WEIGHTS)) if args.weights is None else args.weights
    w = unpack(byte)
    commands = {
        "predict": cmd_predict,
        "trace": cmd_trace,
        "generate": cmd_generate,
        "text": cmd_text,
        "check": cmd_check,
        "survey": cmd_survey,
    }
    commands[args.command](args, w)


if __name__ == "__main__":
    main()
