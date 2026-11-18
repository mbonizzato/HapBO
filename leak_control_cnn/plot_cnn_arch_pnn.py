"""Panel A: the conv-encoder drawn with PlotNeuralNet, with the four searched
hyperparameters labelled. Blocks are kept small relative to the labels.
-> arch_cnn_leak.tex/.pdf."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VEND = os.path.join(HERE, "pnn")
# PlotNeuralNet (https://github.com/HarisIqbal88/PlotNeuralNet) must be available:
# either vendor it under ./pnn/ (so ./pnn/layers exists) or point PLOTNEURALNET_DIR at
# a local checkout.
FALLBACK = os.environ.get("PLOTNEURALNET_DIR", VEND)
PNN = VEND if os.path.isdir(os.path.join(VEND, "layers")) else FALLBACK
sys.path.append(PNN)
from pycore.tikzeng import to_head, to_begin, to_end, to_connection, to_generate  # noqa: E402


def to_box(name, offset, to, width, height, depth, caption, fill, opacity=0.6):
    return (r"""\pic[shift={""" + offset + """}] at """ + to + r"""
    {Box={name=""" + name + """, caption=""" + caption + r""", xlabel={{" ", }}, zlabel=,
        fill=""" + fill + """, opacity=""" + str(opacity) + """, height=""" + str(height) +
        """, width=""" + str(width) + """, depth=""" + str(depth) + """}};
""")


def dim_above(name, text):
    return (r"""\node[above, font=\small, inner sep=3pt] at (""" + name +
            r"""-north) {""" + text + r"""};
""")


def label_below(name, text, drop):
    return (r"""\node[below=""" + str(drop) + r"""pt, font=\small, align=center]
        at (""" + name + r"""-south) {""" + text + r"""};
""")


def param_below(name, text, drop):
    return (r"""\node[below=""" + str(drop) + r"""pt, font=\small, text=red!75!black,
        align=center] at (""" + name + r"""-south) {""" + text + r"""};
""")


COLORS = r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\InputColor{rgb:red,1;green,1;blue,1;white,6}
\def\GapColor{rgb:blue,2;green,1;black,0.3}
"""

# small blocks (heights/depths ~1/3 of a legible-caption scale), large text
arch = [
    to_head(PNN),
    COLORS,
    to_begin(),

    # Title + searched training knobs (large text relative to the small blocks)
    r"""\node[font=\Large\bfseries] at (3.4,3.1) {Custom convolutional encoder (trained from scratch)};""",
    r"""\node[font=\small, text=red!75!black, align=center] at (3.4,2.35)
        {searched (Adam):\ \textbf{learning\_rate}$\in\{10^{-4}\!-\!3\!\cdot\!10^{-2}\}$,\quad
         \textbf{weight\_decay}$\in\{0\!-\!10^{-2}\}$};""",

    to_box("inp", "(0,0,0)", "(0,0,0)", 0.5, 11, 11, " ", r"\InputColor", 0.35),
    dim_above("inp", r"$1\times28\times28$"),
    label_below("inp", r"Input", 4),

    to_box("b1", "(1.5,0,0)", "(inp-east)", 1.0, 7.5, 7.5, " ", r"\ConvColor"),
    dim_above("b1", r"$C\times14\times14$"),
    label_below("b1", r"Conv, BN,\\ReLU, Pool", 4),
    to_connection("inp", "b1"),

    to_box("b2", "(1.7,0,0)", "(b1-east)", 1.6, 4.0, 4.0, " ", r"\ConvColor"),
    dim_above("b2", r"$2C\times7\times7$"),
    label_below("b2", r"Conv, BN,\\ReLU, Pool", 4),
    to_connection("b1", "b2"),

    to_box("gap", "(1.5,0,0)", "(b2-east)", 0.9, 2.7, 2.7, " ", r"\GapColor"),
    dim_above("gap", r"$2C\times4\times4$"),
    label_below("gap", r"Adaptive\\AvgPool", 4),
    to_connection("b2", "gap"),

    to_box("head", "(1.5,0,0)", "(gap-east)", 0.9, 1.0, 4.5, " ", r"\FcColor", 0.85),
    dim_above("head", r"$10$"),
    label_below("head", r"Flatten,\\Dropout,\\Linear", 4),
    to_connection("gap", "head"),

    param_below("b1", r"\textbf{base\_channels}\\$C\in\{8,16,24,32,48\}$", 48),
    param_below("head", r"\textbf{dropout}\\$\in\{0,.1,.25,.4,.5\}$", 74),

    to_end(),
]

if __name__ == "__main__":
    outdir = os.path.join(HERE, "figures")
    os.makedirs(outdir, exist_ok=True)
    tex = os.path.join(outdir, "arch_cnn_leak.tex")
    to_generate(arch, tex)
    print("wrote", tex, "| PNN =", PNN)
