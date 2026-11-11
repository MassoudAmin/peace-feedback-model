import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path
from PIL import Image

BLUE = "#2B4396"; MAROON = "#7A0019"; BLACK = "#000000"; GRAY = "#505050"
SERIF = "DejaVu Serif"; SANS = "DejaVu Sans"

fig = plt.figure(figsize=(16, 11))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 110); ax.axis("off")
ax.text(80, 107, "The loops that decide the outcome", ha="center", va="center", fontsize=26.0, color=BLUE, weight="bold", family=SERIF)
ax.text(80, 102.8, "An arrow marked + means the two quantities move in the same direction; \u2212 means opposite. "
        "A loop with an even number of \u2212 links feeds itself (R). An odd number corrects (B).",
        ha="center", va="center", fontsize=11.5, color=GRAY, family=SANS)
ax.plot([95, 95], [8, 99], color=GRAY, lw=0.6, ls=(0, (4, 4)))
ax.text(128, 96.5, "Between states", ha="center", va="center", fontsize=16.9, color=BLUE, weight="bold", family=SERIF)
ax.text(14, 99.2, "Inside a society", ha="center", va="center", fontsize=16.9, color=BLUE, weight="bold", family=SERIF)

N = {
 "T": (12, 92, 16, 6, "Transparency"),
 "P": (20, 14, 20, 9, "Problems seen\nby institutions"),
 "I": (52, 24, 18, 9, "Institutional\nresponse"),
 "G": (34, 44, 16, 6, "Grievance"),
 "V": (70, 44, 14, 6, "Violence"),
 "R": (70, 12, 15, 6, "Retaliation"),
 "F": (70, 74, 18, 9, "Fear and\nrumor"),
 "S": (34, 74, 20, 9, "Shortage,\nrising prices"),
 "H": (52, 92, 14, 6, "Hoarding"),
 "M": (18, 60, 14, 9, "Reserves\nat home"),
 "A": (128, 86, 22, 9, "Weapons\nspending"),
 "Th": (128, 62, 22, 9, "Threat felt by\nthe other side"),
 "D": (128, 38, 22, 9, "Talks and\ninspections"),
 "Mo": (128, 14, 22, 9, "Reserves\nat home"),
}
for k, (x, y, w, h, name) in N.items():
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0,rounding_size=1", ec=BLUE, fc="white", lw=1.6))
    ax.text(x, y, name, ha="center", va="center", fontsize=15.0, color=BLUE, weight="bold", family=SERIF)

def L(p0, p1, sign, sxy, ctrl=None, ctrl2=None, ls="-"):
    if ctrl is None:
        path = Path([p0, p1], [Path.MOVETO, Path.LINETO])
    elif ctrl2 is None:
        path = Path([p0, ctrl, p1], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    else:
        path = Path([p0, ctrl, ctrl2, p1], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=18, color=BLACK, lw=1.5, fc=BLACK, ls=ls))
    ax.text(sxy[0], sxy[1], sign, ha="center", va="center", fontsize=20.8, color=BLACK, weight="bold", family=SANS)

m = "\u2212"
# B1: grievance -> problems seen -> response -> grievance
L((29, 41), (22, 18.5), "+", (28.5, 30))
L((30, 14), (43, 22), "+", (41.5, 16.5))
L((46, 28.5), (38, 41), m, (36.5, 37.5))
# transparency
L((7, 89), (14, 18.5), "+", (9.5, 24))
L((20, 92), (78, 78.5), m, (82, 83), ctrl=(60, 109))
# R1: fear -> hoarding -> shortage -> fear ; shortage -> grievance
L((66, 78.5), (59, 92), "+", (66, 89), ctrl=(66, 92))
L((45, 92), (36, 78.5), "+", (37, 88), ctrl=(38, 92))
L((44, 74), (61, 74), "+", (52, 76.5))
L((34, 69.5), (34, 47), "+", (36.5, 58))
# R2: grievance -> violence ; violence <-> retaliation ; violence -> fear ; response -> violence
L((42, 44), (63, 44), "+", (52, 46.5))
L((73, 41), (73, 15), "+", (81, 20), ctrl=(84, 28))
L((67, 15), (67, 41), "+", (56.5, 33), ctrl=(56, 28))
L((70, 47), (70, 69.5), "+", (72.5, 58))
L((58, 28.5), (65, 41), m, (66.5, 35.5))
# reserves -> shortage
L((25, 62), (28, 69.5), m, (31, 66))
# outer loop
L((122, 81.5), (122, 66.5), "+", (118, 74), ctrl=(117, 74))
L((134, 66.5), (134, 81.5), "+", (138, 74), ctrl=(139, 74))
L((134, 57.5), (134, 42.5), "+", (138, 50), ctrl=(139, 50))
L((122, 42.5), (122, 57.5), m, (118, 50), ctrl=(117, 50))
L((139, 86), (139, 14), m, (152, 48), ctrl=(160, 50))
L((117, 63), (79, 73), "+", (98, 70), ls="--")

def loop(x, y, name, kind, desc, dxy):
    col = MAROON if kind == "R" else BLUE
    ax.add_patch(Circle((x, y), 3.2, ec=col, fc="white", lw=1.6))
    ax.text(x, y, name, ha="center", va="center", fontsize=14.3, color=col, weight="bold", family=SANS)
    ax.text(x + dxy[0], y + dxy[1], desc, ha="center", va="center", fontsize=11.4, color=col, family=SANS, style="italic")

loop(34, 29, "B1", "B", "a complaint is heard\nand answered", (0, -5.6))
loop(52, 84, "R1", "R", "fear feeds shortage,\nshortage feeds fear", (0, -5.4))
loop(90, 30, "R2", "R", "each blow\nbrings the next", (0, -5.6))
loop(128, 74, "R3", "R", "arms race", (0, -5.4))
loop(128, 50, "B2", "B", "talks lower\nthe threat", (0, -5.6))

ax.text(80, 3.6, "The dashed arrow joins the two panels: threat across a border raises fear at home.\n"
        "Reserves at home appear in both panels: they absorb shortages on the left and are spent on weapons on the right.",
        ha="center", va="center", fontsize=13.0, color=BLACK, family=SANS)

fig.savefig("figures/peace_causal_loops.png", dpi=300, facecolor="white")
fig.savefig("figures/peace_causal_loops.pdf", facecolor="white")
Image.open("figures/peace_causal_loops.png").convert("RGB").save("figures/peace_causal_loops.jpg", quality=95)
print("ok")
