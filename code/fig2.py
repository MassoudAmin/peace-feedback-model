import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image

BLUE = "#2B4396"; MAROON = "#7A0019"; BLACK = "#000000"; GRAY = "#505050"; TEAL = "#2B4396"; RED = "#7A0019"
SERIF = "DejaVu Serif"; SANS = "DejaVu Sans"

def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=300, facecolor="white")
    fig.savefig(f"figures/{name}.pdf", facecolor="white")
    Image.open(f"figures/{name}.png").convert("RGB").save(f"figures/{name}.jpg", quality=95)

# =====================================================================
# Figure 1: standard feedback control block diagram
# =====================================================================
fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 160); ax.set_ylim(0, 90); ax.axis("off")

def block(x, y, w, h, title, sub, ec=BLUE, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1", ec=ec, fc="white", lw=1.8, ls=ls))
    ax.text(x + w/2, y + h*0.70, title, ha="center", va="center", fontsize=14.9, color=ec, weight="bold", family=SERIF)
    ax.text(x + w/2, y + h*0.33, sub, ha="center", va="center", fontsize=10.9, color=BLACK, family=SANS, linespacing=1.25)

def arr(p, q, color=BLACK, lw=1.8, label=None, lxy=None, fs=10, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=20, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))
    if label:
        ax.text(lxy[0], lxy[1], label, ha="center", va="center", fontsize=fs*1.15, color=color, family=SANS)

ax.text(80, 84, "Peace as a feedback loop", ha="center", va="center", fontsize=23.0, color=BLUE, weight="bold", family=SERIF)
ax.text(80, 79.5, "The block diagram used for a power grid, with a society in place of the grid. Peace lasts while this loop closes faster than the disturbances arrive.",
        ha="center", va="center", fontsize=11, color=GRAY, family=SANS)

Y = 46
# setpoint
ax.text(9, Y + 1.5, "Setpoint", ha="center", va="bottom", fontsize=12.6, color=BLACK, weight="bold", family=SANS)
ax.text(9, Y - 1.5, "what people expect:\nsafety, services,\nfair rules, a livelihood", ha="center", va="top", fontsize=9.8, color=BLACK, family=SANS)
# comparator
ax.add_patch(Circle((22, Y), 2.6, ec=BLACK, fc="white", lw=1.8))
ax.text(22, Y, "\u2211", ha="center", va="center", fontsize=16.1, color=BLACK, family=SANS)
arr((14.5, Y), (19.3, Y)); ax.text(17, Y + 1.8, "+", ha="center", va="center", fontsize=13.8, color=BLACK, family=SANS)
# error
arr((24.6, Y), (33, Y), label="error signal\n(grievance)", lxy=(28.8, Y + 5.2), fs=9)
# controller
block(33, Y - 8, 30, 16, "Correction: institutions", "courts, elections, markets,\nlocal government, civil society\ngain \u00b7 speed \u00b7 saturation", ec=BLUE)
arr((63, Y), (70, Y))
# actuators
block(70, Y - 8, 26, 16, "Actions", "rulings, spending, repair crews,\nnegotiation, publishing numbers\n(delay: hours to years)", ec=BLUE)
arr((96, Y), (103, Y))
# plant
block(103, Y - 8, 32, 16, "The society", "livelihoods \u00b7 basic services\nsecurity \u00b7 trust\nreserves absorb shocks", ec=BLUE)
# disturbance
block(106, 66, 26, 9, "Disturbances", "drought, price shocks, outages,\nepidemic, war, disinformation", ec=BLUE, ls="--")
arr((119, 66), (119, Y + 8), color=BLACK)
# output
arr((135, Y), (146, Y), label="observed\nconditions", lxy=(141, Y + 5), fs=9)
# feedback path down and back
ax.plot([146, 146, 92], [Y, 20, 20], color=TEAL, lw=1.8)
# sensor
block(62, 14, 30, 12, "Measurement", "transparency: prices, outage records,\ncourt backlog, free press, audits", ec=TEAL)
ax.add_patch(FancyArrowPatch((146, 20), (92, 20), arrowstyle="-|>", mutation_scale=20, color=TEAL, lw=1.8))
ax.plot([62, 22, 22], [20, 20, Y - 2.6], color=TEAL, lw=1.8)
ax.add_patch(FancyArrowPatch((22, 25), (22, Y - 2.6), arrowstyle="-|>", mutation_scale=20, color=TEAL, lw=1.8))
ax.text(24.3, Y - 5, "\u2212", ha="center", va="center", fontsize=16.1, color=TEAL, family=SANS)
ax.text(42, 17.2, "measured state (delay: days to years)", ha="center", va="center", fontsize=10.3, color=TEAL, family=SANS)
# secrecy note
ax.text(112, 15.3, "Secrecy cuts this wire. With it cut, institutions\nact on rumor and the loop runs open.", ha="center", va="center", fontsize=10.3, color=RED, family=SANS, style="italic")

# stability box
ax.add_patch(FancyBboxPatch((8, 1.2), 144, 8.6, boxstyle="round,pad=0,rounding_size=1", ec=MAROON, fc="white", lw=1.4))
ax.text(80, 8.2, "Stability conditions", ha="center", va="center", fontsize=12.6, color=MAROON, weight="bold", family=SERIF)
ax.text(80, 4.0, "1. The measurement must show the true state (transparency).    2. The correction must match the error: the penalty fits the offense, the aid fits the loss.\n"
        "3. The loop must close faster than a shock spreads.    4. Reserves must last until it does.",
        ha="center", va="center", fontsize=10.3, color=BLACK, family=SANS, linespacing=1.4)
save(fig, "peace_control_loop")
plt.close(fig)

