import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

BLUE = "#2B4396"; MAROON = "#7A0019"; BLACK = "#000000"; GRAY = "#505050"; TEAL = "#2B4396"; RED = "#7A0019"
SERIF = "DejaVu Serif"; SANS = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"

fig = plt.figure(figsize=(17, 26))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 170); ax.set_ylim(-20, 240); ax.axis("off")
y = [234]

def T(txt, x=6, dy=0, size=11.5, color=BLACK, fam=SANS, weight="normal", ha="left", style="normal"):
    ax.text(x, y[0], txt, ha=ha, va="top", fontsize=size, color=color, family=fam, weight=weight, style=style)
    y[0] -= dy

def H(txt, dy=6.2):
    y[0] -= 2.5
    ax.plot([6, 164], [y[0] + 1.2, y[0] + 1.2], color=BLUE, lw=0.8)
    T(txt, size=14, color=BLUE, fam=SERIF, weight="bold", dy=dy)

def E(latex, note, dy=7.2, x=10, nx=88):
    ax.text(x, y[0], latex, ha="left", va="top", fontsize=14.6, color=BLACK)
    ax.text(nx, y[0] + 0.2, note, ha="left", va="top", fontsize=10.8, color=GRAY, family=SANS, linespacing=1.3)
    y[0] -= dy

ax.text(85, 237.5, "Peace as a coupled dynamical system: the equations behind the diagrams", ha="center", va="top", fontsize=21.3, color=BLUE, weight="bold", family=SERIF)
y[0] = 231.5
T("Every claim in the diagrams is written here as a term with a sign, a coupling, and a time constant. Coefficients can be estimated from public series (FAO food prices, utility outage reports,\n"
  "UCDP fatalities, SIPRI military spending, newspaper archives, World Values Survey) on wars that began 1990 to 2020, and checked on wars outside the fit. Spatial spread adds a network term.",
  size=10.2, color=GRAY, dy=11)

H("1.  Variables")
T(r"State $\mathbf{x}(t)$ (what the society is):   $L$ livelihoods   $S$ basic services   $V$ violence   $F$ fear   $T$ trust   $M$ margins (reserves)", size=11.2, dy=4.4)
T(r"Outer loop states:   $A$ own arms   $B$ the other side's arms   $\Theta$ threat perceived by them   $H$ hostility   $C_r$ credibility", size=11.2, dy=5.2)
T(r"Disturbances $\mathbf{d}(t)$ (not chosen):   $d_P$ price shock   $d_S$ outage, drought, epidemic   $d_R$ disinformation   $d_X$ external pressure", size=11.2, dy=5.2)
T(r"Controls $\mathbf{u}(t)$ (chosen by institutions):   $u_{law}$ rulings and enforcement   $u_{sp}$ spending and aid   $u_{rep}$ repair   $u_{talk}$ dialogue   $u_{arm}$ arms spending   $\tau\in[0,1]$ transparency   $v\in[0,1]$ verification", size=11.2, dy=5.2)
T(r"Setpoint $\mathbf{r}$: what people expect of $L, S, V, T$.   Measurements $\mathbf{y}(t)$: what institutions can actually see.   Delays $\delta_s$ (sensing), $\delta_a$ (acting).", size=11.2, dy=7)

H("2.  Plant: the society")
E(r"$\dot L = a_1 S \;-\; a_2 V \;-\; d_P(t) \;+\; u_{sp} \;-\; \kappa\, u_{arm}$",
  "Income rises with working services, falls with violence and price shocks; spending helps;\narms spending crowds it out (the guns-and-butter coupling).")
E(r"$\dot S = u_{rep} \;-\; \gamma S \;-\; \phi\left(d_S(t),\, M\right), \qquad \phi(d,M)=d\cdot\dfrac{m_0}{m_0+M}$",
  "Services decay unless repaired. A shock reaches them only to the degree the margins\ncannot absorb it: large $M$ makes $\\phi$ small. This is the first nonlinearity.", dy=8.5)
E(r"$\dot M = \rho \;-\; \theta\, d_S(t) \;-\; \kappa\, u_{arm} \;-\; \lambda\, h F$",
  "Reserves are rebuilt slowly ($\\rho$), spent by shocks, by arms, and by hoarding ($hF$).")
E(r"$\dot V = b_1 G \;+\; b_2 V \;-\; b_3\, u_{law}\, T \;-\; b_4 T$",
  "Grievance starts violence; $b_2 V$ is revenge (each act breeds the next); enforcement\nworks only multiplied by trust, so $u_{law}$ with $T\\approx 0$ does almost nothing.")
E(r"$\dot F = c_1 V \;+\; c_2\, h F \;+\; c_3 \Theta_{self} \;+\; c_4 d_R(t) \;-\; c_5\, \tau \;-\; c_6 F$",
  "Fear grows with violence, with its own hoarding effect ($hF$: fear empties shelves,\nempty shelves feed fear), with external threat and disinformation; transparency ($\\tau$)\ndrains it; $c_6$ is natural decay.", dy=9.5)
E(r"$\dot T = e_1\, \|\mathbf{u}\| \;-\; e_2 V \;-\; e_3\, \|\mathbf{y}-\mathbf{x}\|\;-\; e_4 T$",
  "Trust is earned by visible response, lost to violence, and lost fastest when what is\npublished ($\\mathbf{y}$) differs from what people live ($\\mathbf{x}$).")
E(r"$G = \sum_i w_i\,(r_i - x_i)^{+}$",
  "Grievance is the felt gap between expectation and reality, on the true state, not the\npublished one. $(\\cdot)^+$ means only shortfalls count.", dy=8)

H("3.  Sensor: transparency as observability")
E(r"$\mathbf{y}(t) = \tau\, C\,\mathbf{x}(t-\delta_s) \;+\; (1-\tau)\,\boldsymbol{\eta}(t) \;+\; \boldsymbol{\nu}(t)$",
  "What institutions see is the true state, delayed, scaled by transparency $\\tau$, plus\nrumor $\\eta$ that fills whatever $\\tau$ leaves dark, plus ordinary noise $\\nu$.")
E(r"$\tau \to 0 \;\Rightarrow\; \mathbf{y} \to \boldsymbol{\eta}: \quad$ the controller acts on rumor; the loop is open.",
  "Secrecy does not remove the feedback loop; it replaces the sensor with rumor.", dy=8)

H("4.  Controller: institutions")
E(r"$\mathbf{e}(t) = \mathbf{r} - \mathbf{y}(t), \qquad \mathbf{u}(t) = \mathrm{sat}(K\, \mathbf{e}(t-\delta_a),\; \mathbf{u}_{max})$",
  "Institutions act on the measured error, after a delay, with gain $K$ (penalty fits the\noffense, aid fits the loss) and a hard ceiling $\\mathbf{u}_{max}$ (courts, crews, budgets).")
E(r"$K = \mathrm{diag}(k_{courts},\, k_{elections},\, k_{markets},\, k_{local},\, k_{civil})$",
  "Several controllers in parallel, each with its own gain and speed, with time constants from hours to years.\nRedundancy: if one saturates or is captured ($k_i \\to 0$), the others must still carry the error.", dy=8)

H("5.  Outer loop: between states (Richardson, extended)")
E(r"$\dot A = k\,\Theta_{self} \;-\; m A \;+\; g, \qquad \dot B = k'\,\Theta_{other} \;-\; m' B \;+\; g'$",
  "Each side arms in proportion to the threat it perceives, restrained by cost ($m, m'$),\nwith standing grievance $g, g'$. Richardson, 1919.")
E(r"$\Theta_{other} = A + (1-v)\,\epsilon \;+\; H, \qquad \dot H = h_1 \Theta_{other} \;-\; h_2\, u_{talk}\,C_r \;-\; h_3\, \mathrm{Trade} \;+\; h_4 H$",
  "Threat is real arms plus the uncertainty $\\epsilon$ that verification $v$ removes, plus\nhostility $H$. Talks lower hostility only as far as credibility $C_r$ allows; trade\nbrakes it; $h_4 H$ is the spiral.", dy=9.5)
E(r"$\dot{C_r} = \sigma\,(\mathrm{kept}) \;-\; \omega\,(\mathrm{broken}) \;-\; \zeta\,C_r$",
  "Credibility is the intangible state of the outer loop: built slowly, spent fast, decays\nwith memory.", dy=8)

H("6.  What the equations say about stability")
T(r"Arms race (R3) is stable only if  $m\,m' > k\,k'$ : restraint on both sides must outweigh reaction. Verification lowers $\epsilon$ and so lowers the effective $k$.", size=11.2, dy=5)
T(r"Fear loop (R1) is stable only if  $c_2 h < c_6 + c_5\,\tau/F$ : hoarding gain must stay below the drain, and transparency is the one term the controller can raise quickly.", size=11.2, dy=5)
T(r"Revenge loop (R2) is stable only if  $b_2 < b_3 u_{law} T + b_4 T$ : enforcement times trust must exceed the revenge rate; with $T \to 0$ the loop runs free whatever $u_{law}$ is.", size=11.2, dy=5)
T(r"Loop delay: the correction must arrive before the shock spreads, $\delta_s + \delta_a < 1/\max(b_2,\, c_2 h,\, h_4)$; margins $M$ buy exactly that much time and no more.", size=11.2, dy=5)
T(r"Coupling: $u_{arm}$ appears in $\dot L$, $\dot M$ and $\dot A$ at once. Winning the outer loop by spending drains the margins that keep the inner loops balanced.", size=11.2, dy=7)

ax.add_patch(FancyBboxPatch((6, y[0] - 9.5), 158, 10, boxstyle="round,pad=0,rounding_size=1", ec=MAROON, fc="white", lw=1.3))
ax.text(85, y[0] - 2.2, "Measures that expose each term", ha="center", va="top", fontsize=12.9, color=MAROON, weight="bold", family=SERIF)
ax.text(85, y[0] - 5.4, r"$L$: real wages, bread and fuel price ratio.  $S$: outage hours, repair time, water days.  $V$: homicide rate.  $F$: hoarding, queues, rumor velocity.  $T$: surveys, tax compliance, reporting rates."
        "\n" + r"$M$: days of grain and fuel, fiscal reserve.  $\tau$: share of budgets, audits, and statistics published on time.  $v$: inspections completed.  $C_r$: agreements kept over ten years.",
        ha="center", va="top", fontsize=10.8, color=BLACK, family=SANS, linespacing=1.5)

fig.savefig("figures/peace_equations.png", dpi=300, facecolor="white")
fig.savefig("figures/peace_equations.pdf", facecolor="white")
Image.open("figures/peace_equations.png").convert("RGB").save("figures/peace_equations.jpg", quality=95)
print(y[0])
