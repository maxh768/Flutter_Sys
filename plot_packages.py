# post process settings
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib

my_blue = "#4C72B0"
my_red = "#C54E52"
my_green = "#56A968"
my_brown = "#b4943e"
my_purple = "#684c6b"
my_orange = "#cc5500"

matplotlib.rcParams.update({"font.size": 20})
rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

# # Define some packages and commands in preamble
# rc(
#     "text.latex",
#     preamble=[
#         r"\usepackage{amsmath}",
#         r"\usepackage{amsfonts}",
#         r"\newcommand{\overbar}[1]{\mkern 1.5mu\overline{\mkern-1.5mu#1\mkern-1.5mu}\mkern 1.5mu}",
#     ],
# )


def plot_legend(ax, x1, x2, y, x_text, y_text, text, color, alpha, shape):

    xc = (x1 + x2) / 2
    ax.plot([x1, xc], [y, y], "-", color=color, alpha=alpha)
    ax.plot([xc, x2], [y, y], "-", color=color, alpha=alpha)
    ax.plot([xc, xc], [y, y], shape, color="w", markersize=15)
    ax.plot([xc, xc], [y, y], shape, color=color, markersize=10, alpha=alpha)

    ax.text(x_text, y_text, text, color=color, fontsize=20, alpha=1)

