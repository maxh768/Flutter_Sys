import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as matplotlib

# color
my_blue = '#4C72B0'
my_red = '#C54E52'
my_green = '#56A968' 
my_brown = '#b4943e'
my_purple = '#684c6b'
my_orange = '#cc5500'

# font
matplotlib.rcParams.update({'font.size': 20})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)