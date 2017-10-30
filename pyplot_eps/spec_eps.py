import numpy as np
import os
import dft
from mpEntropy import mpSystem
import matplotlib as mpl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
# This is a workaround until scipy fixes the issue
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def lorentzian(x, x0, s, a):
    return a/np.pi * s / (s**2 + (x-x0)**2)

# Load sysVar
sysVar = mpSystem("../interact_0.ini", plotOnly=True)

# Create plot folder
pltfolder = "./epsplots/"
if not os.path.exists(pltfolder):
    os.mkdir(pltfolder)

print("Plotting", end='')
mpl.use('Agg')
# styles and stuff
avgstyle = 'dashed'
avgsize = 0.6
expectstyle = 'solid'
expectsize = 1
legend_size = 10
font_size = 10
# https://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/LaTeX_Examples.html
fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inches
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
# padding in units of fontsize
padding = 0.32

params = {
    'axes.labelsize': 10,
    'font.size': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'lines.linewidth': 1,
    'figure.figsize': fig_size,
    'legend.frameon': False,
    'legend.loc': 'best',
    'mathtext.default': 'rm'  # see http://matplotlib.org/users/customizing.html
}
plt.rcParams['agg.path.chunksize'] = 0
plt.rcParams.update(params)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

greendat = np.loadtxt('../data/green.txt')
greendat[:, 0] = greendat[:, 0] * sysVar.plotTimeScale

spec = []
for i in range(0, sysVar.m):
    step_array = greendat[:, 0] * sysVar.plotTimeScale
    plt.title(r'two time Green function of level $%i$' % i)
    ind = 2 * i + 1
    plt.plot(step_array, greendat[:, ind], lw=0.2, color='red', label='real')
    plt.plot(step_array, greendat[:, ind + 1], lw=0.2, color='blue', label='imaginary')
    plt.ylabel(r'$G^R(\tau)$')
    plt.xlabel(r'$J\,\tau$')
    plt.legend()
    plt.tight_layout(padding)
    plt.savefig(pltfolder + 'green_%i.eps' % i, format='eps', dpi=1000)
    plt.clf()
    ###
    plt.title(r'Spectral function of level $%i$' % i)
    if not os.path.isfile('../data/spectral_%i.txt' % i) or True:
        green_ret = greendat[:, ind] + 1j * greendat[:, ind + 1]
        if i == 0:
            sample_spacing = (greendat[-1, 0] - greendat[0, 0]) / (len(green_ret) - 1)
        green_ret_freq = dft.rearrange(dft.dft(green_ret, 1.0 / sample_spacing))
        spec_tmp = np.column_stack((green_ret_freq[:, 0].real, -2 * green_ret_freq[:, 1].imag))
        print(np.shape(spec_tmp))
        np.savetxt('../data/spectral_%i.txt' % i, spec_tmp, "%16e", delimiter=' ')
    else:
        spec_tmp = np.loadtxt('../data/spectral_%i.txt' % i)
    if i == 0:
        spec_total = spec_tmp[:]
    else:
        spec_total[:, 1] += spec_tmp[:, 1]
    spec.append(spec_tmp)
    plt.plot(spec_tmp[:, 0], spec_tmp[:, 1], color='red', lw=0.3)
    plt.xlim(xmin=0)
    plt.minorticks_on()
    plt.ylabel(r'$A$')
    plt.xlabel(r'$\omega / J$')
    plt.tight_layout(padding)
    plt.savefig(pltfolder + 'spectral_%i.eps' % i, format='eps', dpi=1000)
    plt.clf()
plt.title(r'Spectral function')
plt.plot(spec_total[:, 0], spec_total[:, 1], color='red', lw=0.5)
plt.ylabel(r'$A$')
plt.xlabel(r'$\omega / J$')
plt.xlim([0, 70])
plt.tight_layout(padding)
plt.savefig(pltfolder + 'spec_trace.eps', format='eps', dpi=1000)
plt.clf()

plt.plot(spec_total[:, 0], np.abs(spec_total[:, 1]), color='red', lw=0.5)
plt.ylabel(r'$|A|$')
plt.xlabel(r'$\omega / J$')
plt.xlim([0, 70])
plt.tight_layout(padding)
plt.savefig(pltfolder + 'spec_trace_modulus.eps', format='eps', dpi=1000)
plt.clf()

ranges = np.array([[0, 22], [20, 31], [29, 41], [38, 50], [50, 80]])  # lo, hi, x0
inits = np.array([[17, 1, 50], [27, 2, 60], [34, 1, 60], [43, 3, 30], [60, 3, 30]])
params = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
for i in range(0, len(ranges)):
    loind = np.argmin(np.abs(spec_total[:, 0] - ranges[i, 0]))
    hiind = np.argmin(np.abs(spec_total[:, 0] - ranges[i, 1]))
    params[i], covar = curve_fit(lorentzian, spec_total[:, 0][loind:hiind], np.abs(spec_total[:, 1])[loind:hiind],
                                 inits[i])
print('\n', params)
print('.', end='', flush=True)

plt.plot(spec_total[:, 0], np.abs(spec_total[:, 1]), color='red', lw=0.5)
plt.ylabel(r'$|A|$')
plt.xlabel(r'$\omega / J$')
plt.xlim([0, 70])
for i in range(0, len(ranges)):
    plt.plot(spec_total[:, 0], lorentzian(spec_total[:, 0], params[i, 0], params[i, 1], params[i, 2]), color="black")

plt.tight_layout(padding)
plt.savefig(pltfolder + 'spec_trace_modulus_fits.eps', format='eps', dpi=1000)
plt.clf()



loavgpercent = sysVar.plotLoAvgPerc  # percentage of time evolution to start averaging
loavgind = int(loavgpercent * sysVar.dataPoints)  # index to start at when calculating average and stddev

occ_array = np.loadtxt('../data/occupation.txt')

avg = np.zeros(sysVar.m)
for i in range(sysVar.m):
    avg[i] = np.mean(occ_array[loavgind:, i + 1], dtype=np.float64)
energies = params[:, 0]

relocc = np.column_stack((energies, avg))
plt.plot(relocc[:, 0], np.abs(relocc[:, 1]), color='red', ls="none", marker=".")
plt.ylabel(r'$n_i$')
plt.xlabel(r'$\omega / J$')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'occ_average_renormalized.eps', format='eps', dpi=1000)
plt.clf()


def occno(freq, temp, mu):
    return 1. / (np.exp((freq - mu) / temp) - 1.0)


def bestat(args):
    temp = args[0]
    mu = args[1]
    ret = []
    for i in range(0, sysVar.m):
        ret.append(occno(relocc[i, 0], temp, mu))
    return np.array(ret)


def bestatd(args):
    temp = args[0]
    mu = args[1]
    ret = []
    for i in range(0, sysVar.m):
        ret.append(occno(relocc[i, 0], temp, mu) - relocc[i, 1])
    return np.array(ret)

strt = np.array([-100, -100])
bnds = np.array([[-100, -500], [10000, relocc[0, 0]]])
rgs = least_squares(bestatd, x0=strt, bounds=bnds, loss='soft_l1')
print(rgs)
print(rgs.x)
print(bestat(rgs.x))
print(relocc[:, 1])

# occupation number in levels against renormalized energy
xvals = np.linspace(relocc[0, 0], relocc[-1, 0], 1e3)
yvals = occno(xvals, rgs.x[0], rgs.x[1]) / sysVar.N
plt.plot(relocc[:, 0], np.abs(relocc[:, 1]/sysVar.N), color='red', ls="none", marker=".")
plt.plot(xvals, yvals, color='blue', lw=0.4)
plt.ylabel(r'$n_i / N$')
plt.xlabel(r'$\omega$')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'occ_average_renormalized_boseeinstein.eps', format='eps', dpi=1000)
plt.clf()