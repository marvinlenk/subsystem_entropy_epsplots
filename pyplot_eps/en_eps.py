import numpy as np
import os
from mpEntropy import mpSystem
import matplotlib as mpl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# This is a workaround until scipy fixes the issue
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

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

loavgpercent = sysVar.plotLoAvgPerc  # percentage of time evolution to start averaging
loavgind = int(loavgpercent * sysVar.dataPoints)  # index to start at when calculating average and stddev
loavgtime = np.round(loavgpercent * (sysVar.deltaT * sysVar.steps * sysVar.plotTimeScale), 2)

if sysVar.boolPlotAverages:
    print(' with averaging from Jt=%.2f' % loavgtime, end='')
fwidth = sysVar.plotSavgolFrame
ford = sysVar.plotSavgolOrder
params = {
    'legend.fontsize': sysVar.plotLegendSize,
    'font.size': sysVar.plotFontSize,
    'mathtext.default': 'rm'  # see http://matplotlib.org/users/customizing.html
}
plt.rcParams['agg.path.chunksize'] = 0
plt.rcParams.update(params)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

bool_eigenvalues = False  # plot eigenvalues and decomposition?
if os.path.isfile('../data/hamiltonian_eigvals.txt'):
    bool_eigenvalues = True
    energy_array = np.loadtxt('../data/hamiltonian_eigvals.txt')

bool_total = False  # plot total energy?
if os.path.isfile('../data/total_energy.txt'):
    bool_total = True
    total_energy_array = np.loadtxt('../data/total_energy.txt')
elif os.path.isfile('../data/energy.txt'):  # old file path
    bool_total = True
    total_energy_array = np.loadtxt('../data/energy.txt')

# eigenvalues and spectral decomposition
if bool_eigenvalues:
    fig, ax1 = plt.subplots()
    energy_markersize = 0.7
    energy_barsize = 0.06
    if sysVar.dim != 1:
        energy_markersize *= (2.0 / np.log10(sysVar.dim))
        energy_barsize *= (4.0 / np.log10(sysVar.dim))
    ax1.plot(energy_array[:, 0], energy_array[:, 1], linestyle='none', marker='o', ms=energy_markersize, color='blue')
    ax1.set_ylabel(r'Energy / J')
    ax1.set_xlabel(r'Eigenvalue index $n$')
    plt.grid(False)
    ax1.set_xlim(xmin=-(len(energy_array[:, 0]) * (5.0 / 100)))
    if np.shape(energy_array)[0] > 2:
        ax2 = ax1.twinx()
        ax2.bar(energy_array[:, 0], energy_array[:, 2], alpha=0.8, color='red', width=energy_barsize, align='center')
        ax2.set_ylabel(r'$|c_n|^2$')
        # inlay with small region around maximum
        ax_inlay = plt.axes([0.62, 0.6, 0.28, 0.28])
        index_range = np.floor(sysVar.dim / 200)
        index_maximum = np.amax(energy_array[:, 2])
        index_lo = index_maximum - index_range
        index_hi = index_maximum + index_range
        decomp_max = np.max(energy_array[index_lo:index_hi, 2])
        decomp_min = np.min(energy_array[index_lo:index_hi, 2])
        ax_inlay.bar(energy_array[index_lo:index_hi, 0], energy_array[index_lo:index_hi, 2], color='red', width=energy_barsize * 10, align='center')
        ax_inlay.set_xticks([energy_array[index_lo, 0], energy_array[index_hi, 0]])
        ax_inlay.set_yticks([decomp_min,decomp_max])
    plt.tight_layout()
    ###
    plt.savefig(pltfolder + 'energy_eigenvalues.eps', format='eps', dpi=1000)
    plt.clf()
    print('.', end='', flush=True)

    # Eigenvalue decomposition with energy x-axis
    plt.bar(energy_array[:, 1], energy_array[:, 2], alpha=0.8, color='red', width=energy_barsize, align='center')
    plt.xlabel(r'Energy / J')
    plt.ylabel(r'$|c_E|^2$')
    plt.grid(False)
    plt.xlim(xmin=-(np.abs(energy_array[0, 1] - energy_array[-1, 1]) * (5.0 / 100)))
    plt.tight_layout()
    ###
    plt.savefig(pltfolder + 'energy_decomposition.eps', format='eps', dpi=1000)
    plt.clf()
    print('.', end='', flush=True)

# Total energy
if bool_total:
    en0 = total_energy_array[0, 1]
    total_energy_array[:, 1] -= en0
    en0_magnitude = np.floor(np.log10(np.abs(en0)))
    en0 /= np.power(10, en0_magnitude)
    magnitude = np.floor(np.log10(np.max(np.abs(total_energy_array[:, 1]))))
    plt.plot(total_energy_array[:, 0] * sysVar.plotTimeScale, total_energy_array[:, 1] / (np.power(10, magnitude)),
             linewidth=0.6)
    plt.figtext(0.9, 0.85, r'$E_0 / J = %.2f \cdot 10^{%i}$' % (en0, en0_magnitude), horizontalalignment='right',
                verticalalignment='bottom')
    plt.ylabel(r'$E_{tot} - E_0 / (J \cdot 10^{%i})$' % magnitude)
    plt.xlabel(r'$J\,t$')
    plt.grid(False)
    plt.tight_layout()
    ###
    plt.savefig(pltfolder + 'energy_total.eps', format='eps', dpi=1000)
    plt.clf()
    print('.', end='', flush=True)

print(" done!")
