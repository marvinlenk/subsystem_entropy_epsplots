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

occ_array = np.loadtxt('../data/occupation.txt')
# multiply step array with time scale
step_array = occ_array[:, 0] * sysVar.plotTimeScale

### Single-level occupation numbers

for i in range(0, sysVar.m):
    plt.plot(step_array, occ_array[:, i + 1], label=r'$n_' + str(i) + '$', linewidth=0.5)
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
        plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')

plt.ylabel(r'Occupation number')
plt.xlabel(r'$J\,t$')
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.savefig(pltfolder + 'occupation_single.eps', format='eps', dpi=1000)
plt.clf()
print('.', end='', flush=True)

### Traced out (bath) occupation numbers
for i in sysVar.kRed:
    plt.plot(step_array, occ_array[:, i + 1], label=r'$n_' + str(i) + '$', linewidth=0.6)
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
        plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')

plt.ylabel(r'Occupation number')
plt.xlabel(r'$J\,t$')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.savefig(pltfolder + 'occupation_bath.eps', format='eps', dpi=1000)
plt.clf()
print('.', end='', flush=True)

### Leftover (system) occupation numbers

for i in np.arange(sysVar.m)[sysVar.mask]:
    plt.plot(step_array, occ_array[:, i + 1], label=r'$n_' + str(i) + '$', linewidth=0.6)
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
        plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')

plt.ylabel(r'Occupation number')
plt.xlabel(r'$J\,t$')
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.savefig(pltfolder + 'occupation_system.eps', format='eps', dpi=1000)
plt.clf()
print('.', end='', flush=True)

### Subsystems occupation numbers
# store fluctuations in a data
fldat = open(pltfolder + 'occ_fluctuation.txt', 'w')
fldat.write('N_tot: %i\n' % sysVar.N)
tmp = np.zeros(len(step_array))
for i in sysVar.kRed:
    tmp += occ_array[:, i + 1]
plt.plot(step_array, tmp, label="bath", linewidth=0.8, color='magenta')

if sysVar.boolPlotAverages:
    tavg = savgol_filter(tmp, fwidth, ford)
    plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')

avg = np.mean(tmp[loavgind:], dtype=np.float64)
stddev = np.std(tmp[loavgind:], dtype=np.float64)
fldat.write('bath_average: %.16e\n' % avg)
fldat.write('bath_stddev: %.16e\n' % stddev)
fldat.write('bath_rel._fluctuation: %.16e\n' % (stddev / avg))

tmp.fill(0)
for i in np.arange(sysVar.m)[sysVar.mask]:
    tmp += occ_array[:, i + 1]
plt.plot(step_array, tmp, label="system", linewidth=0.8, color='darkgreen')

if sysVar.boolPlotAverages:
    tavg = savgol_filter(tmp, fwidth, ford)
    plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')

avg = np.mean(tmp[loavgind:], dtype=np.float64)
stddev = np.std(tmp[loavgind:], dtype=np.float64)
fldat.write('system_average: %.16e\n' % avg)
fldat.write('system_stddev: %.16e\n' % stddev)
fldat.write('system_rel._fluctuation: %.16e\n' % (stddev / avg))

for i in range(sysVar.m):
    avg = np.mean(occ_array[loavgind:, i + 1], dtype=np.float64)
    stddev = np.std(occ_array[loavgind:, i + 1], dtype=np.float64)
    fldat.write('n%i_average: %.16e\n' % (i, avg))
    fldat.write('n%i_stddev: %.16e\n' % (i, stddev))
    fldat.write('n%i_rel._fluctuation: %.16e\n' % (i, (stddev / avg)))
fldat.close()

plt.ylabel(r'Occupation number')
plt.xlabel(r'$J\,t$')
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.savefig(pltfolder + 'occupation_systems.eps', format='eps', dpi=1000)
plt.clf()
print('.', end='', flush=True)

### occupation number in levels against level index

occavg = np.loadtxt(pltfolder + 'occ_fluctuation.txt', usecols=(1,))
plt.xlim(-0.1, sysVar.m - 0.9)
for l in range(0, sysVar.m):
    plt.errorbar(l, occavg[int(7 + 3 * l)] / sysVar.N, xerr=None, yerr=occavg[int(8 + 3 * l)] / sysVar.N,
                 marker='o', color=cm.Set1(0))
plt.ylabel(r'Relative level occupation')
plt.xlabel(r'Level index')
plt.grid()
plt.tight_layout()
plt.savefig(pltfolder + 'occupation_distribution.eps', format='eps', dpi=1000)
plt.clf()

plt.xlim(np.min(np.arange(sysVar.m)[sysVar.mask]) - 0.1, np.max(np.arange(sysVar.m)[sysVar.mask]) + 0.1)
for l in np.arange(sysVar.m)[sysVar.mask]:
    plt.errorbar(l, occavg[int(7 + 3 * l)] / sysVar.N, xerr=None, yerr=occavg[int(8 + 3 * l)] / sysVar.N,
                 marker='o', color=cm.Set1(0))
plt.ylabel(r'Relative level occupation')
plt.xlabel(r'Level index')
plt.grid()
plt.tight_layout()
plt.savefig(pltfolder + 'occupation_distribution_sys.eps', format='eps', dpi=1000)
plt.clf()

plt.xlim(np.min(sysVar.kRed) - 0.1, np.max(sysVar.kRed) + 0.1)
for l in sysVar.kRed:
    plt.errorbar(l, occavg[int(7 + 3 * l)] / sysVar.N, xerr=None, yerr=occavg[int(8 + 3 * l)] / sysVar.N,
                 marker='o', color=cm.Set1(0))
plt.ylabel(r'Relative level occupation')
plt.xlabel(r'Level index')
plt.grid()
plt.tight_layout()
plt.savefig(pltfolder + 'occupation_distribution_bath.eps', format='eps', dpi=1000)
plt.clf()

print(" done!")
