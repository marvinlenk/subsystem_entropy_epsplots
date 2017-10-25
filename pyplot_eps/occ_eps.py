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
# minimum and maximum times to plot
min_time = 0
max_time = 30
inlay_min_time = 10
inlay_max_time = 100
inlay_log_min_time = 0
inlay_log_max_time = 20
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
    'lines.linewidth': 0.7,
    'figure.figsize': fig_size,
    'legend.frameon': False,
    'legend.loc': 'best',
    'mathtext.default': 'rm'  # see http://matplotlib.org/users/customizing.html
}
plt.rcParams['agg.path.chunksize'] = 0
plt.rcParams.update(params)
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})

loavgpercent = sysVar.plotLoAvgPerc  # percentage of time evolution to start averaging
loavgind = int(loavgpercent * sysVar.dataPoints)  # index to start at when calculating average and stddev
loavgtime = np.round(loavgpercent * (sysVar.deltaT * sysVar.steps * sysVar.plotTimeScale), 2)

if sysVar.boolPlotAverages:
    print(' with averaging from Jt=%.2f' % loavgtime, end='')

fwidth = sysVar.plotSavgolFrame
ford = sysVar.plotSavgolOrder

occ_array = np.loadtxt('../data/occupation.txt')
# multiply step array with time scale
step_array = occ_array[:, 0] * sysVar.plotTimeScale
min_index = int(min_time / step_array[-1] * len(step_array))
max_index = int(max_time / step_array[-1] * len(step_array))
inlay_min_index = int(inlay_min_time / step_array[-1] * len(step_array))
inlay_max_index = int(inlay_max_time / step_array[-1] * len(step_array))
inlay_log_min_index = int(inlay_log_min_time / step_array[-1] * len(step_array))
inlay_log_max_index = int(inlay_log_max_time / step_array[-1] * len(step_array))

### Single-level occupation numbers

for i in range(0, sysVar.m):
    plt.plot(step_array[min_index:max_index], occ_array[min_index:max_index, i + 1], label=r'$n_' + str(i) + '$')
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
        plt.plot(step_array[loavgind:], tavg[loavgind:], linestyle=avgstyle, color='black')

plt.ylabel(r'$\langle n \rangle$')
plt.xlabel(r'$J\,t$')
plt.legend(loc='upper right')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'occupation_single.eps', format='eps', dpi=1000)
plt.clf()

### Single-level occupation number difference to mean

for i in range(0, sysVar.m):
    plt.semilogy(step_array[min_index:max_index], np.abs(occ_array[min_index:max_index, i + 1] - np.average(occ_array[loavgind:, i + 1]))*np.exp(-i*2.1))

plt.ylabel(r'$| \bar{\langle n \rangle} - \langle n \rangle|$')
plt.xlabel(r'$J\,t$')
plt.ylim(ymin=1e-6, ymax=5e0)
plt.yticks([])
plt.tight_layout(padding)
plt.savefig(pltfolder + 'occupation_single_difftomean.eps', format='eps', dpi=1000)
plt.clf()

# with inlay
f = plt.figure(1)
a1 = plt.subplot(1, 1, 1)
for i in range(0, sysVar.m):
    a1.plot(step_array[min_index:max_index], occ_array[min_index:max_index, i + 1], label=r'$n_' + str(i) + '$')
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
        plt.plot(step_array[loavgind:], tavg[loavgind:], linestyle=avgstyle, color='black')
    a2 = plt.axes([.3, .575, .4, .35])
    a2.plot(step_array[inlay_min_index:inlay_max_index], occ_array[inlay_min_index:inlay_max_index, i + 1],
            linewidth=0.2)
    a2.yaxis.tick_right()

tmp_ticks = list(a2.get_xticks())
tmp_ticks.pop(0)
tmp_ticks.pop(0)
tmp_ticks.pop(1)
if tmp_ticks[-1] >= inlay_max_time:
    tmp_ticks.pop(-1)
    if tmp_ticks[-1] < inlay_max_time or len(tmp_ticks) == 0:
        tmp_ticks = tmp_ticks + [inlay_max_time]
a2.set_xticks(tmp_ticks + [inlay_min_time])
a1.set_ylabel(r'$\langle n \rangle$')
a1.set_xlabel(r'$J\,t$')
a1.legend(loc='upper right')
f.savefig(pltfolder + 'occupation_single_inlay.eps', format='eps', dpi=1000)
f.clf()
print('.', end='', flush=True)

### Traced out (bath) occupation numbers
for i in sysVar.kRed:
    plt.plot(step_array[min_index:max_index], occ_array[min_index:max_index, i + 1], label=r'$n_' + str(i) + '$',
             color='C%i' % i)
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
        plt.plot(step_array[loavgind:], tavg[loavgind:], linestyle=avgstyle, color='black')

plt.ylabel(r'$\langle n \rangle$')
plt.xlabel(r'$J\,t$')
plt.legend(loc='upper right')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'occupation_bath.eps', format='eps', dpi=1000)
plt.clf()

# with inlay
f = plt.figure(1)
a1 = plt.subplot(1, 1, 1)
a1.set_ylabel(r'$\langle n \rangle$')
a1.set_xlabel(r'$J\,t$')
plt.tight_layout(padding)
for i in sysVar.kRed:
    a1.plot(step_array[min_index:max_index], occ_array[min_index:max_index, i + 1], label=r'$n_' + str(i) + '$',
            color='C%i' % i)
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
        plt.plot(step_array[loavgind:], tavg[loavgind:], linestyle=avgstyle, color='black')
    a2 = plt.axes([.27, .575, .4, .35])
    a2.plot(step_array[inlay_min_index:inlay_max_index], occ_array[inlay_min_index:inlay_max_index, i + 1],
            linewidth=0.2, color='C%i' % i)
    a2.yaxis.tick_right()

tmp_ticks = list(a2.get_xticks())
tmp_ticks.pop(0)
tmp_ticks.pop(0)
tmp_ticks.pop(1)
if tmp_ticks[-1] >= inlay_max_time:
    tmp_ticks.pop(-1)
    if tmp_ticks[-1] < inlay_max_time or len(tmp_ticks) == 0:
        tmp_ticks = tmp_ticks + [inlay_max_time]
a2.set_xticks(tmp_ticks + [inlay_min_time])
a1.legend(loc='upper right')
f.savefig(pltfolder + 'occupation_bath_inlay.eps', format='eps', dpi=1000)
f.clf()
print('.', end='', flush=True)

### Leftover (system) occupation numbers

for i in np.arange(sysVar.m)[sysVar.mask]:
    plt.plot(step_array[min_index:max_index], occ_array[min_index:max_index, i + 1], label=r'$n_' + str(i) + '$',
             color='C%i' % i)
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
        plt.plot(step_array[loavgind:], tavg[loavgind:], linestyle=avgstyle, color='black')

plt.ylabel(r'$\langle n \rangle$')
plt.xlabel(r'$J\,t$')
plt.legend(loc='lower right')
plt.tight_layout(padding+0.4)
plt.savefig(pltfolder + 'occupation_system.eps', format='eps', dpi=1000)
plt.clf()

# with inlay
f = plt.figure(1)
a1 = plt.subplot(1, 1, 1)
for i in np.arange(sysVar.m)[sysVar.mask]:
    a1.plot(step_array[min_index:max_index], occ_array[min_index:max_index, i + 1], label=r'$n_' + str(i) + '$',
            color='C%i' % i)
    if sysVar.boolPlotAverages:
        tavg = savgol_filter(occ_array[:, i + 1], fwidth, ford)
        plt.plot(step_array[loavgind:], tavg[loavgind:], linestyle=avgstyle, color='black')
    a2 = plt.axes([.3, .275, .4, .35])
    a2.plot(step_array[inlay_min_index:inlay_max_index], occ_array[inlay_min_index:inlay_max_index, i + 1],
            linewidth=0.2, color='C%i' % i)
    a2.yaxis.tick_right()

tmp_ticks = list(a2.get_xticks())
tmp_ticks.pop(0)
tmp_ticks.pop(0)
tmp_ticks.pop(1)
if tmp_ticks[-1] >= inlay_max_time:
    tmp_ticks.pop(-1)
    if tmp_ticks[-1] < inlay_max_time or len(tmp_ticks) == 0:
        tmp_ticks = tmp_ticks + [inlay_max_time]
a2.set_xticks(tmp_ticks + [inlay_min_time])
a1.set_ylabel(r'$\langle n \rangle$')
a1.set_xlabel(r'$J\,t$')
a1.legend(loc='lower right')
f.savefig(pltfolder + 'occupation_system_inlay.eps', format='eps', dpi=1000)
f.clf()
print('.', end='', flush=True)

### Subsystems occupation numbers
# store fluctuations in a data
fldat = open(pltfolder + 'occ_fluctuation_N' + str(sysVar.N) + '.txt', 'w')
fldat.write('N_tot: %i\n' % sysVar.N)
tmp = np.zeros(len(step_array))
for i in sysVar.kRed:
    tmp += occ_array[:, i + 1]
plt.plot(step_array[min_index:max_index], tmp[min_index:max_index], label="bath", color='magenta')

if sysVar.boolPlotAverages:
    tavg = savgol_filter(tmp, fwidth, ford)
    plt.plot(step_array[loavgind:], tavg[loavgind:], linestyle=avgstyle, color='black')

avg = np.mean(tmp[loavgind:], dtype=np.float64)
stddev = np.std(tmp[loavgind:], dtype=np.float64)
fldat.write('bath_average: %.16e\n' % avg)
fldat.write('bath_stddev: %.16e\n' % stddev)
fldat.write('bath_rel._fluctuation: %.16e\n' % (stddev / avg))

tmp.fill(0)
for i in np.arange(sysVar.m)[sysVar.mask]:
    tmp += occ_array[:, i + 1]
plt.plot(step_array[min_index:max_index], tmp[min_index:max_index], label="system", color='darkgreen')

if sysVar.boolPlotAverages:
    tavg = savgol_filter(tmp, fwidth, ford)
    plt.plot(step_array[loavgind:], tavg[loavgind:], linestyle=avgstyle, color='black')

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

plt.ylabel(r'$\langle n \rangle$')
plt.xlabel(r'$J\,t$')
plt.legend(loc='upper right')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'occupation_systems.eps', format='eps', dpi=1000)
plt.clf()
print('.', end='', flush=True)

### occupation number in levels against level index

occavg = np.loadtxt(pltfolder + 'occ_fluctuation_N' + str(sysVar.N) + '.txt', usecols=(1,))
plt.xlim(-0.1, sysVar.m - 0.9)
for l in range(0, sysVar.m):
    plt.errorbar(l, occavg[int(7 + 3 * l)] / sysVar.N, xerr=None, yerr=occavg[int(8 + 3 * l)] / sysVar.N,
                 marker='o', color=cm.Set1(0))
plt.ylabel(r'Relative level occupation')
plt.xlabel(r'Level index')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'occupation_distribution.eps', format='eps', dpi=1000)
plt.clf()

plt.xlim(np.min(np.arange(sysVar.m)[sysVar.mask]) - 0.1, np.max(np.arange(sysVar.m)[sysVar.mask]) + 0.1)
for l in np.arange(sysVar.m)[sysVar.mask]:
    plt.errorbar(l, occavg[int(7 + 3 * l)] / sysVar.N, xerr=None, yerr=occavg[int(8 + 3 * l)] / sysVar.N,
                 marker='o', color=cm.Set1(0))
plt.ylabel(r'Relative level occupation')
plt.xlabel(r'Level index')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'occupation_distribution_sys.eps', format='eps', dpi=1000)
plt.clf()

plt.xlim(np.min(sysVar.kRed) - 0.1, np.max(sysVar.kRed) + 0.1)
for l in sysVar.kRed:
    plt.errorbar(l, occavg[int(7 + 3 * l)] / sysVar.N, xerr=None, yerr=occavg[int(8 + 3 * l)] / sysVar.N,
                 marker='o', color=cm.Set1(0))
plt.ylabel(r'Relative level occupation')
plt.xlabel(r'Level index')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'occupation_distribution_bath.eps', format='eps', dpi=1000)
plt.clf()

print(" done!")
