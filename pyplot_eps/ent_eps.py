import numpy as np
import os
from mpEntropy import mpSystem
import matplotlib as mpl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
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
max_time = 3
inlay_min_time = 10
inlay_max_time = 100
inlay_log_min_time = 0
inlay_log_max_time = 3
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

loavgpercent = sysVar.plotLoAvgPerc  # percentage of time evolution to start averaging
loavgind = int(loavgpercent * sysVar.dataPoints)  # index to start at when calculating average and stddev
loavgtime = np.round(loavgpercent * (sysVar.deltaT * sysVar.steps * sysVar.plotTimeScale), 2)

# stuff for averaging
if sysVar.boolPlotAverages:
    print(' with averaging from Jt=%.2f' % loavgtime, end='')
fwidth = sysVar.plotSavgolFrame
ford = sysVar.plotSavgolOrder

ent_array = np.loadtxt('../data/entropy.txt')

# multiply step array with time scale
step_array = ent_array[:, 0] * sysVar.plotTimeScale
min_index = int(min_time / step_array[-1] * len(step_array))
max_index = int(max_time / step_array[-1] * len(step_array))
inlay_min_index = int(inlay_min_time / step_array[-1] * len(step_array))
inlay_max_index = int(inlay_max_time / step_array[-1] * len(step_array))
inlay_log_min_index = int(inlay_log_min_time / step_array[-1] * len(step_array))
inlay_log_max_index = int(inlay_log_max_time / step_array[-1] * len(step_array))
#### Complete system Entropy
if os.path.isfile('../data/total_entropy.txt'):
    totent_array = np.loadtxt('../data/total_entropy.txt')
    plt.plot(totent_array[min_index:max_index, 0] * sysVar.plotTimeScale, totent_array[min_index:max_index, 1] * 1e13,
             linewidth=0.6, color='r')

    plt.grid()
    plt.xlabel(r'$J\,t$')
    plt.ylabel(r'Total system entropy $/ 10^{-13}$')
    plt.tight_layout(padding)
    ###
    plt.savefig(pltfolder + 'entropy_total.eps', format='eps', dpi=1000)
    plt.clf()
    print('.', end='', flush=True)

### Subsystem Entropy
fldat = open(pltfolder + 'ent_fluctuation_N' + str(sysVar.N) + '.txt', 'w')
fldat.write('N_tot: %i\n' % sysVar.N)
avg = np.mean(ent_array[loavgind:, 1], dtype=np.float64)
stddev = np.std(ent_array[loavgind:, 1], dtype=np.float64)
fldat.write('ssent_average: %.16e\n' % avg)
fldat.write('ssent_stddev: %.16e\n' % stddev)
fldat.write('ssent_rel._fluctuation: %.16e\n' % (stddev / avg))
fldat.close()

plt.plot(step_array[min_index:max_index], ent_array[min_index:max_index, 1], color='r')
if sysVar.boolPlotAverages:
    tavg = savgol_filter(ent_array[:, 1], fwidth, ford)
    plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')
plt.xlabel(r'$J\,t$')
plt.ylabel(r'$S\textsubscript{sys}$')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'entropy_subsystem.eps', format='eps', dpi=1000)
plt.clf()

# Subsystem entropy with logarithmic inlay
plt.plot(step_array[min_index:max_index], ent_array[min_index:max_index, 1], color='r')
if sysVar.boolPlotAverages:
    tavg = savgol_filter(ent_array[:, 1], fwidth, ford)
    plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')
plt.xlabel(r'$J\,t$')
plt.ylabel(r'$S\textsubscript{sys}$')
a = plt.axes([.5, .35, .4, .4])
plt.semilogy(step_array[inlay_log_min_index:inlay_log_max_index],
             np.abs(avg - ent_array[inlay_log_min_index:inlay_log_max_index, 1]), color='r')
plt.ylabel(r'$|\,\overline{S}\textsubscript{sys} - S\textsubscript{sys}(t)|$')
plt.yticks([])
plt.savefig(pltfolder + 'entropy_subsystem_inlay_log.eps', format='eps', dpi=1000)
plt.clf()

# Subsystem entropy with inlay
plt.plot(step_array[min_index:max_index], ent_array[min_index:max_index, 1], color='r')
if sysVar.boolPlotAverages:
    tavg = savgol_filter(ent_array[:, 1], fwidth, ford)
    plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')
plt.xlabel(r'$J\,t$')
plt.ylabel(r'$S\textsubscript{sys}$')
a = plt.axes([.45, .35, .4, .4])
plt.plot(step_array[inlay_min_index:inlay_max_index], avg - ent_array[inlay_min_index:inlay_max_index, 1],
         linewidth=0.2, color='r')
plt.ylabel(r'$\overline{S}\textsubscript{sys} - S\textsubscript{sys}(t)$')
a.yaxis.tick_right()
tmp_ticks = list(a.get_xticks())
tmp_ticks.pop(0)
if tmp_ticks[-1] >= inlay_max_time:
    tmp_ticks.pop(-1)
a.set_xticks(tmp_ticks + [inlay_min_time])
plt.savefig(pltfolder + 'entropy_subsystem_inlay.eps', format='eps', dpi=1000)
plt.clf()
print('.', end='', flush=True)

# histogram of fluctuations
n, bins, patches = plt.hist(ent_array[loavgind:, 1] - avg, 51, normed=1, rwidth=0.8, align='mid')
(mu, sigma) = norm.fit(ent_array[loavgind:, 1] - avg)
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--')
mu_magnitude = np.floor(np.log10(np.abs(mu)))
mu /= np.power(10, mu_magnitude)
sigma_magnitude = np.floor(np.log10(sigma))
sigma /= np.power(10, sigma_magnitude)
plt.figtext(0.965, 0.80,
            '$\mu = %.2f \cdot 10^{%i}$\n$\sigma = %.2f \cdot 10^{%i}$' % (mu, mu_magnitude, sigma, sigma_magnitude),
            ha='right', va='bottom', multialignment="left")
plt.xlabel(r'$\Delta S_{sub}$')
plt.ylabel(r'PD')
plt.tight_layout(padding)
plt.savefig(pltfolder + 'entropy_subsystem_fluctuations.eps', format='eps', dpi=1000)
plt.clf()
print('.', end='', flush=True)

print(" done!")
