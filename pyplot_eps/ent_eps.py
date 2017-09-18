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

ent_array = np.loadtxt('../data/entropy.txt')
# multiply step array with time scale
step_array = ent_array[:, 0] * sysVar.plotTimeScale

#### Complete system Entropy
if os.path.isfile('../data/total_entropy.txt'):
    totent_array = np.loadtxt('../data/total_entropy.txt')
    plt.plot(totent_array[:, 0] * sysVar.plotTimeScale, totent_array[:, 1] * 1e13, linewidth=0.6, color='r')

    plt.grid()
    plt.xlabel(r'$J\,t$')
    plt.ylabel(r'Total system entropy $/ 10^{-13}$')
    plt.tight_layout()
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

plt.plot(step_array[:], ent_array[:, 1], linewidth=0.3, color='r')
plt.grid()
if sysVar.boolPlotAverages:
    tavg = savgol_filter(ent_array[:, 1], fwidth, ford)
    plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')
plt.xlabel(r'$J\,t$')
plt.ylabel('Subsystem entropy')
plt.tight_layout()
plt.savefig(pltfolder + 'entropy_subsystem.eps', format='eps', dpi=1000)
plt.clf()


# Subsystem entropy with inlay
max_time = 1.5
max_ind = int(max_time / step_array[-1] * len(step_array))

plt.plot(step_array[:], ent_array[:, 1], linewidth=0.3, color='r')
if sysVar.boolPlotAverages:
    tavg = savgol_filter(ent_array[:, 1], fwidth, ford)
    plt.plot(step_array[loavgind:], tavg[loavgind:], linewidth=avgsize, linestyle=avgstyle, color='black')
plt.xlabel(r'$J\,t$')
plt.ylabel('Subsystem entropy')
a = plt.axes([.5, .3, .4, .4])
plt.plot(step_array[:max_ind], ent_array[:max_ind, 1], linewidth=0.3, color='r')
plt.savefig(pltfolder + 'entropy_subsystem_inlay.eps', format='eps', dpi=1000)
plt.clf()
print('.', end='', flush=True)

# histogram of fluctuations
n, bins, patches = plt.hist(ent_array[loavgind:, 1] - avg, 51, normed=1, rwidth=0.8, align='mid')
(mu, sigma) = norm.fit(ent_array[loavgind:, 1] - avg)
y = mlab.normpdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
mu_magnitude = np.floor(np.log10(np.abs(mu)))
mu /= np.power(10, mu_magnitude)
sigma_magnitude = np.floor(np.log10(sigma))
sigma /= np.power(10, sigma_magnitude)
plt.figtext(0.92, 0.82,
            '$\mu = %.2f \cdot 10^{%i}$\n$\sigma = %.2f \cdot 10^{%i}$' % (mu, mu_magnitude, sigma, sigma_magnitude),
            ha='right', va='bottom', multialignment="left", bbox=dict(fc="none"))
plt.xlabel(r'$S_{sub}$ fluctuation amplitude')
plt.ylabel(r'Probability density')
plt.tight_layout()
plt.savefig(pltfolder + 'entropy_subsystem_fluctuations.eps', format='eps', dpi=1000)
plt.clf()
print('.', end='', flush=True)

print(" done!")
