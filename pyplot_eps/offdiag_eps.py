import numpy as np
import os
from mpEntropy import mpSystem
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
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
max_time = 10
log_min_time = 0
log_max_time = 15
inlay_min_time = 00
inlay_max_time = 10
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

# ## occupation number operator offdiagonals
if os.path.isfile('../data/offdiagocc.txt') or os.path.isfile('../data/offdiagonal.txt'):
    # if old path - load old data
    if os.path.isfile('../data/offdiagonal.txt'):
        offdiagocc = np.loadtxt('../data/offdiagonal.txt')
    else:
        offdiagocc = np.loadtxt('../data/offdiagocc.txt')
    step_array = offdiagocc[:, 0] * sysVar.plotTimeScale
    # multiply step array with time scale
    min_index = int(min_time / step_array[-1] * len(step_array))
    max_index = int(max_time / step_array[-1] * len(step_array))
    inlay_min_index = int(inlay_min_time / step_array[-1] * len(step_array))
    inlay_max_index = int(inlay_max_time / step_array[-1] * len(step_array))
    inlay_log_min_index = int(inlay_log_min_time / step_array[-1] * len(step_array))
    inlay_log_max_index = int(inlay_log_max_time / step_array[-1] * len(step_array))
    log_min_index = int(log_min_time / step_array[-1] * len(step_array))
    log_max_index = int(log_max_time / step_array[-1] * len(step_array))

    for i in range(0, sysVar.m):
        plt.plot(step_array[min_index:max_index], offdiagocc[min_index:max_index, i + 1],
                 label=r'$n_' + str(i) + '$', linewidth=0.6)
    plt.ylabel(r'$\sum\limits_{k\neq l} \langle E_k | n | E_l \rangle c_k^\ast c_l$')
    plt.xlabel(r'$J\,t$')
    plt.legend()
    plt.tight_layout(padding-0.2)
    plt.gca().yaxis.set_label_coords(-0.09, 0.5)
    ###
    plt.savefig(pltfolder + 'offdiag_occupations.eps', format='eps', dpi=1000)
    plt.clf()

    #for i in range(0, sysVar.m):
    for i in [0,2,4]:
        plt.semilogy(step_array[log_min_index:log_max_index], np.abs(offdiagocc[log_min_index:log_max_index, i + 1]),
                 label=r'$n_' + str(i) + '$', linewidth=0.6)
    plt.ylim(ymin=2e-2, ymax=9e0)
    plt.legend()
    plt.ylabel(r'$| \sum\limits_{k\neq l} \langle E_k | n | E_l \rangle c_k^\ast c_l |$')
    plt.xlabel(r'$J\,t$')
    plt.tight_layout(padding)
    ###
    plt.savefig(pltfolder + 'offdiag_occupations_semilog.eps', format='eps', dpi=1000)
    plt.clf()
    dt = offdiagocc[1, 0] - offdiagocc[0, 0]
    nrm = offdiagocc[:, 0] / dt
    nrm[1:] = 1 / nrm[1:]
    for i in range(0, sysVar.m):
        # ##### only sum (subsystem-thermalization)
        plt.ylabel(r'$\sum\limits_{n\neq m} n^{%i}_{n,m}$' % i)
        plt.xlabel(r'$J\,t$')
        plt.plot(offdiagocc[min_index:max_index, 0], offdiagocc[min_index:max_index, i + 1])
        plt.tight_layout(padding + 0.2)
        # ##inlay with the whole deal
        a = plt.axes([0.62, 0.6, 0.28, 0.28])
        a.plot(offdiagocc[inlay_min_index:inlay_max_index, 0], offdiagocc[inlay_min_index:inlay_max_index, i + 1])
        a.set_xticks([])
        a.set_yticks([])
        ###
        plt.savefig(pltfolder + 'offdiag_occupation_%i.eps' % i, format='eps', dpi=1000)
        plt.clf()

        plt.ylabel(r'$\sum\limits_{k\neq l} n_{E_k,E_l}$')
        plt.semilogy(offdiagocc[log_min_index:log_max_index, 0], np.abs(offdiagocc[log_min_index:log_max_index, i + 1]))
        plt.ylim(ymin=1e-2)
        plt.tight_layout(padding)
        ###
        plt.savefig(pltfolder + 'offdiag_occupation_%i_semilog.eps' % i, format='eps', dpi=1000)
        plt.clf()

        # ##### average (eigenstate-thermalization)
        f, (ax1, ax2) = plt.subplots(2, sharex='none', sharey='none')
        tmp = cumtrapz(offdiagocc[:, i + 1], offdiagocc[:, 0], initial=offdiagocc[0, i + 1])
        tmp = np.multiply(tmp, nrm)
        f.text(0.07, 0.5,
               r'$\frac{1}{t} \int\limits_{0}^{t} \mathop{dt^\prime} \sum\limits_{n\neq m} n_{n,m}(t^\prime)$',
               ha='center', va='center', rotation='vertical')
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax1.plot(offdiagocc[min_index:max_index, 0], tmp[min_index:max_index])

        if min_index == 0:
            min_index_opt = 1
        else:
            min_index_opt = min_index
        ax2.plot(1/offdiagocc[min_index_opt:max_index, 0], np.abs(tmp[min_index_opt:max_index]))
        # ax2.set_ylim(bottom=1e-4)

        plt.tight_layout(padding)
        ###
        plt.savefig(pltfolder + 'offdiag_occupation_%i_eth.eps' % i, format='eps', dpi=1000)
        plt.clf()

    print('.', end='', flush=True)

# ## density matrix offdiagonals
if os.path.isfile('../data/offdiagdens.txt'):
    offdiagdens = np.loadtxt('../data/offdiagdens.txt')
    step_array = offdiagdens[:, 0] * sysVar.plotTimeScale
    # multiply step array with time scale
    min_index = int(min_time / step_array[-1] * len(step_array))
    max_index = int(max_time / step_array[-1] * len(step_array))
    inlay_min_index = int(inlay_min_time / step_array[-1] * len(step_array))
    inlay_max_index = int(inlay_max_time / step_array[-1] * len(step_array))
    inlay_log_min_index = int(inlay_log_min_time / step_array[-1] * len(step_array))
    inlay_log_max_index = int(inlay_log_max_time / step_array[-1] * len(step_array))

    plt.plot(step_array[min_index:max_index], offdiagdens[min_index:max_index, 1])
    plt.ylabel(r'Sum of off diagonals (dens. mat.)')
    plt.xlabel(r'$J\,t$')
    plt.tight_layout(padding)
    ###
    plt.savefig(pltfolder + 'offdiag_densitymatrix.eps', format='eps', dpi=1000)
    plt.clf()
    print('\n densavg = %f' % np.average(offdiagdens[loavgind:, 1]))

# ## reduced density matrix offdiagonals
if os.path.isfile('../data/offdiagdensred.txt'):
    offdiagdensred = np.loadtxt('../data/offdiagdensred.txt')
    step_array = offdiagdensred[:, 0] * sysVar.plotTimeScale
    # multiply step array with time scale
    min_index = int(min_time / step_array[-1] * len(step_array))
    max_index = int(max_time / step_array[-1] * len(step_array))
    inlay_min_index = int(inlay_min_time / step_array[-1] * len(step_array))
    inlay_max_index = int(inlay_max_time / step_array[-1] * len(step_array))
    inlay_log_min_index = int(inlay_log_min_time / step_array[-1] * len(step_array))
    inlay_log_max_index = int(inlay_log_max_time / step_array[-1] * len(step_array))

    plt.plot(step_array[min_index:max_index], offdiagdensred[min_index:max_index, 1])
    plt.ylabel(r'Sum of off diagonals (red. dens. mat.)')
    plt.xlabel(r'$J\,t$')
    plt.tight_layout(padding)
    ###
    plt.savefig(pltfolder + 'offdiag_densitymatrix_reduced.eps', format='eps', dpi=1000)
    plt.clf()
    print('\n densredavg = %f' % np.average(offdiagdens[loavgind:, 1]))

# ## single off diagonals
if os.path.isfile('../data/offdiagsingle.txt') and os.path.isfile('../data/offdiagsingleinfo.txt'):
    singlesdat = np.loadtxt('../data/offdiagsingle.txt')
    singlesinfo = np.loadtxt('../data/offdiagsingleinfo.txt')

    dt = singlesdat[1, 0] - singlesdat[0, 0]
    nrm = singlesdat[:, 0] / dt
    nrm[1:] = 1 / nrm[1:]

    '''
    for i in range(0,sysVar.m):
        for j in range(0,sysVar.occEnSingle):
            infoind = 1+4*j+2 #so we start at the first energy
            f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
            f.suptitle(r'$n_{%i} \; E_1=%.2e \; E_2=%.2e$' % (i, singlesinfo[i,infoind], singlesinfo[i,infoind+1]))
            ind = 1+2*j+(i*sysVar.occEnSingle*2)
            comp = singlesdat[:,ind] + 1j*singlesdat[:,ind+1]
            ax1.set_ylabel(r'$|A_{n,m}|$')
            ax1.plot(singlesdat[:,0], np.abs(comp), linewidth = 0.5)
            tmp = cumtrapz(comp,singlesdat[:,0]/dt,initial=comp[0])
            tmp = np.multiply(tmp,nrm)
            ax2.set_ylabel(r'average $|A_{n,m}|$')
            ax2.plot(singlesdat[:,0], np.abs(tmp), linewidth = 0.5)
            ax3.set_ylabel(r'arg$/\pi$')
            plt.xlabel(r'$J\,t$')
            ax3.plot(singlesdat[:,0], np.angle(comp)/(np.pi), linewidth = 0.5)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9, left=0.1)
            pp.savefig(f)
            f.clf()
            # do the double log plot
            de = np.abs(singlesinfo[i,infoind] - singlesinfo[i,infoind+1])
            linar = np.zeros(len(singlesdat[:,0]), dtype=np.float64)
            linar[0] = 0
            linar[1:] = 2/(singlesdat[1:,0] * de)
            plt.xlabel(r'$J\,t$')
            plt.ylabel(r'relative average $|A_{n,m}|$')
            plt.loglog(singlesdat[1:,0], np.abs(tmp/np.abs(comp[0]))[1:], singlesdat[1:,0], linar[1:], lw=0.5)
            pp.savefig()
            plt.clf()
        print('.',end='',flush=True)
    '''
    for i in range(0, sysVar.m):
        for j in range(0, sysVar.occEnSingle):
            infoind = 1 + 4 * j + 2  # so we start at the first energy
            # fetch the exponents. if abs(ordr)==1 set to zero for more readability
            f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
            ordr1 = int(np.log10(np.abs(singlesinfo[i, infoind])))
            if ordr1 == 1 or ordr1 == -1:
                ordr1 = 0
            ordr2 = int(np.log10(np.abs(singlesinfo[i, infoind + 1])))
            if ordr2 == 1 or ordr2 == -1:
                ordr2 = 0
            if ordr1 == 0 and ordr2 == 0:
                f.suptitle(
                    r'$n_{%i} \quad E_n=%.2f \; E_m=%.2f$' % (i, singlesinfo[i, infoind], singlesinfo[i, infoind + 1]))
            elif ordr1 == 0:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \; E_m=%.2f \cdot 10^{%i}$' % (
                    i, singlesinfo[i, infoind], singlesinfo[i, infoind + 1] / (10 ** ordr2), ordr2))
            elif ordr2 == 0:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \cdot 10^{%i} \; E_m=%.2f$' % (
                    i, singlesinfo[i, infoind] / (10 ** ordr1), ordr1, singlesinfo[i, infoind + 1]))
            else:
                f.suptitle(r'$n_{%i} \quad E_n=%.2f \cdot 10^{%i} \; E_m=%.2f \cdot 10^{%i}$' % (
                    i, singlesinfo[i, infoind] / (10 ** ordr1), ordr1, singlesinfo[i, infoind + 1] / (10 ** ordr2),
                    ordr2))
            #
            ind = 1 + 2 * j + (i * sysVar.occEnSingle * 2)
            comp = singlesdat[:, ind] + 1j * singlesdat[:, ind + 1]

            # order of magnitude of the deviation
            if not (np.abs(np.abs(comp[0]) - np.abs(comp[-1])) == 0):
                ordr = int(np.log10(np.abs(np.abs(comp[0]) - np.abs(comp[-1])))) - 1
            else:
                ordr = 0
            ax1.set_ylabel(r'$|n(t)| - |n(0)| / 10^{%i}$' % ordr)
            ax1.plot(singlesdat[:, 0], (np.abs(comp) - np.abs(comp[0])) / np.abs(comp[-1]), linewidth=0.5)
            tmp = cumtrapz(comp, singlesdat[:, 0] / dt, initial=comp[0])
            tmp = np.multiply(tmp, nrm)

            # order of magnitude of the average
            if not (np.abs(tmp[1]) == 0):
                ordr = int(np.log10(np.abs(tmp[1]))) - 1
            else:
                ordr = 0
            ax2.set_ylabel(r'$|\overline{n}_{n,m}^{%i}| / 10^{%i}$' % (i, ordr))
            ax2.plot(singlesdat[:, 0], np.abs(tmp) / (10 ** ordr), linewidth=0.5)
            ax2.set_xlabel(r'$J\,t$')
            plt.tight_layout(padding)
            plt.savefig(pltfolder + 'offdiag_occupation_single_n%i_%i.eps' % (i, j), format='eps', dpi=1000)
            f.clf()
            plt.close()
            # do the double log plot
            de = np.abs(singlesinfo[i, infoind] - singlesinfo[i, infoind + 1])
            linar = np.zeros(len(singlesdat[:, 0]), dtype=np.float64)
            linar[0] = 0
            linar[1:] = 2 / (singlesdat[1:, 0] * de)
            plt.xlabel(r'$J\,t$')
            plt.ylabel(r'relative average $|n_{n,m}^{%i}|$' % i)
            plt.loglog(singlesdat[1:, 0], np.abs(tmp / np.abs(comp[0]))[1:], singlesdat[1:, 0], linar[1:], lw=0.5)
            plt.savefig(pltfolder + 'offdiag_occupation_single_n%i_%i_loglog.eps' % (i, j), format='eps', dpi=1000)
            plt.clf()
            plt.close()
        print('.', end='', flush=True)

    if os.path.isfile('../data/diagsingles.txt'):
        diagdat = np.loadtxt('../data/diagsingles.txt')
        if os.path.isfile('../data/energy.txt') and os.path.isfile('../data/hamiltonian_eigvals.txt'):
            # ## look for energy - this works because the energies are sorted
            engy = np.loadtxt('../data/energy.txt')
            eigengy = np.loadtxt('../data/hamiltonian_eigvals.txt')
            diff = 0
            for l in range(0, sysVar.dim):
                if np.abs(eigengy[l, 1] - engy[0, 1]) > diff and l != 0:
                    eind = l - 1
                    break
                else:
                    diff = np.abs(eigengy[l, 1] - engy[0, 1])
            if eind < 15:
                loran = 0
            else:
                loran = eind - 15

        for i in range(0, sysVar.m):
            if os.path.isfile('../data/energy.txt') and os.path.isfile('../data/hamiltonian_eigvals.txt'):
                plt.title(r'Diagonal weighted elements of $n_{%i}$ in spectral decomp.' % (i))
                lo = np.int32(sysVar.dim * i)
                hi = np.int32(lo + sysVar.dim)
                plt.ylabel(r'$|n%i_{E}|$' % i)
                plt.xlabel(r'$E / J$')
                # plt.plot(diagdat[lo:hi,1], diagdat[lo:hi,2],linestyle='none',marker='o',ms=0.5)
                plt.plot(diagdat[lo + loran:hi, 1][:30], diagdat[lo + loran:hi, 2][:30], marker='o', ms=2)
                plt.axvline(x=engy[0, 1], linewidth=0.8, color='red')

                ###inlay
                a = plt.axes([0.18, 0.6, 0.28, 0.28])
                a.plot(diagdat[lo:hi - 300, 1], diagdat[lo:hi - 300, 2], marker='o', ms=0.6, ls='none')
                a.set_xticks([])
                a.set_yticks([])
                plt.tight_layout(padding)
                plt.savefig(pltfolder + 'diag_occupations_n%i_weighted.eps' % i, format='eps', dpi=1000)
                plt.clf()
                if os.path.isfile('../data/occ' + str(i) + '_re.txt'):
                    occmat = np.loadtxt('../data/occ' + str(i) + '_re.txt')
                    diags = np.zeros(sysVar.dim)

                    ### large plot
                    plt.title(r'Diagonal elements of $n_{%i}$ in spectral decomposition' % (i))
                    plt.ylabel(r'$|n%i_{E}|$' % (i))
                    plt.xlabel(r'$E / J$')
                    for el in range(0, sysVar.dim):
                        diags[el] = occmat[el, el]

                    plt.plot(diagdat[lo + loran:hi, 1][:30], diags[loran:][:30], marker='o', ms=2)
                    plt.axvline(x=engy[0, 1], linewidth=0.8, color='red')
                    ### inlay
                    a = plt.axes([0.18, 0.6, 0.28, 0.28])
                    a.plot(diagdat[lo:hi - 50, 1], diags[:-50], marker='o', ms=0.5, ls='none')
                    a.set_xticks([])
                    a.set_yticks([])
                    plt.savefig(pltfolder + 'diag_occupations_n%i.eps' % i, format='eps', dpi=1000)
                    plt.clf()
            else:
                plt.title(r'Diagonal weighted elements of $n_{%i}$ in spectral decomp.' % (i))
                lo = np.int32(sysVar.dim * i)
                hi = np.int32(lo + sysVar.dim)
                plt.ylabel(r'$|n%i_{E}|$' % i)
                plt.xlabel(r'$E / J$')
                # plt.plot(diagdat[lo:hi,1], diagdat[lo:hi,2],linestyle='none',marker='o',ms=0.5)
                plt.plot(diagdat[lo:hi - np.int32(sysVar.dim / 100), 1], diagdat[lo:hi - np.int32(sysVar.dim / 100), 2],
                         marker='o', ms=0.6, ls='none')
                plt.tight_layout(padding)
                plt.savefig(pltfolder + 'diag_occupations_n%i_weighted.eps' % i, format='eps', dpi=1000)
                plt.clf()

        print('.', end='', flush=True)
        plt.close()

print(' done!')
