"""This is the primary entrypoint for generating the analysis that appears in "Unsupervised Transient Light
Curve Analysis Via Hierarchical Bayesian Inference" by Sanders, Betancourt, and Soderberg (arXiv:1404.3619).
"""

# TODO This is a bad pattern!  We should better protect the global namespace
from utils import *


#############################################################
######## Load cold chains
#############################################################

ml_snames = [obj for obj in allevents if 'iip' in ttype_u[obj].lower()]

ml_sfit = getCmdStanSamples(rootdir+'stan/mod7_4/samples_mod7_?_?.csv',warmup=0,nodiv=0)

runtime = 3.*24 # hours
Nchains = 32

writetex(ml_valfile,'Nsampcold',len(ml_sfit['lp__']))
writetex(ml_valfile,'Nsampcoldrate','%0.2f'%(len(ml_sfit['lp__'])/(Nchains * runtime)))
writetex(ml_valfile,'Nsampcoldperchain','%0.0f'%(len(ml_sfit['lp__'])/(Nchains)))



#############################################################
######## Some statistics
#############################################################

## Total number of parameters
transformed_parameters = ['dm','mm','pt0','t1','t2','td','tp','lalpha','lbeta1','lbeta2','lbetadN','lbetadC','Mp','Yb','V','M1','M2','Md']
generated_quantities = ['fL_out','t0','mpeak','tplateau']
params = 0
for key in ml_sfit.keys():
    if '__' not in key and '#' not in key and key not in transformed_parameters and key not in generated_quantities:
	params += product(shape(ml_sfit[key])[1:])
	print key

writetex(ml_valfile,'totalparams',"{:,d}".format(int(params)))


## Grab some statistics about the PS1 sample from the other paper's file
textransfer(valfile,ml_valfile,'stackNtemptotalALL')
textransfer(valfile,ml_valfile,'NIIPtotalF')
textransfer(valfile,ml_valfile,'NphotTot')
textransfer(valfile,ml_valfile,'NphotTotDet')

## grab object names
for name in ml_snames: textransfer(valfile,ml_valfile,'PSO'+numsafe(name.split('-')[-1]))


#############################################################
######## Sampler diagnostic plots
#############################################################

### Treedepth illustration
max_treedepth = 16.

fig,ax = plt.subplots(1,figsize=(4,3))
plt.subplots_adjust(left=0.15,bottom=0.15)
s = ax.scatter(ml_sfit['stepsize__'], ml_sfit['n_leapfrog__'], marker='.', s = 12, c = ((ml_sfit['treedepth__']-1)/max_treedepth), vmin=-0.001, vmax=1.001, cmap=CB2cm['OrSB'], edgecolors='none')
plt.loglog()
plt.axis([1e-7,1e-1,1,1e6])

cb = plt.colorbar(s)
cb.set_label('Treedepth')
cb.set_ticks(np.linspace(0,1,5))
cb.set_ticklabels([str(c) for c in max_treedepth * np.linspace(0,1,5)])
cb.ax.minorticks_on()

ax.set_xlabel('Step size')
ax.set_ylabel('$N_{\\rm{leapfrog}}$')

plt.savefig(paperdirml+'treedepth_leapfrog.pdf')





#############################################################
######## Trace plot
#############################################################


Nchains = int(max(ml_sfit['chain#']))
Nwarmup=30
all_chains = unique(ml_sfit['chain#'])
random.shuffle(all_chains)
#chain_id = np.digitize(ml_sfit['chain#'],arange(0,Nchains+1,1)+.5)


## Hyperparameter plot
fig,ax = plt.subplots(1,figsize=(4,3))
plt.subplots_adjust(left=0.2, bottom=0.15)
for i in all_chains[0:4]: # plot just a few chains
    sel = (ml_sfit['chain#'] == i)
    plt.plot(ml_sfit['r_hP'][sel,2],alpha=0.5)

plt.ylabel('$r_{hP,\\beta_2}$')
plt.xlabel('Step')
plt.semilogx()
plt.axvline(Nwarmup,c='r')

plt.savefig(paperdirml+'trace_r_hP2.pdf')





## Bottom level parameter plot
fig,ax = plt.subplots(1,figsize=(4,3))
plt.subplots_adjust(left=0.2, bottom=0.15)
goodobjs = where(std(ml_sfit['lbeta2'][:,:,1],axis=0) < 0.1)[0]
pickgood = goodobjs[random.randint(0,len(goodobjs))]
for i in all_chains[0:4]: # plot just a few chains
    sel = (ml_sfit['chain#'] == i)
    plt.plot(ml_sfit['lbeta2'][sel,pickgood,1],alpha=0.5)

plt.ylabel('$\log \\beta_2$')
plt.xlabel('Step')
plt.semilogx()
plt.axvline(Nwarmup,c='r')

plt.savefig(paperdirml+'trace_beta2.pdf')

writetex(ml_valfile, 'betatwotracepick', pso_shortname_dict[ml_snames[pickgood]])



## Log prob maximization
fig,ax = plt.subplots(1,figsize=(4,3))
plt.subplots_adjust(left=0.2, bottom=0.15)
for i in all_chains[0:4]: # plot just a few chains
    sel = (ml_sfit['chain#'] == i)
    plt.plot(ml_sfit['lp__'][sel],alpha=0.5)

plt.ylabel('$\log P$')
plt.xlabel('Step')
plt.semilogx()
ax.yaxis.set_major_locator(MaxNLocator(5,prune='lower'))
ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(arange(-5e4,3e4,2e3)))
plt.axvline(Nwarmup,c='r')

plt.savefig(paperdirml+'trace_logprob.pdf')





#############################################################
######## Load warm chains
#############################################################

plt.close('all')

## unload cold chains to save memory
del ml_sfit

ml_sfit_warm = getCmdStanSamples(rootdir+'stan/mod7_4/samples_mod7_?_?.csv',warmup=50,nodiv=0)

writetex(ml_valfile,'Nsampwarm',len(ml_sfit_warm['lp__']))

writetex(ml_valfile, 'rhPtracerhat', '%0.2f'%calc_rhat(ml_sfit_warm,'r_hP',takelast=None,sdim1=2))
writetex(ml_valfile, 'betatwotracepickrhat', '%0.2f'%calc_rhat(ml_sfit_warm,'lbeta2',takelast=None,sdim1=pickgood,sdim2=1))



#############################################################
######## PPC comparisons
#############################################################

#### First do a poorly-identified SN
fig,axs = plt.subplots(2,figsize=(4,6),sharex='all',sharey='all')
bax = plt.axes([0.1,0.07,0.9,.9],frameon=False)
bax.set_xticks([]); bax.set_yticks([])
bax.set_ylabel('PS1 $l$ (r-band)')
plt.subplots_adjust(left=0.2)
## Individual model fit
name='2010-G-061196'
SNquickload(name,getfit=1)
SPII_plot_flux(name,'r',dpeak,fit,plotlines=20,multi=1,MJD=1,mag=0,ax=axs[0],ML=0,Kcor=1)
axs[0].set_title('Individual')
axs[0].set_xlabel(''); axs[0].set_ylabel('')

## Hierarchical fit
SPII_plot_tm(name,'r',[ml_snames,ml_sfit_warm],subset='all',plotlines=20,multi=1,MJD=1,mag=0,ax=axs[1],ML=0,Kcor=1)
axs[1].set_title('Hierarchical')
axs[1].set_ylabel('')

axs[1].axis([55329,55639,-500,1e4])
LCticks(axs)

plt.savefig(paperdirml+'PPC_compare_lq.pdf')




#### Next do a suite of well- and poorly-identified SNe
fig,axs = plt.subplots(2,4,figsize=(12,6),sharex='col',sharey='all')
bax = plt.axes([0.03,0.07,0.9,.9],frameon=False)
bax.set_xticks([]); bax.set_yticks([])
bax.set_ylabel('PS1 mag')
plt.subplots_adjust(left=0.1,wspace=.1,right=.95,hspace=0.03)

## What objects to show
ppc_picks = [('2011-K-330027','g'), ('2012-H-420393','r'), ('2011-A-120215','z'), ('2012-C-370519','y')]
for i in range(len(ppc_picks)):
    name,b = ppc_picks[i]
    ## Individual model fit
    SNquickload(name,getfit=1)
    SPII_plot_flux(name,b,dpeak,fit,plotlines=20,multi=1,MJD=1,ax=axs[0,i],ML=0,Kcor=1,mag=1)
    axs[0,i].set_title('abcdef'[i]+'. '+pso_name_dict[name]+' $'+b+'$-band',size=10)
    axs[0,i].set_xlabel(''); axs[0,i].set_ylabel('')

    ## Hierarchical fit
    SPII_plot_tm(name,b,[ml_snames,ml_sfit_warm],subset='all',plotlines=20,multi=1,MJD=1,ax=axs[1,i],ML=0,Kcor=1,mag=1)
    axs[1,i].set_title('')
    axs[1,i].set_ylabel('')

LCticks(axs)
axs[0,0].set_ylabel('Individual')
axs[1,0].set_ylabel('Hierarchical')

axs[0,2].set_xlim([55470,55760])

plt.savefig(paperdirml+'PPC_compare_hq.pdf')




#############################################################
######## Joint posterior slices
#############################################################

### sig_r_hSNF[2,2] versus r_hSNF[2,2] - winner!  High correlation
fig,ax = plt.subplots(1,figsize=(4,4))
plt.subplots_adjust(bottom=0.15,left=0.2)
ax.hist2d(log(ml_sfit_warm['sig_r_hSNF'][:,2,2]),ml_sfit_warm['r_hSNF'][:,2,2],bins=40,cmap=cm.gist_heat_r)
plt.xlabel('$\log~\sigma r_{hSNF,\\beta_2}$')
plt.ylabel('$r_{hSNF,\\beta_2}$')
ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(arange(-10,10,0.1)))
ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(arange(-10,10,0.1)))

plt.savefig(paperdirml+'joint_rhSNF.pdf')



### t_hP[1] versus t_hF[2,2]
fig,ax = plt.subplots(1,figsize=(4,4))
plt.subplots_adjust(bottom=0.15,left=0.2)
ax.hist2d(ml_sfit_warm['t_hP'][:,2],ml_sfit_warm['t_hF'][:,2,2],bins=20,cmap=cm.gist_heat_r)
plt.xlabel('$t_{hP,t_2}$')
plt.ylabel('$t_{hF,t_2,i}$')
ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(arange(-10,10,0.01)))
ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(arange(-10,10,0.02)))

plt.savefig(paperdirml+'joint_thP_thF.pdf')





#############################################################
######## Distribution of plateau duration distributions plot
#############################################################

## Get posterior samples
sf_htp = (log(10) + ml_sfit_warm['t_hP'][:,1] + ml_sfit_warm['sig_t_hP'][:,1] *(ml_sfit_warm['t_hF'][:,1,1] * ml_sfit_warm['sig_t_hF'][:,1,1]))
sf_ht2 = (log(100) + ml_sfit_warm['t_hP'][:,2] + ml_sfit_warm['sig_t_hP'][:,2] *(ml_sfit_warm['t_hF'][:,2,1] * ml_sfit_warm['sig_t_hF'][:,2,1]))

sf_s_htp = (ml_sfit_warm['sig_t_hP'][:,1] * ml_sfit_warm['sig_t_hF'][:,1,1])
sf_s_ht2 = (ml_sfit_warm['sig_t_hP'][:,2] * ml_sfit_warm['sig_t_hF'][:,2,1])

ind_tpf = getLCp('tplateau',1,1,eventsel=ml_snames)[ where(getLCp('tplateau',2,1,eventsel=ml_snames)<15) ]
hier_tpf = median(ml_sfit_warm['tplateau'][:,:,1],axis=0)


## Pick some samples and estimate posterior probabilities
Nplot=25
pd=linspace(80,150,10000)
prob=np.zeros(Nplot);pick=np.zeros(Nplot)
for i in range(Nplot):
    pick[i]=np.random.randint(0,len(sf_htp),1)
    cond1 = (abs(sf_htp - sf_htp[pick[i]]) < std(sf_htp)/10.) & (abs(sf_ht2 - sf_ht2[pick[i]]) < std(sf_ht2)/10.)
    cond2 = (abs(sf_ht2 - sf_ht2[pick[i]]) < std(sf_ht2)/10.) & (abs(sf_ht2 - sf_ht2[pick[i]]) < std(sf_ht2)/10.)
    prob[i]=len(np.where( cond1 & cond2)[0])

## Calculate population stats
Nmc = 1000
fracabove = np.zeros(Nmc)
dist_std = np.zeros(Nmc) ; dist_mean = np.zeros(Nmc)
for i in range(Nmc):
    thispick = np.random.randint(0,len(sf_htp),1)
    slm = sum_of_lognorm([sf_htp[thispick],sf_ht2[thispick]], [sf_s_htp[thispick],sf_s_ht2[thispick]])
    ## Calculate fraction of probability in the posterior above the maximum object
    fracabove[i] = slm.integrate_box(max(hier_tpf),1000)
    ## Calculate distribution statistics
    slm_samp = slm.resample([5000])
    dist_std[i] = std(slm_samp)
    dist_mean[i] = mean(slm_samp)


## Plot
fig,axs = plt.subplots(2,figsize=(5,4),sharex='all')
bax = plt.axes([0.1,0.07,0.9,.9],frameon=False)
bax.set_xticks([]); bax.set_yticks([])
bax.set_ylabel('$f$')
plt.subplots_adjust(left=0.2,bottom=0.1,hspace=0.1,top=0.9)
axs[1].set_xlabel('Plateau duration ($r$-band, rest frame days)')
px = linspace(40,200,200)
for i in range(Nplot):
    slm = sum_of_lognorm([sf_htp[pick[i]],sf_ht2[pick[i]]], [sf_s_htp[pick[i]],sf_s_ht2[pick[i]]])
    axs[1].plot(px, slm(px), alpha=(prob[i]/max(prob)),c='k')

axs[0].hist(ind_tpf,25,range=[20,250],color=CBcdict['Or'],ls='dashed',normed=1,lw=3,histtype='step',label='Individual fits',log=1)
axs[0].hist(hier_tpf,25,range=[20,250],color=CBcdict['rP'],edgecolor='none',normed=1,alpha=0.2,label='Bottom level\nmedians',log=1)
axs[0].axis([50,150,0.001,.12])
axs[0].plot([],[],label='Hyperprior draws',c='k')
axs[0].legend(prop={'size':8},ncol=3,bbox_to_anchor=(1,1.25))

axs[0].xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(arange(0,300,5)))

axs_i_0 = plt.axes([.55,.35,.12,.12])
axs_i_1 = plt.axes([.72,.35,.12,.12])
axs_i_0.set_yticks([]); axs_i_1.set_yticks([])

axs_i_0.hist(dist_mean,25,range=[70,150],color='k',normed=1)
axs_i_0.set_xticks([75,100,125,150])
axs_i_0.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(arange(0,500,5)))
axs_i_0.xaxis.set_tick_params(labelsize=8)
axs_i_0.set_xlabel('Mean',fontsize=8)

axs_i_1.hist(dist_std,25,range=[0,200],color='k',normed=1)
axs_i_1.set_xticks([0,100,200])
axs_i_1.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(arange(0,500,10)))
axs_i_1.xaxis.set_tick_params(labelsize=8)
axs_i_1.set_xlabel('Std. dev.',fontsize=8)

plt.savefig(paperdirml+'tplat_dist_dist.pdf')


## Print out some values
writetex(ml_valfile,'tplatINDdistmean','%0.0f'%mean(ind_tpf))
writetex(ml_valfile,'tplatINDdiststd','%0.0f'%std(ind_tpf))
writetex(ml_valfile,'tplatHIERdistmean','%0.0f'%mean(hier_tpf))
writetex(ml_valfile,'tplatHIERdiststd','%0.0f'%std(hier_tpf))
## Population stats
writetex(ml_valfile,'tplatHIERdistmax','%0.0f'%max(hier_tpf))
writetex(ml_valfile,'tplatAboveMaxPten','%0.0f'%(100*sum(fracabove>0.1)/float(Nmc)))
writetex(ml_valfile,'tplatAboveMaxPtwen','%0.0f'%(100*sum(fracabove>0.2)/float(Nmc)))
writetex(ml_valfile,'tplatHIERdiststdmed','%0.0f'%median(dist_std))
writetex(ml_valfile,'tplatHIERdiststddown','%0.0f'%percentile(dist_std,16))
writetex(ml_valfile,'tplatHIERdiststdup','%0.0f'%percentile(dist_std,84))
writetex(ml_valfile,'tplatHIERdistmeanmed','%0.0f'%median(dist_mean))
writetex(ml_valfile,'tplatHIERdistmeandown','%0.0f'%percentile(dist_mean,16))
writetex(ml_valfile,'tplatHIERdistmeanup','%0.0f'%percentile(dist_mean,84))



