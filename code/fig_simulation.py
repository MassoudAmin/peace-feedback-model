"""Draws the simulation figure from outputs/peace_simulation_series.npz.
Run from the package root, after code/peace_simulation.py:  python code/fig_simulation.py paper figures/figure7_simulation.png   (or: book <file>)
"""
import numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
style=sys.argv[1]; out=sys.argv[2]
S=np.load('outputs/peace_simulation_series.npz')
if style=='book': plt.rcParams.update({'font.family':'Liberation Serif','mathtext.fontset':'stix','font.size':8.5}); C2='#7A0019'
else: plt.rcParams.update({'font.family':'Liberation Sans','font.size':8}); C2='k'
fig,ax=plt.subplots(2,3,figsize=(6.6,4.3)); t=np.arange(61)
def pan(a,tag,yl,l1,l2,title,ymax):
    for side,ls,c,lab in (('stable','-','k',l1),('unstable','--',C2,l2)):
        q=S[f'{tag}_{side}']; a.plot(t,q[1],ls=ls,color=c,lw=1.4,label=lab); a.fill_between(t,q[0],q[2],color=c,alpha=0.12 if style=='book' else 0.10,lw=0)
    a.set_ylim(0,ymax); a.set_xlim(0,60); a.set_xlabel('round'); a.set_ylabel(yl); a.set_title(title,fontsize=8.5,loc='left'); a.legend(frameon=False,fontsize=7,loc='upper left',handlelength=2.2)
pan(ax[0,0],'trust','violence','trust 0.6','trust 0.2','(a) Revenge loop',9)
pan(ax[0,1],'transp','fear','transparency 0.9','transparency 0.2','(b) Fear loop',14)
pan(ax[0,2],'arms','arms','reaction 0.10','reaction 0.45','(c) Arms race',4)
pan(ax[1,0],'delay','violence','total delay 2','total delay 8','(d) Delay',9)
a=ax[1,1]; a.plot(S['trust_grid'],S['trust_fail'],'-o',color='k',lw=1.2,ms=3)
a.axvline(float(S['T_ceiling']),color='k',ls=':',lw=1); a.axvline(float(S['T_budget']),color=C2,ls='--',lw=1.2)
a.text(float(S['T_ceiling'])+0.012,0.18,'enforcement\nceiling',fontsize=6.8,va='center'); a.text(float(S['T_budget'])+0.02,0.86,'budget',fontsize=6.8,color=C2)
a.set_xlabel('trust'); a.set_ylabel('share of plays that fail'); a.set_ylim(-0.03,1.05); a.set_xlim(0,0.92); a.set_title('(e) Trust threshold',fontsize=8.5,loc='left')
a=ax[1,2]; b=S['balance_bins']; labs=['below 0']+[f'{lo:.1f}–{hi:.1f}' for lo,hi in b[:-1,:2]]+['above 0.5']; vals=[float(S['balance_neg'])]+list(b[:,3])
a.bar(range(len(vals)),vals,color=['0.55']+['w']*(len(vals)-1),edgecolor='k',lw=0.9); a.set_xticks(range(len(vals))); a.set_xticklabels(labs,fontsize=6.3,rotation=30,ha='right')
a.set_ylim(0,1.05); a.set_xlabel('reserve balance at the operating point'); a.set_ylabel('mean share of plays that fail'); a.set_title('(f) 3,000 parameter sets',fontsize=8.5,loc='left')
for r in ax:
    for a in r: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.tick_params(labelsize=7)
fig.tight_layout(pad=0.6,w_pad=1.0,h_pad=1.2); fig.savefig(out,dpi=600 if style=='paper' else 400,facecolor='white')
