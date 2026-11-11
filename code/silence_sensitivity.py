"""Sensitivity of the 'eleven of nineteen' count (Chapter 11 / Appendix D).

Baseline (food3.py in the replication package): state-based violence only (type_of_violence==1);
onset = first month t with v[t]>=25 after 24 months each <25, and >=1000 deaths in months t..t+11;
one onset per country (first found). Silence = no recorded state-based deaths in the 6 months before t.
"""
import sys, pandas as pd, numpy as np
csv = sys.argv[1] if len(sys.argv) > 1 else 'data/GEDEvent_v26_1.csv'
g = pd.read_csv(csv, usecols=['country','date_start','type_of_violence','best'])
g['ym'] = pd.to_datetime(g.date_start).dt.to_period('M')
idx = pd.period_range('1989-01','2025-12',freq='M')
def monthly(types):
    h = g[g.type_of_violence.isin(types)]
    return h.groupby(['country','ym']).best.sum().unstack(fill_value=0).reindex(columns=idx,fill_value=0)
cm_sb  = monthly([1]); cm_all = monthly([1,2,3])
def find(cm, quiet_months, quiet, thr12, first, one_per_country=True):
    out=[]
    for c in cm.index:
        v=cm.loc[c].values
        t=quiet_months
        while t < len(v)-12:
            if v[t-quiet_months:t].max()<quiet and v[t:t+12].sum()>=thr12 and v[t]>=first:
                out.append((c,idx[t]))
                if one_per_country: break
                t += 12
            else: t += 1
    return out
def silent(cm_for_silence, onsets, win):
    n=0; rows=[]
    for c,t in onsets:
        k=idx.get_loc(t); s=cm_for_silence.loc[c].values[k-win:k].sum()
        rows.append((c,str(t),int(s))); n += (s==0)
    return n, rows
def row(label, cm_onset, cm_sil, qm=24, quiet=25, thr12=1000, first=25, win=6, one=True):
    on=find(cm_onset,qm,quiet,thr12,first,one); n,rows=silent(cm_sil,on,win)
    print(f'{label:75s} onsets {len(on):3d}  silent {n:3d}  share {n/len(on) if on else float("nan"):.2f}')
    return on,rows
on,rows=row('BASELINE: state-based, quiet 24 mo <25, floor 25, >=1000 in 12 mo, silence 6 mo', cm_sb, cm_sb)
print(pd.DataFrame(rows,columns=['country','onset','deaths_prior_6m']).to_string(index=False))
print()
row('silence window 3 months', cm_sb, cm_sb, win=3)
row('silence window 12 months', cm_sb, cm_sb, win=12)
row('silence window 24 months (= the quiet window itself)', cm_sb, cm_sb, win=24)
row('quiet window 12 months', cm_sb, cm_sb, qm=12)
row('quiet window 36 months', cm_sb, cm_sb, qm=36)
row('floor 5 (quiet <5, first month >=5)', cm_sb, cm_sb, quiet=5, first=5)
row('floor 50', cm_sb, cm_sb, quiet=50, first=50)
row('floor 100', cm_sb, cm_sb, quiet=100, first=100)
row('war threshold 500 in 12 months', cm_sb, cm_sb, thr12=500)
row('war threshold 2000 in 12 months', cm_sb, cm_sb, thr12=2000)
row('conflict onset: quiet <5, >=300 in 12 mo, first >=25 (food3 second definition)', cm_sb, cm_sb, quiet=5, thr12=300)
row('all onsets per country (not only the first)', cm_sb, cm_sb, one=False)
row('silence judged on ALL violence types (non-state + one-sided counted)', cm_sb, cm_all)
row('onset AND silence on all violence types', cm_all, cm_all)
row('all types, silence 12 months', cm_all, cm_all, win=12)
