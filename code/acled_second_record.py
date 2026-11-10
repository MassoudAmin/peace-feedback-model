"""Second-record check of the silent onsets (Chapter 11 / Appendix D) against ACLED.

Inputs: UCDP GED v26.1 events CSV (onset definition as in silence_sensitivity.py) and the ACLED
regional weekly aggregates (Africa, Middle East, Asia-Pacific, Europe-Central Asia, Latin America,
US-Canada), exported 10-17 October 2026, with columns WEEK, COUNTRY, ADMIN1, EVENT_TYPE,
SUB_EVENT_TYPE, EVENTS, FATALITIES, DISORDER_TYPE.

Test 1. For each UCDP war onset (first month t with >=25 state-based deaths after 24 months each <25,
and >=1000 deaths in t..t+11) that falls inside ACLED coverage for the country (at least 12 months of
ACLED coverage before t), count ACLED events and fatalities by disorder type in the 6 months before t.
A UCDP-silent onset with ACLED political violence in the prior 6 months is a threshold artifact of
the UCDP record, not peace.

Test 2. Leading indicator. For the same onsets, compare the monthly rate of ACLED demonstrations
(protests and riots) and of political-violence events in months t-6..t-1 with the rate in t-18..t-7.
"""
import sys, glob, pandas as pd, numpy as np
ged_csv = sys.argv[1] if len(sys.argv) > 1 else 'data/GEDEvent_v26_1.csv'
acled_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/acled'

# ---- UCDP onsets ----
g = pd.read_csv(ged_csv, usecols=['country','date_start','type_of_violence','best'])
g['ym'] = pd.to_datetime(g.date_start).dt.to_period('M')
idx = pd.period_range('1989-01','2025-12',freq='M')
h = g[g.type_of_violence==1]
cm = h.groupby(['country','ym']).best.sum().unstack(fill_value=0).reindex(columns=idx,fill_value=0)
onsets=[]
for c in cm.index:
    v=cm.loc[c].values; t=24
    while t < len(v)-12:
        if v[t-24:t].max()<25 and v[t:t+12].sum()>=1000 and v[t]>=25:
            onsets.append((c,idx[t],int(v[t-6:t].sum()))); break
        t+=1
on = pd.DataFrame(onsets, columns=['country','onset','ucdp_deaths_prior6'])

# ---- ACLED weekly ----
frames=[]
for f in glob.glob(acled_dir+'/*aggregated*.xlsx'):
    d=pd.read_excel(f, usecols=['WEEK','COUNTRY','EVENT_TYPE','SUB_EVENT_TYPE','EVENTS','FATALITIES','DISORDER_TYPE'])
    frames.append(d)
a=pd.concat(frames, ignore_index=True); a['ym']=pd.to_datetime(a.WEEK).dt.to_period('M')
a['demo']=a.DISORDER_TYPE.str.contains('Demonstrations', na=False)
a['pv']=a.DISORDER_TYPE.str.contains('Political violence', na=False)
# ACLED country names -> UCDP names
rename={'Democratic Republic of Congo':'DR Congo (Zaire)','Republic of Congo':'Congo','Russia':'Russia (Soviet Union)',
        'Serbia':'Serbia (Yugoslavia)','Yemen':'Yemen (North Yemen)','Bosnia and Herzegovina':'Bosnia-Herzegovina',
        'Myanmar':'Myanmar (Burma)','Cambodia':'Cambodia (Kampuchea)','Iran':'Iran','Zimbabwe':'Zimbabwe (Rhodesia)'}
a['country']=a.COUNTRY.replace(rename)
start=a.groupby('country').ym.min()
a['demo_ev']=a.EVENTS*a.demo; a['pv_ev']=a.EVENTS*a.pv; a['pv_f']=a.FATALITIES*a.pv
mon=a.groupby(['country','ym']).agg(events=('EVENTS','sum'),fatal=('FATALITIES','sum'),
    demo=('demo_ev','sum'),pv=('pv_ev','sum'),pv_fatal=('pv_f','sum')).reset_index()
a[['country','ym','EVENT_TYPE','EVENTS','FATALITIES','DISORDER_TYPE']].to_pickle(acled_dir+'/acled_all.pkl')
mon=mon.set_index(['country','ym'])

rows=[]
for _,r in on.iterrows():
    c,t=r.country,r.onset
    if c not in start.index: rows.append((c,str(t),r.ucdp_deaths_prior6,'not covered',None,None,None,None,None,None)); continue
    cov=(t-start[c]).n
    if cov<12: rows.append((c,str(t),r.ucdp_deaths_prior6,f'covered from {start[c]}',None,None,None,None,None,None)); continue
    def w(a_,b_,col):
        ks=[(c,t-k) for k in range(a_,b_+1)]
        return float(sum(mon[col].get(k,0) for k in ks))
    rows.append((c,str(t),r.ucdp_deaths_prior6,f'from {start[c]}',
        w(1,6,'pv'),w(1,6,'pv_fatal'),w(1,6,'demo'), w(7,18,'pv')/12, w(7,18,'demo')/12, w(1,6,'demo')/6))
out=pd.DataFrame(rows,columns=['country','onset','ucdp_deaths_prior6','acled_coverage','acled_pv_events_prior6',
    'acled_pv_fatalities_prior6','acled_demos_prior6','pv_per_month_t18_t7','demos_per_month_t18_t7','demos_per_month_t6_t1'])
pd.set_option('display.width',250)
print('UCDP war onsets (baseline definition) and the ACLED record in the six months before onset')
print(out.to_string(index=False))
cov=out[out.acled_pv_events_prior6.notna()]
sil=cov[cov.ucdp_deaths_prior6==0]
print(f'\nOnsets inside ACLED coverage: {len(cov)}; of these UCDP-silent: {len(sil)}')
print(f'UCDP-silent onsets with ACLED political violence in prior 6 months: {(sil.acled_pv_events_prior6>0).sum()} of {len(sil)}')
print(f'UCDP-silent onsets with ACLED reported fatalities in prior 6 months: {(sil.acled_pv_fatalities_prior6>0).sum()} of {len(sil)}')
print(f'Median ACLED political-violence events in prior 6 months, silent onsets: {sil.acled_pv_events_prior6.median():.0f}; fatalities: {sil.acled_pv_fatalities_prior6.median():.0f}')
r1=(cov.demos_per_month_t6_t1/cov.demos_per_month_t18_t7.replace(0,np.nan)); r2=((cov.acled_pv_events_prior6/6)/cov.pv_per_month_t18_t7.replace(0,np.nan))
print(f'Demonstrations, ratio of monthly rate t-6..t-1 to t-18..t-7: median {r1.median():.2f}, rose in {(r1>1).sum()} of {r1.notna().sum()}')
print(f'Political violence events, same ratio: median {r2.median():.2f}, rose in {(r2>1).sum()} of {r2.notna().sum()}')
out.to_csv(acled_dir+'/out_acled_second_record.csv',index=False)
