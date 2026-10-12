import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BLUE='#2B4396'; MAROON='#7A0019'; BLACK='#000000'; GRAY='#505050'; SERIF='DejaVu Serif'; SANS='DejaVu Sans'
a=pd.read_csv('panels/testwars_rank.csv').sort_values('rank',ascending=False)
b=pd.read_csv('panels/rank2026.csv'); b=b[b.model=='escalation'].sort_values('p',ascending=False).head(20).iloc[::-1]
nm=lambda c:{'Russia (Soviet Union)':'Russia','Yemen (North Yemen)':'Yemen','Myanmar (Burma)':'Myanmar','DR Congo (Zaire)':'DR Congo','Cambodia (Kampuchea)':'Cambodia'}.get(c,c)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,8.5),gridspec_kw={'width_ratios':[1,1.15]})
fig.subplots_adjust(left=0.13,right=0.97,top=0.82,bottom=0.14,wspace=0.55)
# panel a
lab=[f"{nm(c)} {y}" for c,y in zip(a.country,a.war_year)]
ax1.barh(lab,a['rank'],color=BLUE,height=0.62)
for i,(r,n) in enumerate(zip(a['rank'],a['of'])): ax1.text(r+0.6,i,f"{r} of {n}",va='center',fontsize=11,color=BLACK,family=SANS)
ax1.set_xlim(0,30); ax1.set_xlabel('Rank of the country the year before the war,\namong all countries below 1,000 deaths for two years (1 = highest risk)',fontsize=11,family=SANS,color=BLACK)
ax1.set_title('A. The 13 wars of 2021–2025:\nwhere the model ranked them the year before',fontsize=14,family=SERIF,weight='bold',color=BLUE,loc='left')
ax1.tick_params(labelsize=11); ax1.spines[['top','right']].set_visible(False)
ax1.axvline(19,color=MAROON,lw=1,ls='--'); ax1.text(19.3,12.6,'top 10 percent',fontsize=10,color=MAROON,family=SANS,style='italic')
# panel b
ax2.barh([nm(c) for c in b.country],b.p*100,color=BLUE,height=0.62)
for i,(p,dd) in enumerate(zip(b.p,b.deaths2025)): ax2.text(p*100+0.4,i,f"{p*100:.0f}%   ({int(dd):,} deaths in 2025)",va='center',fontsize=10.5,color=BLACK,family=SANS)
ax2.set_xlim(0,40); ax2.set_xlabel('Modelled probability of crossing 1,000 battle deaths in 2026',fontsize=11,family=SANS,color=BLACK)
ax2.set_title('B. The same model applied to 2025 data:\ncountries with active violence, ranked for 2026',fontsize=14,family=SERIF,weight='bold',color=BLUE,loc='left')
ax2.tick_params(labelsize=11); ax2.spines[['top','right']].set_visible(False)
fig.suptitle('What the fitted model assesses and what it predicts',fontsize=19,family=SERIF,weight='bold',color=BLUE,y=0.965)
fig.text(0.5,0.9,'Escalation model: conflict history since 1946 and battle deaths in the two prior years, fitted on 1991–2020 (UCDP 26.1; UCDP/PRIO ACD 26.1).\nPanel A is out of sample. Panel B is a forecast made in September 2026 and can be checked in 2027.',ha='center',va='center',fontsize=10.5,color=GRAY,family=SANS,linespacing=1.5)
fig.text(0.5,0.03,'Countries already above 1,000 deaths in 2025 are left out of Panel B. The claim is the ranking; the percentages are uncalibrated model outputs.\nAmong countries with 25 to 499 deaths in a year, 3 to 7 percent cross 1,000 the next year.',ha='center',va='center',fontsize=10,color=GRAY,family=SANS,linespacing=1.5)
fig.savefig('figures/peace_prediction.png',dpi=300,facecolor='white')
fig.savefig('figures/peace_prediction.pdf',facecolor='white')
from PIL import Image; Image.open('figures/peace_prediction.png').convert('RGB').save('figures/peace_prediction.jpg',quality=95)
print(a[['country','war_year','rank','of']].to_string(index=False)); print(b[['country','p','deaths2025']].iloc[::-1].round(3).to_string(index=False))
