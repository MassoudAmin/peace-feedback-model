# Source data

Download each file into this directory under the name shown. Nothing here is redistributed; each provider's terms apply. Version numbers in the file names are the ones used for the book; a later release will change some counts.

| File expected in `data/` | Source | Used by |
|---|---|---|
| `OrganizedViolenceCYDataSet26_1.csv` | UCDP Organized Violence Country-Year Dataset 26.1, ucdp.uu.se/downloads | fit1, predict, tone |
| `GEDEvent_v26_1.csv` | UCDP Georeferenced Event Dataset 26.1, ucdp.uu.se/downloads | fit2, food2, food3, tone2, pair, trust, loops4, silence_sensitivity, acled_second_record, monthly_panel |
| `UcdpPrioConflict_v26_1.csv` | UCDP/PRIO Armed Conflict Dataset 26.1, ucdp.uu.se/downloads | fit3 |
| `Dyadic_v26_1.csv` and `Dyadic_v26_1.xlsx` | UCDP Dyadic Dataset 26.1 (the csv for pair.py, the xlsx for the family tests) | pair, ucdp_family_tests, ucdp_family_tests2 |
| `BattleDeaths_v26_1.xlsx` | UCDP Battle-Related Deaths Dataset 26.1, dyadic | ucdp_family_tests, ucdp_family_tests2 |
| `ucdp-peace-agreements-222.xlsx` | UCDP Peace Agreement Dataset 22.2 | ucdp_family_tests |
| `ucdp-esd-dy-181.xlsx` | UCDP External Support Dataset 18.1, dyad-year | ucdp_family_tests |
| `micFINAL.csv` | UCDP Managing Intrastate Conflict (Africa) | ucdp_family_tests |
| `DECO_v.1.0.xlsx` | Deadly Electoral Conflict Dataset 1.0 | ucdp_family_tests |
| `translate_dyad.csv`, `translate_conf.csv` | UCDP identifier translation tables | ucdp_family_tests |
| `UCDPConflictTerminationDataset_v4_2024_Conflict.csv` | UCDP Conflict Termination Dataset 4-2024 | ucdp_family_tests2 |
| `ucdp-intrastate-conflict-onset-251.csv` | UCDP Onset Dataset 25.1, intrastate | ucdp_family_tests2 |
| `ucdp_issues_dataset_dyadyear_232.xlsx` | UCDP Conflict Issues Dataset 23.2, dyad-year | ucdp_family_tests2 |
| `V-Dem-CY-Core-v16.csv` | V-Dem Country-Year Core v16, v-dem.net | fit3, predict |
| `SIPRI-Milex-data-1949-2025_v1_2.xlsx` | SIPRI Military Expenditure Database (sheet "Share of GDP") | fit1 |
| `food_price_indices_data.csv` | FAO Food Price Index, monthly | fit1, fit2 |
| `Inflation-data.xlsx` | World Bank Global Database of Inflation (sheet `fcpi_m`) | food3 |
| `API_FI.RES.TOTL.MO_DS2_en_csv_v2_407033.csv` | World Bank, total reserves in months of imports | res |
| `API_SP.POP.0014.TO.ZS_*.csv`, `API_SP.POP.GROW_*.csv`, `API_EN.CLC.SPEI.XD_*.csv` | World Bank indicators: population ages 0-14, population growth, SPEI drought index | loops4 |
| `wfp_food_prices_global_YYYY.csv` (one per year) | WFP Global Food Prices, data.humdata.org | food |
| `events.YYYY.tab` or `events_YYYY.tsv` (one per year) | ICEWS event data, dataverse.harvard.edu | tone, tone2 |
| `WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv` | World Values Survey wave 7, cross-national file v6.0 (registration required) | trust |
| `acled/*aggregated*.xlsx` | ACLED regional weekly aggregates (registered account required): exported October 10-17, 2026 for acled_second_record; to the week of October 31, 2026 for monthly_panel | acled_second_record, monthly_panel |
| `GEDEvent_v26_01_26_09.csv` | UCDP Candidate Events Dataset, January-September 2026 (version 26.01.26.09), ucdp.uu.se/downloads | monthly_panel |
| `rtfp/WLD_RTFP_country_2026-11-02.csv` | World Bank Real Time Food Prices, monthly food price inflation by country, microdata.worldbank.org/catalog/4509 | monthly_panel |
| `adjacency.csv` | Land-border pairs, two columns `country_a`, `country_b`, in UCDP country names | loops4 |

Two notes.

`adjacency.csv` is not a download. It is a list of land-border pairs written in UCDP country names. Any land-contiguity source will serve once its country names are matched to UCDP's. The list used for the book is not shipped, so the contagion result of `loops4.py` is the one result here that a second party must rebuild an input for.

The DECO workbook as downloaded did not open in pandas on the original run and was repaired before use. If `DECO_v.1.0.xlsx` fails to open, open it in a spreadsheet program, save it again as `DECO_fixed.xlsx` in this directory, and the script will use that copy.
