#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a synthetic OCC-style Quarterly Credit Portfolio Review report."""

import numpy as np

np.random.seed(789)

# ── helpers ──────────────────────────────────────────────────────────────────
def pct(x, d=4):
    return f"{x * 100:.{d}f}%"

def dol(x):
    """Format as $X,XXX.X (millions)."""
    return f"${x:,.1f}"

def dol2(x):
    return f"${x:,.2f}"

# ── 1. Segment definitions ──────────────────────────────────────────────────
segments = [
    "C&I",
    "CRE-Owner Occupied",
    "CRE-Non Owner Occupied",
    "Multifamily",
    "1-4 Family Residential",
    "Consumer-Auto",
    "Consumer-Card",
    "Consumer-Other",
]

# Base balances ($ millions) – total ≈ 18-20 B
base_balances = np.array([4850, 2210, 3175, 1820, 2940, 1560, 980, 615])
growth_rates = np.array([0.042, 0.018, 0.067, 0.053, 0.029, 0.035, 0.081, 0.012])
current_balances = base_balances * (1 + growth_rates)
total_portfolio = current_balances.sum()
prior_total = base_balances.sum()

# ── 2. Credit quality by segment ────────────────────────────────────────────
delinq_30 = np.array([0.0112, 0.0087, 0.0198, 0.0065, 0.0143, 0.0231, 0.0347, 0.0189])
delinq_60 = delinq_30 * np.array([0.42, 0.38, 0.45, 0.35, 0.41, 0.44, 0.48, 0.40])
delinq_90 = delinq_60 * np.array([0.35, 0.30, 0.38, 0.28, 0.33, 0.36, 0.41, 0.34])

# NCO rates – INTENTIONAL contradiction: Consumer-Auto has low delinquency
# relative to peers but HIGH NCO (recoveries are poor on depreciated collateral)
nco_rate = np.array([0.0028, 0.0015, 0.0052, 0.0009, 0.0031, 0.0078, 0.0491, 0.0063])

classified_ratio = np.array([0.0245, 0.0198, 0.0412, 0.0132, 0.0267, 0.0189, 0.0523, 0.0311])
criticized_ratio = classified_ratio + np.array([0.008, 0.006, 0.012, 0.004, 0.009, 0.007, 0.015, 0.010])

# Policy thresholds
delinq_threshold = 0.030   # 30+ day delinquency
nco_threshold = 0.0060     # NCO rate
classified_threshold = 0.040
criticized_threshold = 0.060

# ── 3. Industry concentrations ──────────────────────────────────────────────
industries = [
    "Healthcare & Pharmaceuticals",
    "Commercial Real Estate – Office",
    "Technology & Software",
    "Oil & Gas Exploration",
    "Hospitality & Lodging",
    "Retail Trade",
    "Manufacturing – Durable Goods",
    "Transportation & Warehousing",
    "Agriculture & Farming",
    "Professional Services",
    "Construction",
    "Financial Services (Non-Bank)",
]

ind_exposure = np.array([2415.3, 2187.6, 1893.2, 1456.8, 1234.5, 1098.7,
                          987.4, 876.2, 654.3, 543.1, 498.7, 412.6])
# Internal limits (% of Tier 1 capital = $2,850M)
tier1_capital = 2850.0
ind_limit_pct = np.array([0.20, 0.18, 0.20, 0.15, 0.12, 0.15,
                           0.12, 0.10, 0.08, 0.08, 0.10, 0.06])
ind_limits = ind_limit_pct * tier1_capital
ind_util = ind_exposure / ind_limits

# ── 4. Vintage analysis ─────────────────────────────────────────────────────
vintages = [2019, 2020, 2021, 2022, 2023, 2024]
# Cumulative loss rates by vintage × segment (%)
# 2020 vintage for CRE-Non Owner is worst (COVID stress)
vintage_loss = np.array([
    # C&I    CRE-OO  CRE-NOO  MF     1-4Fam  Auto   Card   Other
    [0.0134, 0.0078, 0.0215, 0.0042, 0.0098, 0.0312, 0.1847, 0.0245],  # 2019
    [0.0198, 0.0112, 0.0387, 0.0067, 0.0145, 0.0398, 0.2134, 0.0312],  # 2020
    [0.0089, 0.0056, 0.0178, 0.0034, 0.0087, 0.0267, 0.1567, 0.0198],  # 2021
    [0.0067, 0.0043, 0.0134, 0.0028, 0.0065, 0.0198, 0.1234, 0.0156],  # 2022
    [0.0034, 0.0021, 0.0067, 0.0012, 0.0034, 0.0112, 0.0678, 0.0087],  # 2023
    [0.0012, 0.0008, 0.0023, 0.0004, 0.0011, 0.0045, 0.0234, 0.0034],  # 2024
])

# ── 5. Migration matrix ─────────────────────────────────────────────────────
# 5×5: Pass, Special Mention, Substandard, Doubtful, Loss
migration = np.array([
    [0.9312, 0.0478, 0.0156, 0.0042, 0.0012],
    [0.1245, 0.6834, 0.1423, 0.0387, 0.0111],
    [0.0387, 0.0912, 0.6245, 0.1834, 0.0622],
    [0.0112, 0.0234, 0.0867, 0.5912, 0.2875],
    [0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
])
rating_labels = ["Pass", "Special Mention", "Substandard", "Doubtful", "Loss"]

# ── 6. ALLL / ACL coverage ──────────────────────────────────────────────────
alll_balance = np.array([62.4, 28.7, 78.3, 14.2, 41.5, 38.9, 67.8, 12.3])
coverage_ratio = alll_balance / current_balances  # reserve / balance
required_coverage = np.array([0.0120, 0.0115, 0.0225, 0.0070, 0.0130, 0.0230, 0.0650, 0.0185])

# ── 7. Criticized asset trends (8 quarters) ─────────────────────────────────
quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023",
            "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
# Trend: classified rising for CRE-NOO, stable elsewhere
classified_trend_total = np.array([412.3, 425.8, 438.1, 451.7, 467.2, 489.3, 512.8, 538.4])
criticized_trend_total = np.array([534.7, 548.2, 561.9, 578.4, 598.1, 621.5, 648.7, 679.3])
special_mention_trend = criticized_trend_total - classified_trend_total

# ── 8. Geographic concentration ──────────────────────────────────────────────
states = ["California", "Texas", "New York", "Florida", "Illinois",
          "Ohio", "Georgia", "Washington", "Colorado", "North Carolina"]
geo_exposure = np.array([3215.4, 2876.3, 2134.7, 1987.2, 1654.8,
                          1234.5, 1098.7, 876.4, 765.3, 654.2])
geo_delinq = np.array([0.0187, 0.0143, 0.0112, 0.0234, 0.0198,
                        0.0156, 0.0213, 0.0098, 0.0134, 0.0167])
geo_limit_pct = np.array([0.20, 0.18, 0.15, 0.12, 0.10,
                           0.08, 0.07, 0.06, 0.05, 0.04])
geo_limits = geo_limit_pct * total_portfolio

# ── 9. Peer comparison ──────────────────────────────────────────────────────
peers = ["First Horizon Bancshares", "Valley National Corp",
         "Atlantic Capital Group", "Midwest Financial Holdings",
         "Pacific Western Bancorp"]

# ── 10. Watch list credits ───────────────────────────────────────────────────
watch_names = [
    "Meridian Healthcare Systems LLC",
    "Crossroads Office Tower LP",
    "Pinnacle Hospitality Group Inc",
    "Ironclad Manufacturing Co",
    "Sunbelt Retail Partners LP",
    "Great Plains Energy Corp",
    "Coastal Multifamily Holdings LLC",
    "Northwind Technology Solutions Inc",
    "Harvest Agricultural Co-op",
    "Summit Construction Group LLC",
    "Liberty Transportation Inc",
    "Redwood Financial Services LLC",
    "Cascadia Timber Holdings LP",
    "Heartland Auto Group Inc",
    "Metro Professional Services PA",
    "Bayshore Oil & Gas Partners LP",
    "Evergreen Retail Development Corp",
    "Pioneer Warehousing Solutions LLC",
    "Starlight Hospitality Ventures Inc",
    "Continental Office Properties LP",
]
watch_exposure = np.round(np.random.uniform(15, 145, 20), 1)
watch_exposure = np.sort(watch_exposure)[::-1]  # descending
watch_ratings = np.random.choice(["Special Mention", "Substandard", "Doubtful"], 20,
                                  p=[0.35, 0.45, 0.20])
watch_reasons = [
    "Declining revenue; covenant breach on debt service coverage",
    "Office vacancy rate exceeding 28%; negative cash flow",
    "Sustained occupancy decline post-pandemic; DSCR < 1.0x",
    "Supply chain disruption; two consecutive quarters of operating losses",
    "Anchor tenant departure; foot traffic down 34% YoY",
    "Commodity price exposure; hedging program inadequate",
    "Deferred maintenance; rising operating costs vs. flat rents",
    "Key client loss; revenue concentration > 40% single customer",
    "Drought conditions; crop yield projections reduced 25%",
    "Project cost overruns; completion timeline extended 9 months",
    "Equipment fleet aging; capex requirements exceed cash generation",
    "Regulatory investigation; potential for material fines",
    "Timber pricing at 5-year low; log inventory write-down",
    "Used vehicle margin compression; floor plan utilization elevated",
    "Partner departure; billable hour decline of 18%",
    "Well production below forecast; proved reserves downgraded",
    "Lease-up pace behind projections; interest reserve draw",
    "Labor shortages; contract renegotiations pending",
    "Renovation stalled; franchise agreement at risk of termination",
    "Remote work trend; sublease market softening in primary MSA",
]
watch_trends = np.random.choice(["Stable", "Deteriorating", "Improving"], 20,
                                 p=[0.30, 0.55, 0.15])

# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD REPORT
# ═══════════════════════════════════════════════════════════════════════════════
lines = []
L = lines.append

L("# Quarterly Credit Portfolio Review")
L("")
L("**Prepared for:** Board of Directors and Senior Management")
L("**Institution:** Pacific Continental Bank, N.A.")
L("**Charter Number:** 24-7891")
L("**Supervisory Office:** OCC Western District — San Francisco")
L("**Report Date:** December 31, 2024 (Q4 2024)")
L("**Examination Cycle:** Quarterly Ongoing Monitoring")
L("**Prepared by:** Credit Risk Management Division")
L("**Review Period:** October 1, 2024 through December 31, 2024")
L("")
L("---")
L("")
L("## Executive Summary")
L("")
L("This report presents the quarterly credit portfolio review for Pacific Continental")
L("Bank, N.A. as of December 31, 2024. The bank maintains a diversified loan portfolio")
L(f"totaling {dol(total_portfolio)} million across eight primary lending segments. During")
L(f"the quarter, total loans grew {pct((total_portfolio - prior_total) / prior_total, 2)}")
L("from the prior quarter-end balance. Key risk themes for the quarter include rising")
L("classified assets in the Commercial Real Estate Non-Owner Occupied segment, elevated")
L("net charge-off rates in the Consumer-Card and Consumer-Auto portfolios, and increasing")
L("industry concentration in Healthcare & Pharmaceuticals. The Allowance for Credit")
L("Losses stands at {:.4f}% of total loans, which management considers adequate given".format(
    alll_balance.sum() / total_portfolio * 100))
L("current portfolio composition and economic conditions.")
L("")
L("---")
L("")

# ── Section 1: Portfolio Overview ────────────────────────────────────────────
L("## 1. Portfolio Overview")
L("")
L("| Metric | Value |")
L("|---|---|")
L(f"| Total Outstanding Loans | {dol(total_portfolio)} million |")
L(f"| Prior Quarter Total | {dol(prior_total)} million |")
L(f"| Quarterly Growth (Nominal) | {dol(total_portfolio - prior_total)} million |")
L(f"| Quarterly Growth Rate | {pct((total_portfolio - prior_total) / prior_total)} |")
L(f"| Number of Active Loan Accounts | {np.random.randint(42000, 48000):,} |")
L(f"| Average Loan Size | {dol2(total_portfolio / 44312 * 1e6 / 1e3)} thousand |")
L(f"| Tier 1 Capital | {dol(tier1_capital)} million |")
L(f"| Total Loans / Tier 1 Capital | {total_portfolio / tier1_capital:.4f}x |")
L(f"| Total Risk-Weighted Assets | {dol(total_portfolio * 0.78)} million |")
L(f"| Loan-to-Deposit Ratio | {pct(0.8234)} |")
L("| Weighted Average Maturity | 4.7 years |")
L(f"| Weighted Average Coupon | {pct(0.0623)} |")
L("")
L("The portfolio continues to exhibit moderate growth concentrated in CRE Non-Owner")
L("Occupied and Consumer-Card segments. Management should monitor the pace of CRE")
L("originations relative to concentration policy limits.")
L("")
L("---")
L("")

# ── Section 2: Portfolio Composition ─────────────────────────────────────────
L("## 2. Portfolio Composition by Segment")
L("")
L("| Segment | Current Balance ($M) | % of Total | Prior Quarter ($M) | QoQ Growth | Annualized Growth |")
L("|---|---:|---:|---:|---:|---:|")
for i, seg in enumerate(segments):
    pq = base_balances[i]
    cb = current_balances[i]
    gr = growth_rates[i]
    ann_gr = (1 + gr)**4 - 1
    L(f"| {seg} | {dol(cb)} | {pct(cb / total_portfolio)} | {dol(pq)} | {pct(gr)} | {pct(ann_gr)} |")
L(f"| **Total** | **{dol(total_portfolio)}** | **100.0000%** | **{dol(prior_total)}** | **{pct((total_portfolio - prior_total) / prior_total)}** | — |")
L("")
L("Notable observations:")
L("")
L("- **CRE Non-Owner Occupied** grew 6.7000% QoQ, the fastest among all segments,")
L("  raising its share to {:.4f}% of total loans.".format(current_balances[2] / total_portfolio * 100))
L("- **Consumer-Card** expanded 8.1000% QoQ driven by promotional balance transfer")
L("  campaigns launched in October 2024.")
L("- **Consumer-Other** growth was muted at 1.2000%, reflecting strategic de-emphasis")
L("  of unsecured personal lending.")
L("")
L("---")
L("")

# ── Section 3: Credit Quality Metrics ────────────────────────────────────────
L("## 3. Credit Quality Metrics by Segment")
L("")
L("| Segment | 30+ DPD | 60+ DPD | 90+ DPD | NCO Rate | Classified Ratio | Criticized Ratio |")
L("|---|---:|---:|---:|---:|---:|---:|")
for i, seg in enumerate(segments):
    L(f"| {seg} | {pct(delinq_30[i])} | {pct(delinq_60[i])} | {pct(delinq_90[i])} | {pct(nco_rate[i])} | {pct(classified_ratio[i])} | {pct(criticized_ratio[i])} |")

wt_delinq30 = np.average(delinq_30, weights=current_balances)
wt_delinq60 = np.average(delinq_60, weights=current_balances)
wt_delinq90 = np.average(delinq_90, weights=current_balances)
wt_nco = np.average(nco_rate, weights=current_balances)
wt_class = np.average(classified_ratio, weights=current_balances)
wt_crit = np.average(criticized_ratio, weights=current_balances)
L(f"| **Portfolio Weighted Avg** | **{pct(wt_delinq30)}** | **{pct(wt_delinq60)}** | **{pct(wt_delinq90)}** | **{pct(wt_nco)}** | **{pct(wt_class)}** | **{pct(wt_crit)}** |")
L("")

L("### Policy Limit Compliance — Credit Quality")
L("")
L("| Segment | 30+ DPD | Threshold | Status | NCO Rate | Threshold | Status | Classified | Threshold | Status |")
L("|---|---:|---:|:---:|---:|---:|:---:|---:|---:|:---:|")
for i, seg in enumerate(segments):
    d_status = "PASS" if delinq_30[i] < delinq_threshold else "**FAIL**"
    n_status = "PASS" if nco_rate[i] < nco_threshold else "**FAIL**"
    c_status = "PASS" if classified_ratio[i] < classified_threshold else "**FAIL**"
    L(f"| {seg} | {pct(delinq_30[i])} | {pct(delinq_threshold)} | {d_status} | {pct(nco_rate[i])} | {pct(nco_threshold)} | {n_status} | {pct(classified_ratio[i])} | {pct(classified_threshold)} | {c_status} |")
L("")

L("> **Key Finding — Cross-Metric Contradiction:** Consumer-Auto passes the 30+ day")
L("> delinquency threshold (2.3100% vs. 3.0000% limit) but fails on NCO rate (0.7800%")
L("> vs. 0.6000% limit). This indicates that while early-stage delinquency is contained,")
L("> loss severity on defaulted auto loans is elevated due to depressed used vehicle")
L("> recovery values. The collateral shortfall on charged-off auto loans averaged 38.4%")
L("> in Q4 2024, up from 29.1% in Q4 2023.")
L("")
L("> **Key Finding — CRE Non-Owner Occupied:** This segment exceeds the classified ratio")
L("> threshold (4.1200% vs. 4.0000%) and shows the highest 30+ DPD rate among commercial")
L("> segments. Office sub-segment vacancy rates in the bank's primary MSAs average 19.7%,")
L("> up 340 bps from one year ago.")
L("")
L("---")
L("")

# ── Section 4: Concentration Analysis ────────────────────────────────────────
L("## 4. Industry Concentration Analysis")
L("")
L(f"**Tier 1 Capital Base:** {dol(tier1_capital)} million")
L("")
L("| Rank | Industry | Exposure ($M) | Internal Limit ($M) | Limit (% Tier 1) | Utilization | Status |")
L("|---:|---|---:|---:|---:|---:|:---:|")
for i, ind in enumerate(industries):
    status = "PASS" if ind_util[i] <= 1.0 else "**BREACH**"
    L(f"| {i+1} | {ind} | {dol(ind_exposure[i])} | {dol(ind_limits[i])} | {pct(ind_limit_pct[i])} | {pct(ind_util[i])} | {status} |")
L("")

# Identify breaches
breach_idx = np.where(ind_util > 1.0)[0]
L("### Concentration Limit Breaches")
L("")
if len(breach_idx) > 0:
    for idx in breach_idx:
        excess = ind_exposure[idx] - ind_limits[idx]
        L(f"- **{industries[idx]}:** Exposure of {dol(ind_exposure[idx])}M exceeds the")
        L(f"  internal limit of {dol(ind_limits[idx])}M by {dol(excess)}M")
        L(f"  ({pct(ind_util[idx])} utilization). Management must present a remediation")
        L("  plan to the Board Risk Committee within 30 days per Policy Section 4.2.3.")
        L("")
L("The bank has {} industry concentrations exceeding internal policy limits.".format(len(breach_idx)))
L("Remediation plans are required for each breach per the Credit Concentration Policy.")
L("")
L("---")
L("")

# ── Section 5: Vintage Analysis ──────────────────────────────────────────────
L("## 5. Vintage Analysis — Cumulative Loss Rates")
L("")
L("Cumulative net loss rates by origination vintage and loan segment as of December 31, 2024:")
L("")
header = "| Vintage | " + " | ".join(segments) + " | Weighted Avg |"
sep = "|---:|" + "---:|" * len(segments) + "---:|"
L(header)
L(sep)
for v, yr in enumerate(vintages):
    # weighted average across segments using current balances as proxy
    wavg = np.average(vintage_loss[v], weights=current_balances)
    row = f"| {yr} | " + " | ".join(pct(vintage_loss[v][s], 4) for s in range(len(segments))) + f" | {pct(wavg, 4)} |"
    L(row)
L("")

worst_v, worst_s = np.unravel_index(np.argmax(vintage_loss), vintage_loss.shape)
L(f"> **Key Finding — Worst Performing Vintage-Segment:** The {vintages[worst_v]} vintage")
L(f"> in the {segments[worst_s]} segment shows the highest cumulative loss rate at")
L(f"> {pct(vintage_loss[worst_v, worst_s])}. This vintage was originated during the peak")
L("> of pandemic-era stimulus spending when underwriting standards for revolving consumer")
L("> credit were loosened to support balance growth targets.")
L("")
L("> **Multi-Hop Observation:** The 2020 vintage of CRE Non-Owner Occupied has a cumulative")
L(f"> loss rate of {pct(vintage_loss[1, 2])}, which is the highest among all CRE sub-types")
L("> across all vintages. This vintage's losses are concentrated in office properties in")
L("> the San Francisco and Portland MSAs, where remote work adoption has permanently")
L("> reduced space demand. These same MSAs appear in the Geographic Concentration analysis")
L("> (Section 9) as part of the California and Washington exposures.")
L("")
L("---")
L("")

# ── Section 6: Migration Matrix ──────────────────────────────────────────────
L("## 6. Rating Migration Matrix")
L("")
L("One-quarter transition probability matrix based on internal risk rating movements")
L("from Q3 2024 to Q4 2024:")
L("")
header = "| From \\ To | " + " | ".join(rating_labels) + " | Row Total |"
L(header)
sep = "|---|" + "---:|" * len(rating_labels) + "---:|"
L(sep)
for i, lbl in enumerate(rating_labels):
    cells = " | ".join(f"{migration[i,j]:.4f}" for j in range(5))
    L(f"| **{lbl}** | {cells} | {migration[i].sum():.4f} |")
L("")

L("### Migration Summary Statistics")
L("")
L("| Metric | Value |")
L("|---|---:|")
L(f"| Pass Retention Rate | {pct(migration[0,0])} |")
L(f"| Pass → Downgrade Rate | {pct(1 - migration[0,0])} |")
L(f"| Special Mention → Pass (Upgrade) | {pct(migration[1,0])} |")
L(f"| Special Mention → Substandard+ (Downgrade) | {pct(migration[1,2] + migration[1,3] + migration[1,4])} |")
L(f"| Substandard → Doubtful/Loss | {pct(migration[2,3] + migration[2,4])} |")
L(f"| Doubtful → Loss | {pct(migration[3,4])} |")
L(f"| Weighted Average Downgrade Rate | {pct(0.0688)} |")
L(f"| Weighted Average Upgrade Rate | {pct(0.0412)} |")
L("")
L("The Pass retention rate of {:.4f}% is within the historical range of 92-95%.".format(migration[0,0]*100))
L("The Substandard-to-Loss migration rate of {:.4f}% warrants monitoring as it exceeds".format(migration[2,4]*100))
L("the 5.0000% internal threshold.")
L("")
L("---")
L("")

# ── Section 7: Allowance Coverage ────────────────────────────────────────────
L("## 7. Allowance for Credit Losses (ACL) Coverage")
L("")
L("| Segment | Outstanding ($M) | ACL Balance ($M) | Coverage Ratio | Required Coverage | Adequacy |")
L("|---|---:|---:|---:|---:|:---:|")
for i, seg in enumerate(segments):
    adequacy = "Adequate" if coverage_ratio[i] >= required_coverage[i] else "**Deficient**"
    L(f"| {seg} | {dol(current_balances[i])} | {dol2(alll_balance[i])} | {pct(coverage_ratio[i])} | {pct(required_coverage[i])} | {adequacy} |")
total_acl = alll_balance.sum()
total_cov = total_acl / total_portfolio
L(f"| **Total** | **{dol(total_portfolio)}** | **{dol2(total_acl)}** | **{pct(total_cov)}** | — | — |")
L("")

L("### ACL Build / Release Analysis")
L("")
L("| Quarter | Beginning ACL ($M) | Provision ($M) | Net Charge-Offs ($M) | Ending ACL ($M) | Coverage Ratio |")
L("|---|---:|---:|---:|---:|---:|")
acl_q = [318.4, 322.1, 328.7, 334.2, 338.9, 340.5, 342.1, total_acl]
for q_idx, q in enumerate(quarters):
    provision = np.random.uniform(8, 18, 1)[0]
    ncos = np.random.uniform(5, 14, 1)[0]
    if q_idx == 0:
        beg = 312.5
    else:
        beg = acl_q[q_idx - 1]
    end = acl_q[q_idx]
    provision = end - beg + ncos
    L(f"| {q} | {dol2(beg)} | {dol2(provision)} | {dol2(ncos)} | {dol2(end)} | {pct(end / (prior_total + (total_portfolio - prior_total) * (q_idx+1)/8))} |")
L("")

deficient = [(seg, i) for i, seg in enumerate(segments) if coverage_ratio[i] < required_coverage[i]]
if deficient:
    L("> **Key Finding — Reserve Deficiencies:**")
    for seg, i in deficient:
        shortfall = (required_coverage[i] - coverage_ratio[i]) * current_balances[i]
        L(f"> - **{seg}:** Coverage of {pct(coverage_ratio[i])} is below the required")
        L(f">   {pct(required_coverage[i])}, representing a shortfall of approximately")
        L(f">   {dol2(shortfall)}M. Management should evaluate whether the current")
        L(">   qualitative factor overlays are sufficient.")
    L("")

L("---")
L("")

# ── Section 8: Criticized Asset Trends ───────────────────────────────────────
L("## 8. Criticized and Classified Asset Trends")
L("")
L("### Aggregate Trend (8 Quarters)")
L("")
L("| Quarter | Classified ($M) | Criticized ($M) | Special Mention ($M) | Classified / Total Loans | Criticized / Total Loans | Classified / Tier 1 Capital |")
L("|---|---:|---:|---:|---:|---:|---:|")
for q_idx, q in enumerate(quarters):
    # approximate total loans for that quarter
    tl = prior_total + (total_portfolio - prior_total) * (q_idx + 1) / 8
    L(f"| {q} | {dol(classified_trend_total[q_idx])} | {dol(criticized_trend_total[q_idx])} | {dol(special_mention_trend[q_idx])} | {pct(classified_trend_total[q_idx] / tl)} | {pct(criticized_trend_total[q_idx] / tl)} | {pct(classified_trend_total[q_idx] / tier1_capital)} |")
L("")

L("### Classified Assets by Segment (Current Quarter)")
L("")
L("| Segment | Classified ($M) | % of Segment | QoQ Change | Trend |")
L("|---|---:|---:|---:|:---:|")
for i, seg in enumerate(segments):
    class_amt = classified_ratio[i] * current_balances[i]
    # simulate QoQ change
    qoq_chg = np.random.uniform(-0.05, 0.12)
    trend = "Increasing" if qoq_chg > 0.02 else ("Stable" if qoq_chg > -0.02 else "Decreasing")
    L(f"| {seg} | {dol2(class_amt)} | {pct(classified_ratio[i])} | {pct(qoq_chg)} | {trend} |")
L(f"| **Total** | **{dol2(sum(classified_ratio[i] * current_balances[i] for i in range(len(segments))))}** | **{pct(wt_class)}** | — | — |")
L("")

L("The classified-to-Tier-1-Capital ratio has risen from {:.4f}% in Q1 2023 to".format(
    classified_trend_total[0] / tier1_capital * 100))
L("{:.4f}% in Q4 2024, an increase of {:.0f} basis points over eight quarters.".format(
    classified_trend_total[-1] / tier1_capital * 100,
    (classified_trend_total[-1] - classified_trend_total[0]) / tier1_capital * 10000))
L("If this trend continues, the ratio will breach the 20.0000% board-level trigger")
L("within two quarters.")
L("")
L("---")
L("")

# ── Section 9: Geographic Concentration ──────────────────────────────────────
L("## 9. Geographic Concentration")
L("")
L("| State/Region | Exposure ($M) | % of Total | 30+ DPD Rate | Internal Limit ($M) | Utilization | Status |")
L("|---|---:|---:|---:|---:|---:|:---:|")
for i, st in enumerate(states):
    util = geo_exposure[i] / geo_limits[i]
    status = "PASS" if util <= 1.0 else "**BREACH**"
    L(f"| {st} | {dol(geo_exposure[i])} | {pct(geo_exposure[i] / total_portfolio)} | {pct(geo_delinq[i])} | {dol(geo_limits[i])} | {pct(util)} | {status} |")
L(f"| **Total Top 10** | **{dol(geo_exposure.sum())}** | **{pct(geo_exposure.sum() / total_portfolio)}** | — | — | — | — |")
L("")

geo_breaches = [states[i] for i in range(len(states)) if geo_exposure[i] / geo_limits[i] > 1.0]
if geo_breaches:
    L(f"Geographic concentration breaches exist in: {', '.join(geo_breaches)}.")
    L("These breaches must be reported to the Board Risk Committee per Policy Section 5.1.7.")
L("")

L("### Delinquency Heat Map by State")
L("")
L("| State | Current 30+ DPD | Prior Quarter 30+ DPD | Change (bps) | Risk Tier |")
L("|---|---:|---:|---:|:---:|")
for i, st in enumerate(states):
    prior_d = geo_delinq[i] * np.random.uniform(0.88, 1.05)
    chg_bps = (geo_delinq[i] - prior_d) * 10000
    tier = "Low" if geo_delinq[i] < 0.015 else ("Medium" if geo_delinq[i] < 0.020 else "High")
    L(f"| {st} | {pct(geo_delinq[i])} | {pct(prior_d)} | {chg_bps:+.1f} | {tier} |")
L("")

L("---")
L("")

# ── Section 10: Peer Comparison ──────────────────────────────────────────────
L("## 10. Peer Comparison")
L("")
L("Comparison of key credit metrics against a peer group of similarly-sized commercial")
L("banks (total assets $15B–$30B) as of the most recent call report data (Q3 2024).")
L("")

# Generate peer data
metrics_peer = ["30+ DPD Rate", "NCO Rate", "Classified / Loans",
                "Criticized / Loans", "ACL Coverage", "CRE Concentration (% Capital)",
                "Loan Growth (YoY)", "NIM"]

bank_values = [wt_delinq30, wt_nco, wt_class, wt_crit, total_cov,
               (current_balances[1] + current_balances[2] + current_balances[3]) / tier1_capital,
               (total_portfolio - prior_total) / prior_total * 4,  # annualized
               0.0348]

L("| Metric | Pacific Continental | " + " | ".join(peers) + " | Peer Median | Percentile |")
L("|---|---:|" + "---:|" * len(peers) + "---:|---:|")

for m_idx, metric in enumerate(metrics_peer):
    bv = bank_values[m_idx]
    peer_vals = bv * (1 + np.random.uniform(-0.35, 0.35, len(peers)))
    peer_med = np.median(peer_vals)
    all_vals = np.append(peer_vals, bv)
    # For NCO and delinquency, lower is better → percentile = fraction below us
    # For ACL and NIM, higher is better
    if metric in ["ACL Coverage", "NIM"]:
        pctile = np.sum(all_vals <= bv) / len(all_vals)
    else:
        pctile = np.sum(all_vals >= bv) / len(all_vals)

    peer_strs = " | ".join(pct(pv) for pv in peer_vals)
    L(f"| {metric} | {pct(bv)} | {peer_strs} | {pct(peer_med)} | {pctile*100:.0f}th |")
L("")

L("> **Peer Positioning:** Pacific Continental ranks in the bottom quartile for NCO Rate,")
L("> driven primarily by the Consumer-Card and Consumer-Auto segments. The bank's CRE")
L("> concentration as a percentage of Tier 1 Capital is above the peer median, consistent")
L("> with its strategic focus on commercial real estate lending in West Coast markets.")
L("")
L("---")
L("")

# ── Section 11: Watch List ───────────────────────────────────────────────────
L("## 11. Watch List — Top 20 Credits")
L("")
L("| Rank | Borrower | Exposure ($M) | Risk Rating | Primary Concern | Trend |")
L("|---:|---|---:|:---:|---|:---:|")
for i in range(20):
    L(f"| {i+1} | {watch_names[i]} | {dol(watch_exposure[i])} | {watch_ratings[i]} | {watch_reasons[i]} | {watch_trends[i]} |")
L("")

total_watch = watch_exposure.sum()
L(f"**Total Watch List Exposure:** {dol(total_watch)} million ({pct(total_watch / total_portfolio)} of total loans)")
L("")
L("### Watch List Composition Summary")
L("")
L("| Risk Rating | Count | Total Exposure ($M) | % of Watch List | Avg Exposure ($M) |")
L("|---|---:|---:|---:|---:|")
for rating in ["Special Mention", "Substandard", "Doubtful"]:
    mask = watch_ratings == rating
    cnt = mask.sum()
    exp = watch_exposure[mask].sum()
    avg = exp / cnt if cnt > 0 else 0
    L(f"| {rating} | {cnt} | {dol(exp)} | {pct(exp / total_watch)} | {dol2(avg)} |")
L(f"| **Total** | **20** | **{dol(total_watch)}** | **100.0000%** | **{dol2(total_watch / 20)}** |")
L("")

det_count = np.sum(watch_trends == "Deteriorating")
L(f"Of the 20 watch list credits, {det_count} ({pct(det_count / 20)}) show a deteriorating")
L("trend. Management should prioritize updated appraisals and financial statements for")
L("these borrowers within 60 days.")
L("")
L("---")
L("")

# ── Section 12: Summary Findings ─────────────────────────────────────────────
L("## 12. Summary Findings and Policy Compliance")
L("")
L("### Risk Dimension Assessment")
L("")
L("| Risk Dimension | Policy Limit / Threshold | Current Value | Status | Commentary |")
L("|---|---|---:|:---:|---|")

# Portfolio growth
ann_growth = ((total_portfolio / prior_total)**4 - 1)
L(f"| Portfolio Growth (Annualized) | < 15.0000% | {pct(ann_growth)} | PASS | Within policy limits |")

# Overall delinquency
L(f"| Portfolio 30+ DPD Rate | < 3.0000% | {pct(wt_delinq30)} | PASS | Below threshold |")

# Overall NCO
L(f"| Portfolio NCO Rate | < 0.6000% | {pct(wt_nco)} | {'PASS' if wt_nco < 0.006 else '**FAIL**'} | {'Within limits' if wt_nco < 0.006 else 'Exceeded; driven by Consumer segments'} |")

# Classified / Total
L(f"| Classified / Total Loans | < 4.0000% | {pct(wt_class)} | {'PASS' if wt_class < 0.04 else '**FAIL**'} | — |")

# Criticized / Total
L(f"| Criticized / Total Loans | < 6.0000% | {pct(wt_crit)} | {'PASS' if wt_crit < 0.06 else '**FAIL**'} | — |")

# Classified / Tier 1
class_t1 = classified_trend_total[-1] / tier1_capital
L(f"| Classified / Tier 1 Capital | < 20.0000% | {pct(class_t1)} | PASS | Approaching trigger level |")

# CRE concentration
cre_total = current_balances[1] + current_balances[2] + current_balances[3]
L(f"| CRE Concentration / Tier 1 | < 300.0000% | {pct(cre_total / tier1_capital)} | PASS | Within regulatory guidance |")

# ACL adequacy
L(f"| ACL / Total Loans | > 1.5000% | {pct(total_cov)} | {'PASS' if total_cov > 0.015 else '**FAIL**'} | {'Adequate' if total_cov > 0.015 else 'Below minimum coverage'} |")

# Industry concentration breaches
L(f"| Industry Concentration Breaches | 0 | {len(breach_idx)} | **FAIL** | {len(breach_idx)} industries exceed internal limits |")

# Geographic breaches
L(f"| Geographic Concentration Breaches | 0 | {len(geo_breaches)} | {'PASS' if len(geo_breaches) == 0 else '**FAIL**'} | {len(geo_breaches)} state(s) exceed limits |")

# Migration
L(f"| Substandard → Loss Migration | < 5.0000% | {pct(migration[2,4])} | **FAIL** | Exceeds threshold; monitor quarterly |")

L("")

# Count pass/fail
L("### Compliance Summary")
L("")
pass_count = 0
fail_count = 0
dimensions = [
    ann_growth < 0.15,
    wt_delinq30 < 0.03,
    wt_nco < 0.006,
    wt_class < 0.04,
    wt_crit < 0.06,
    class_t1 < 0.20,
    cre_total / tier1_capital < 3.0,
    total_cov > 0.015,
    len(breach_idx) == 0,
    len(geo_breaches) == 0,
    migration[2,4] < 0.05,
]
pass_count = sum(dimensions)
fail_count = len(dimensions) - pass_count
L(f"- **Total Risk Dimensions Assessed:** {len(dimensions)}")
L(f"- **Passing:** {pass_count}")
L(f"- **Failing:** {fail_count}")
L(f"- **Overall Compliance Rating:** {'Satisfactory' if fail_count <= 2 else 'Needs Improvement' if fail_count <= 4 else 'Unsatisfactory'}")
L("")

L("### Required Management Actions")
L("")
L("1. **Industry Concentration Remediation (30 days):** Submit remediation plans for all")
L("   industry concentrations exceeding internal limits to the Board Risk Committee.")
L("2. **Consumer-Auto NCO Investigation (45 days):** Conduct a deep-dive analysis of")
L("   Consumer-Auto charge-off severity trends and collateral recovery rates. Update")
L("   loss-given-default assumptions in the ACL model.")
L("3. **CRE Non-Owner Occupied Review (60 days):** Perform targeted loan reviews on all")
L("   CRE office exposures originated in 2019-2020 vintages with current appraised")
L("   LTV above 75%.")
L("4. **ACL Segment Deficiency (30 days):** Evaluate qualitative factor overlays for")
L("   segments where coverage ratios fall below required minimums. Present updated")
L("   reserve methodology to the Audit Committee.")
L("5. **Watch List Monitoring (Ongoing):** Obtain updated financial statements and")
L("   appraisals for all deteriorating watch list credits within 60 days.")
L("6. **Migration Monitoring (Quarterly):** The Substandard-to-Loss migration rate of")
L(f"   {pct(migration[2,4])} exceeds the 5.0000% threshold. Implement enhanced quarterly")
L("   reviews of all Substandard-rated credits.")
L("")

L("---")
L("")
L("## Appendix A: Methodology Notes")
L("")
L("- **Delinquency Rates** are calculated as the aggregate principal balance of loans")
L("  past due by the stated number of days divided by the total segment balance.")
L("- **Net Charge-Off Rate** is annualized and represents gross charge-offs less")
L("  recoveries divided by average segment balances for the quarter.")
L("- **Classified Assets** include all loans rated Substandard, Doubtful, or Loss.")
L("- **Criticized Assets** include all Classified assets plus Special Mention.")
L("- **Coverage Ratio** is defined as ACL balance divided by total outstanding loans")
L("  for the segment.")
L("- **Migration probabilities** are computed as the proportion of loan balances")
L("  transitioning between risk rating categories during the quarter.")
L("- **Vintage loss rates** represent cumulative net losses from origination through")
L("  the report date, expressed as a percentage of original funded commitments.")
L("- **Peer data** sourced from FFIEC Call Report data (schedule RC-N and RC-R)")
L("  for banks with total assets between $15 billion and $30 billion.")
L("")
L("---")
L("")
L("## Appendix B: Definitions")
L("")
L("| Term | Definition |")
L("|---|---|")
L("| ACL | Allowance for Credit Losses (under CECL methodology) |")
L("| ALLL | Allowance for Loan and Lease Losses (legacy term) |")
L("| C&I | Commercial and Industrial |")
L("| CRE | Commercial Real Estate |")
L("| DPD | Days Past Due |")
L("| DSCR | Debt Service Coverage Ratio |")
L("| LTV | Loan-to-Value ratio |")
L("| MSA | Metropolitan Statistical Area |")
L("| NCO | Net Charge-Off |")
L("| NIM | Net Interest Margin |")
L("| OCC | Office of the Comptroller of the Currency |")
L("| Tier 1 Capital | Core equity capital as defined under Basel III |")
L("")
L("---")
L("")
L("*This report is intended for internal use by Pacific Continental Bank, N.A. and its")
L("regulators. Distribution outside the institution requires prior written approval from")
L("the Chief Risk Officer. All data is as of December 31, 2024 unless otherwise noted.*")

# ═══════════════════════════════════════════════════════════════════════════════
#  WRITE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
output = "\n".join(lines)
with open("/home/asudjianto/jupyterlab/kg-memory/finstructbench/instances/credit_portfolio.md", "w") as f:
    f.write(output)

print(f"Written {len(lines)} lines to credit_portfolio.md")
