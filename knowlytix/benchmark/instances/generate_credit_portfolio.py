#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate a synthetic Credit Portfolio Review Report (OCC guidelines)
for a regional commercial bank. Uses numpy with seed=321 for reproducibility.
Outputs markdown to finstructbench/instances/credit_portfolio.md
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(321)

# ── Configuration ──────────────────────────────────────────────────────────
BANK_NAME = "Ridgeline Community Bancshares, Inc."
REPORT_DATE = "September 30, 2025"
PRIOR_DATE = "September 30, 2024"
TOTAL_ASSETS_B = 12.4  # billions
TOTAL_LOANS_B = 8.73   # billions
TIER1_CAPITAL_B = 1.18

SEGMENTS = [
    "C&I",
    "CRE - Owner Occupied",
    "CRE - Non-Owner Occupied",
    "Construction & Land",
    "Residential 1-4 Family",
    "Home Equity",
    "Consumer - Auto",
    "Consumer - Other",
    "Agricultural",
]

INDUSTRIES = [
    "Healthcare & Social Assistance",
    "Manufacturing",
    "Retail Trade",
    "Real Estate & Leasing",
    "Accommodation & Food Services",
    "Professional Services",
    "Transportation & Warehousing",
    "Wholesale Trade",
    "Agriculture, Forestry & Fishing",
    "Construction",
]

GEOGRAPHIES = [
    "Tennessee",
    "North Carolina",
    "Virginia",
    "Georgia",
    "South Carolina",
    "Kentucky",
    "Alabama",
]

RISK_RATINGS = ["Pass", "Special Mention", "Substandard", "Doubtful", "Loss"]

ORIGINATION_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# ── Helper functions ───────────────────────────────────────────────────────

def fmt_pct(x, decimals=2):
    return f"{x:.{decimals}f}%"

def fmt_dollar(x, decimals=1):
    return f"${x:,.{decimals}f}"

def make_table(df, index=True):
    """Return pipe-separated markdown table via pandas."""
    return df.to_markdown(index=index)

# ── 1. Portfolio Composition ──────────────────────────────────────────────
def gen_portfolio_composition():
    """Generate portfolio balances by segment for current and prior period."""
    # Current period balances ($M) — hand-tuned with noise
    base_balances = np.array([
        1820,  # C&I
        1340,  # CRE OO
        1580,  # CRE NOO
        620,   # Construction
        1450,  # Resi 1-4
        680,   # HELOC
        540,   # Auto
        320,   # Consumer Other
        380,   # Ag
    ], dtype=float)
    noise = np.random.normal(0, 15, len(base_balances))
    current = np.round(base_balances + noise, 1)

    # Prior period: growth rates vary by segment (some growing, some shrinking)
    growth_rates = np.array([0.06, 0.03, 0.09, 0.14, 0.02, -0.01, 0.05, 0.01, -0.03])
    prior = np.round(current / (1 + growth_rates), 1)

    total_current = current.sum()
    total_prior = prior.sum()

    df = pd.DataFrame({
        "Segment": SEGMENTS,
        "Balance ($M) Current": current,
        "Balance ($M) Prior Year": prior,
        "Change ($M)": np.round(current - prior, 1),
        "Growth (%)": np.round(growth_rates * 100, 2),
        "% of Total Portfolio": np.round(current / total_current * 100, 1),
    })

    totals = pd.DataFrame({
        "Segment": ["Total"],
        "Balance ($M) Current": [round(total_current, 1)],
        "Balance ($M) Prior Year": [round(total_prior, 1)],
        "Change ($M)": [round(total_current - total_prior, 1)],
        "Growth (%)": [round((total_current / total_prior - 1) * 100, 2)],
        "% of Total Portfolio": [100.0],
    })
    df = pd.concat([df, totals], ignore_index=True)
    return df, current

# ── 2. Credit Quality Metrics ─────────────────────────────────────────────
def gen_credit_quality(balances):
    """Delinquency, NPL, and charge-off rates by segment."""
    # Delinquency rates — C&I and CRE NOO deteriorating, resi stable
    del_30 = np.array([1.82, 0.95, 2.41, 3.15, 1.10, 0.78, 2.05, 1.55, 0.92])
    del_60 = np.array([0.74, 0.38, 1.12, 1.68, 0.42, 0.31, 0.88, 0.64, 0.35])
    del_90 = np.array([0.41, 0.18, 0.72, 1.05, 0.21, 0.14, 0.53, 0.37, 0.19])

    # Add small random noise
    del_30 = np.round(del_30 + np.random.uniform(-0.05, 0.05, len(del_30)), 2)
    del_60 = np.round(del_60 + np.random.uniform(-0.03, 0.03, len(del_60)), 2)
    del_90 = np.round(del_90 + np.random.uniform(-0.02, 0.02, len(del_90)), 2)

    # NPL rate (>= 90 days + nonaccrual)
    npl_rate = np.round(del_90 + np.random.uniform(0.1, 0.4, len(del_90)), 2)

    # Charge-off rates — Construction & CRE NOO elevated
    chargeoff = np.array([0.32, 0.12, 0.58, 0.85, 0.08, 0.05, 0.41, 0.28, 0.11])
    chargeoff = np.round(chargeoff + np.random.uniform(-0.03, 0.03, len(chargeoff)), 2)

    # NPL dollar amounts
    npl_dollars = np.round(balances * npl_rate / 100, 1)

    # Prior-year NPL for comparison (some worse, some better)
    npl_rate_prior = np.array([0.38, 0.22, 0.45, 0.62, 0.24, 0.18, 0.40, 0.35, 0.20])

    df = pd.DataFrame({
        "Segment": SEGMENTS,
        "30-Day Delinq (%)": del_30,
        "60-Day Delinq (%)": del_60,
        "90+ Day Delinq (%)": del_90,
        "NPL Rate (%) Current": npl_rate,
        "NPL Rate (%) Prior Yr": npl_rate_prior,
        "NPL Balance ($M)": npl_dollars,
        "Net Charge-Off Rate (%)": chargeoff,
    })

    return df, npl_rate, npl_dollars

# ── 3. Risk Rating Distribution ───────────────────────────────────────────
def gen_risk_rating_distribution(balances):
    """Assign risk ratings by segment — some segments have elevated classified."""
    # Pass percentages — lower for distressed segments
    pass_pct = np.array([84.2, 91.5, 78.3, 72.1, 93.4, 95.1, 86.7, 89.0, 90.8])
    sm_pct   = np.array([7.5,  4.2,  9.1, 11.8,  3.2,  2.5,  6.1,  5.2,  4.5])
    sub_pct  = np.array([6.1,  3.1,  9.8, 12.0,  2.4,  1.8,  5.3,  4.1,  3.5])
    doubt_pct = np.array([1.8,  1.0,  2.4,  3.5,  0.8,  0.5,  1.5,  1.3,  1.0])
    loss_pct = np.array([0.4,  0.2,  0.4,  0.6,  0.2,  0.1,  0.4,  0.4,  0.2])

    # Normalize so rows sum to 100
    for i in range(len(SEGMENTS)):
        total = pass_pct[i] + sm_pct[i] + sub_pct[i] + doubt_pct[i] + loss_pct[i]
        pass_pct[i] = round(pass_pct[i] / total * 100, 1)
        sm_pct[i] = round(sm_pct[i] / total * 100, 1)
        sub_pct[i] = round(sub_pct[i] / total * 100, 1)
        doubt_pct[i] = round(doubt_pct[i] / total * 100, 1)
        loss_pct[i] = round(100 - pass_pct[i] - sm_pct[i] - sub_pct[i] - doubt_pct[i], 1)

    # Dollar amounts
    pass_amt = np.round(balances * pass_pct / 100, 1)
    sm_amt   = np.round(balances * sm_pct / 100, 1)
    sub_amt  = np.round(balances * sub_pct / 100, 1)
    doubt_amt = np.round(balances * doubt_pct / 100, 1)
    loss_amt = np.round(balances * loss_pct / 100, 1)

    # Percentage table
    df_pct = pd.DataFrame({
        "Segment": SEGMENTS,
        "Pass (%)": pass_pct,
        "Special Mention (%)": sm_pct,
        "Substandard (%)": sub_pct,
        "Doubtful (%)": doubt_pct,
        "Loss (%)": loss_pct,
    })

    # Dollar table
    df_amt = pd.DataFrame({
        "Segment": SEGMENTS,
        "Pass ($M)": pass_amt,
        "Special Mention ($M)": sm_amt,
        "Substandard ($M)": sub_amt,
        "Doubtful ($M)": doubt_amt,
        "Loss ($M)": loss_amt,
        "Total ($M)": np.round(pass_amt + sm_amt + sub_amt + doubt_amt + loss_amt, 1),
    })

    classified_total = np.round(sub_amt + doubt_amt + loss_amt, 1)
    classified_pct = np.round((sub_pct + doubt_pct + loss_pct), 1)

    return df_pct, df_amt, classified_total, classified_pct

# ── 4. Concentration Analysis ─────────────────────────────────────────────
def gen_concentration_analysis(total_loans_m):
    """Industry, geographic, and single-borrower concentrations."""
    # Industry concentrations
    ind_pct = np.array([14.8, 12.3, 11.1, 10.5, 9.2, 8.7, 7.4, 6.8, 5.9, 5.1])
    # Add noise and re-sort
    ind_pct = ind_pct + np.random.uniform(-0.3, 0.3, len(ind_pct))
    ind_pct = np.round(ind_pct, 1)
    other_pct = round(100 - ind_pct.sum(), 1)
    ind_bal = np.round(total_loans_m * ind_pct / 100, 1)
    other_bal = round(total_loans_m * other_pct / 100, 1)

    # Policy limit for any single industry = 20% of total loans
    policy_limit = 20.0
    limit_status = ["Within Limit" if p < policy_limit else "EXCEEDS LIMIT" for p in ind_pct]

    df_ind = pd.DataFrame({
        "Industry": INDUSTRIES + ["All Other"],
        "Balance ($M)": list(ind_bal) + [other_bal],
        "% of Portfolio": list(ind_pct) + [other_pct],
        "Policy Limit (%)": [policy_limit] * len(INDUSTRIES) + [policy_limit],
        "Status": limit_status + ["Within Limit"],
    })

    # Geographic concentrations
    geo_pct = np.array([32.5, 22.1, 15.8, 12.4, 8.7, 5.3, 3.2])
    geo_pct = np.round(geo_pct + np.random.uniform(-0.5, 0.5, len(geo_pct)), 1)
    geo_bal = np.round(total_loans_m * geo_pct / 100, 1)

    # CRE concentration (300% of Tier 1 Capital is supervisory threshold)
    cre_total = 1580 + 1340 + 620  # CRE NOO + CRE OO + Construction
    cre_pct_tier1 = round(cre_total / (TIER1_CAPITAL_B * 1000) * 100, 1)

    df_geo = pd.DataFrame({
        "State": GEOGRAPHIES,
        "Balance ($M)": geo_bal,
        "% of Portfolio": geo_pct,
    })

    # Top 10 borrower concentrations
    top_borrowers = [
        "Appalachian Health Systems",
        "Blue Ridge Manufacturing Co.",
        "Southeastern Retail Partners",
        "Piedmont Real Estate Holdings",
        "Mountain View Hotels LLC",
        "Carolina Logistics Group",
        "Valley Agricultural Co-op",
        "Ridgeline Auto Dealers Inc.",
        "Southern Professional Plaza",
        "Heritage Construction Group",
    ]
    top_bal = np.round(np.array([185, 142, 128, 115, 98, 87, 76, 68, 62, 55], dtype=float)
                       + np.random.uniform(-5, 5, 10), 1)
    top_pct_capital = np.round(top_bal / (TIER1_CAPITAL_B * 1000) * 100, 1)
    # Legal lending limit = 15% of Tier 1 capital
    lll = round(TIER1_CAPITAL_B * 1000 * 0.15, 1)
    lll_status = ["Within Limit" if b < lll else "EXCEEDS LIMIT" for b in top_bal]

    df_borrowers = pd.DataFrame({
        "Borrower": top_borrowers,
        "Outstanding ($M)": top_bal,
        "% of Tier 1 Capital": top_pct_capital,
        "Legal Lending Limit ($M)": [lll] * 10,
        "Status": lll_status,
    })

    return df_ind, df_geo, df_borrowers, cre_pct_tier1

# ── 5. Allowance for Credit Losses ────────────────────────────────────────
def gen_acl(balances, npl_rate, npl_dollars):
    """ACL adequacy by segment."""
    # ACL as % of segment balance — higher for riskier segments
    acl_pct = np.array([1.65, 1.20, 2.10, 2.85, 0.72, 0.55, 1.90, 1.45, 0.95])
    acl_pct = np.round(acl_pct + np.random.uniform(-0.05, 0.05, len(acl_pct)), 2)

    acl_balance = np.round(balances * acl_pct / 100, 1)

    # Coverage ratio = ACL / NPL balance
    coverage = np.where(npl_dollars > 0,
                        np.round(acl_balance / npl_dollars * 100, 1),
                        999.9)

    # Peer comparison ACL rate
    peer_acl = np.array([1.50, 1.15, 1.80, 2.50, 0.68, 0.50, 1.70, 1.35, 0.90])

    total_acl = round(acl_balance.sum(), 1)
    total_loans = round(balances.sum(), 1)
    total_npl = round(npl_dollars.sum(), 1)

    df = pd.DataFrame({
        "Segment": SEGMENTS,
        "Segment Balance ($M)": balances,
        "ACL Balance ($M)": acl_balance,
        "ACL / Loans (%)": acl_pct,
        "Peer ACL / Loans (%)": peer_acl,
        "NPL Balance ($M)": npl_dollars,
        "Coverage Ratio (%)": coverage,
    })

    totals = pd.DataFrame({
        "Segment": ["Total"],
        "Segment Balance ($M)": [total_loans],
        "ACL Balance ($M)": [total_acl],
        "ACL / Loans (%)": [round(total_acl / total_loans * 100, 2)],
        "Peer ACL / Loans (%)": ["--"],
        "NPL Balance ($M)": [total_npl],
        "Coverage Ratio (%)": [round(total_acl / total_npl * 100, 1) if total_npl > 0 else 999.9],
    })
    df = pd.concat([df, totals], ignore_index=True)
    return df, total_acl

# ── 6. Vintage Analysis ──────────────────────────────────────────────────
def gen_vintage_analysis(balances):
    """Performance by origination year."""
    # For each segment pick a few representative vintages
    # We'll show segment x vintage grid of cumulative loss rates
    rows = []
    for i, seg in enumerate(SEGMENTS):
        base_loss = np.array([0.85, 0.42, 0.28, 0.55, 1.10, 0.35, 0.15]) * (1 + i * 0.08)
        # 2020 and 2021 vintages underwritten during COVID — weaker
        base_loss[1] *= 1.6  # 2020
        base_loss[2] *= 1.3  # 2021
        # Construction segment has elevated recent vintages
        if seg == "Construction & Land":
            base_loss[4] *= 1.8
            base_loss[5] *= 1.5
        # CRE NOO also deteriorating
        if seg == "CRE - Non-Owner Occupied":
            base_loss[3] *= 1.4
            base_loss[4] *= 1.6

        base_loss = np.round(base_loss + np.random.uniform(-0.03, 0.03, len(base_loss)), 2)
        base_loss = np.clip(base_loss, 0.01, None)

        row = {"Segment": seg}
        for j, yr in enumerate(ORIGINATION_YEARS):
            row[str(yr)] = base_loss[j]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

# ── 7. Migration Analysis ────────────────────────────────────────────────
def gen_migration_analysis():
    """Risk rating migration period-over-period."""
    # Transition matrix: from (rows) -> to (columns)
    # Columns: Pass, SM, Sub, Doubtful, Loss
    migration = np.array([
        [91.2,  5.1,  2.8,  0.7, 0.2],  # from Pass
        [18.5, 52.3, 22.1,  5.8, 1.3],  # from Special Mention
        [ 5.2, 12.8, 55.4, 20.3, 6.3],  # from Substandard
        [ 1.1,  2.5, 10.2, 58.7, 27.5], # from Doubtful
        [ 0.0,  0.0,  0.0,  5.0, 95.0], # from Loss
    ])
    # Add small noise and re-normalize rows
    noise = np.random.uniform(-0.3, 0.3, migration.shape)
    migration = migration + noise
    migration = np.clip(migration, 0, None)
    for i in range(migration.shape[0]):
        migration[i] = np.round(migration[i] / migration[i].sum() * 100, 1)
        # Fix rounding to 100
        migration[i][-1] = round(100 - migration[i][:-1].sum(), 1)

    df = pd.DataFrame(migration, columns=[f"To {r}" for r in RISK_RATINGS])
    df.insert(0, "From Rating", RISK_RATINGS)

    # Upgrade / downgrade counts
    upgrades = 342 + int(np.random.randint(-20, 20))
    downgrades = 518 + int(np.random.randint(-20, 20))
    unchanged = 4215 + int(np.random.randint(-50, 50))

    return df, upgrades, downgrades, unchanged

# ── 8. Stress Scenario Loss Projections ───────────────────────────────────
def gen_stress_projections(balances):
    """Loss rates under baseline, adverse, and severely adverse scenarios."""
    # Baseline loss rates
    base_loss = np.array([0.35, 0.18, 0.65, 0.95, 0.12, 0.08, 0.52, 0.38, 0.14])
    adv_loss  = np.array([1.20, 0.75, 2.80, 4.50, 0.55, 0.35, 1.85, 1.30, 0.60])
    sev_loss  = np.array([2.45, 1.60, 5.20, 8.10, 1.25, 0.82, 3.60, 2.55, 1.20])

    # Add noise
    base_loss = np.round(base_loss + np.random.uniform(-0.02, 0.02, len(base_loss)), 2)
    adv_loss  = np.round(adv_loss + np.random.uniform(-0.1, 0.1, len(adv_loss)), 2)
    sev_loss  = np.round(sev_loss + np.random.uniform(-0.2, 0.2, len(sev_loss)), 2)

    base_dollars = np.round(balances * base_loss / 100, 1)
    adv_dollars  = np.round(balances * adv_loss / 100, 1)
    sev_dollars  = np.round(balances * sev_loss / 100, 1)

    df = pd.DataFrame({
        "Segment": SEGMENTS,
        "Balance ($M)": balances,
        "Baseline Loss (%)": base_loss,
        "Baseline Loss ($M)": base_dollars,
        "Adverse Loss (%)": adv_loss,
        "Adverse Loss ($M)": adv_dollars,
        "Severely Adverse Loss (%)": sev_loss,
        "Severely Adverse Loss ($M)": sev_dollars,
    })

    totals = pd.DataFrame({
        "Segment": ["Total"],
        "Balance ($M)": [round(balances.sum(), 1)],
        "Baseline Loss (%)": [round(base_dollars.sum() / balances.sum() * 100, 2)],
        "Baseline Loss ($M)": [round(base_dollars.sum(), 1)],
        "Adverse Loss (%)": [round(adv_dollars.sum() / balances.sum() * 100, 2)],
        "Adverse Loss ($M)": [round(adv_dollars.sum(), 1)],
        "Severely Adverse Loss (%)": [round(sev_dollars.sum() / balances.sum() * 100, 2)],
        "Severely Adverse Loss ($M)": [round(sev_dollars.sum(), 1)],
    })
    df = pd.concat([df, totals], ignore_index=True)
    return df, base_dollars.sum(), adv_dollars.sum(), sev_dollars.sum()


# ══════════════════════════════════════════════════════════════════════════
# BUILD THE REPORT
# ══════════════════════════════════════════════════════════════════════════

def build_report():
    lines = []

    # ── Title and Metadata ─────────────────────────────────────────────
    lines.append("# Credit Portfolio Review Report")
    lines.append("")
    lines.append("## Report Overview")
    lines.append("")
    lines.append(f"- **Institution:** {BANK_NAME}")
    lines.append(f"- **Report Date:** {REPORT_DATE}")
    lines.append(f"- **Prior Comparison Date:** {PRIOR_DATE}")
    lines.append(f"- **Total Assets:** ${TOTAL_ASSETS_B:.1f} billion")
    lines.append(f"- **Total Loans & Leases:** ${TOTAL_LOANS_B:.2f} billion")
    lines.append(f"- **Tier 1 Capital:** ${TIER1_CAPITAL_B:.2f} billion")
    lines.append("- **Regulatory Framework:** OCC Comptroller's Handbook - Credit Portfolio Management")
    lines.append("- **Examination Type:** Full-scope credit review")
    lines.append("- **CAMELS Component Assessed:** Asset Quality")
    lines.append("- **Prepared By:** Chief Credit Officer / Credit Risk Management Division")
    lines.append("")

    # ── Executive Summary ──────────────────────────────────────────────
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("The credit portfolio of Ridgeline Community Bancshares reflects a mixed "
                 "risk profile as of the review date. While the residential mortgage and "
                 "home equity segments continue to demonstrate stable performance with low "
                 "delinquency and adequate reserve coverage, the Construction & Land and "
                 "CRE Non-Owner Occupied segments show material deterioration in credit "
                 "quality metrics. Classified assets in these segments have increased "
                 "significantly period-over-period, driven by softening commercial real "
                 "estate valuations and project completion delays in the southeastern "
                 "construction market.")
    lines.append("")
    lines.append("Key concerns identified during this review include:")
    lines.append("")
    lines.append("- **Construction & Land segment** classified ratio of 16.1% exceeds the "
                 "institution's internal threshold of 10%")
    lines.append("- **CRE Non-Owner Occupied** net charge-off rate has increased 38 bps "
                 "year-over-year")
    lines.append("- **CRE concentration** relative to Tier 1 capital warrants ongoing "
                 "monitoring under interagency guidance")
    lines.append("- **2020 and 2021 vintage loans** across multiple segments exhibit "
                 "elevated cumulative loss rates attributable to pandemic-era underwriting")
    lines.append("")
    lines.append("Positively, the Residential 1-4 Family and Home Equity portfolios "
                 "maintain strong credit discipline with NPL rates below 0.60% and "
                 "coverage ratios exceeding 200%. The C&I segment, while showing modest "
                 "credit migration, remains within acceptable risk parameters.")
    lines.append("")

    # ── Section 1: Portfolio Composition ───────────────────────────────
    lines.append("## 1. Portfolio Composition by Loan Type")
    lines.append("")
    df_comp, balances = gen_portfolio_composition()
    lines.append(make_table(df_comp, index=False))
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")
    lines.append("- **Largest segment:** C&I loans represent the highest concentration "
                 "at approximately 21% of total loans")
    lines.append("- **Fastest growing:** Construction & Land grew 14.0% year-over-year, "
                 "reflecting regional development activity")
    lines.append("- **Declining segments:** Home Equity (-1.0%) and Agricultural (-3.0%) "
                 "portfolios contracted modestly")
    lines.append("- **CRE concentration (OO + NOO + Construction):** combined $3,540M "
                 "representing approximately 41% of total loans")
    lines.append("")

    # ── Section 2: Credit Quality ─────────────────────────────────────
    lines.append("## 2. Credit Quality Metrics")
    lines.append("")
    lines.append("### 2.1 Delinquency and Non-Performing Loans")
    lines.append("")
    df_cq, npl_rate, npl_dollars = gen_credit_quality(balances)
    lines.append(make_table(df_cq, index=False))
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")
    lines.append("- **Construction & Land** exhibits the highest delinquency across all "
                 "aging buckets (30-day: 3.15%, 90+: 1.05%), consistent with project "
                 "completion delays")
    lines.append("- **CRE Non-Owner Occupied** 90+ day delinquency of 0.72% is "
                 "approximately 3.4x the rate observed in CRE Owner Occupied")
    lines.append("- **Residential 1-4 Family** and **Home Equity** maintain delinquency "
                 "rates well below peer medians")
    lines.append("- **Consumer Auto** charge-off rate of ~0.41% is elevated relative to "
                 "the 0.30% peer benchmark but within acceptable range")
    lines.append("- NPL rates in Construction & Land and CRE NOO increased materially "
                 "versus prior year, while Residential improved")
    lines.append("")

    # ── Section 3: Risk Rating Distribution ───────────────────────────
    lines.append("## 3. Risk Rating Distribution")
    lines.append("")
    lines.append("### 3.1 Distribution by Percentage")
    lines.append("")
    df_rr_pct, df_rr_amt, classified_total, classified_pct = gen_risk_rating_distribution(balances)
    lines.append(make_table(df_rr_pct, index=False))
    lines.append("")
    lines.append("### 3.2 Distribution by Dollar Amount ($M)")
    lines.append("")
    lines.append(make_table(df_rr_amt, index=False))
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")
    lines.append("- **Total classified assets (Substandard + Doubtful + Loss):** "
                 f"${classified_total.sum():,.1f}M")
    lines.append(f"- **Classified / Tier 1 Capital:** "
                 f"{classified_total.sum() / (TIER1_CAPITAL_B * 1000) * 100:.1f}%")
    lines.append("- **Construction & Land** has the highest classified ratio at "
                 f"{classified_pct[3]:.1f}% of segment balance")
    lines.append("- **Residential 1-4 Family** classified ratio of "
                 f"{classified_pct[4]:.1f}% reflects strong underwriting standards")
    lines.append("- Doubtful-rated assets in Construction & Land increased $8.2M "
                 "quarter-over-quarter, requiring enhanced monitoring")
    lines.append("")

    # ── Section 4: Concentration Analysis ─────────────────────────────
    lines.append("## 4. Concentration Analysis")
    lines.append("")
    df_ind, df_geo, df_borr, cre_pct_tier1 = gen_concentration_analysis(balances.sum())
    lines.append("### 4.1 Industry Concentrations")
    lines.append("")
    lines.append(make_table(df_ind, index=False))
    lines.append("")
    lines.append("### 4.2 Geographic Concentrations")
    lines.append("")
    lines.append(make_table(df_geo, index=False))
    lines.append("")
    lines.append("### 4.3 Top 10 Borrower Concentrations")
    lines.append("")
    lines.append(make_table(df_borr, index=False))
    lines.append("")
    lines.append("### 4.4 CRE Concentration Assessment")
    lines.append("")
    lines.append("- **Total CRE Exposure (OO + NOO + Construction):** $3,540M")
    lines.append(f"- **CRE / Tier 1 Capital:** {cre_pct_tier1:.1f}%")
    lines.append("- **Supervisory Threshold (Interagency CRE Guidance):** 300%")
    threshold_status = "EXCEEDS" if cre_pct_tier1 > 300 else "Below"
    lines.append(f"- **Status:** {threshold_status} supervisory threshold")
    lines.append("- **Construction & Land / Tier 1 Capital:** "
                 f"{620 / (TIER1_CAPITAL_B * 1000) * 100:.1f}% "
                 "(supervisory threshold: 100%)")
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")
    lines.append("- All individual industry concentrations are within the 20% policy limit")
    lines.append("- Geographic concentration is heavily weighted toward Tennessee (32.5%) "
                 "and North Carolina (22.1%), creating regional economic risk")
    lines.append(f"- CRE concentration of {cre_pct_tier1:.1f}% of Tier 1 Capital "
                 "approaches the 300% interagency guideline threshold")
    lines.append("- Top single borrower (Appalachian Health Systems) represents "
                 "approximately 15.7% of Tier 1 Capital")
    lines.append("")

    # ── Section 5: Allowance for Credit Losses ────────────────────────
    lines.append("## 5. Allowance for Credit Losses (ACL) Adequacy")
    lines.append("")
    df_acl, total_acl = gen_acl(balances, npl_rate, npl_dollars)
    lines.append(make_table(df_acl, index=False))
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")
    lines.append(f"- **Total ACL:** ${total_acl:,.1f}M representing "
                 f"{total_acl / balances.sum() * 100:.2f}% of total loans")
    lines.append("- **Construction & Land** ACL/Loans ratio of ~2.85% is the highest, "
                 "reflecting elevated segment risk")
    lines.append("- **CRE Non-Owner Occupied** ACL coverage exceeds peer benchmark "
                 "by approximately 30 bps, considered appropriate given deteriorating "
                 "conditions")
    lines.append("- **Residential 1-4 Family** and **Home Equity** ACL rates are "
                 "in line with peer levels")
    lines.append("- Coverage ratio (ACL/NPL) for the total portfolio is adequate; "
                 "however, Construction & Land coverage warrants review given increasing "
                 "NPL formation")
    lines.append("")

    # ── Section 6: Vintage Analysis ───────────────────────────────────
    lines.append("## 6. Vintage Analysis: Cumulative Loss Rate (%) by Origination Year")
    lines.append("")
    df_vintage = gen_vintage_analysis(balances)
    lines.append(make_table(df_vintage, index=False))
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")
    lines.append("- **2020 vintages** exhibit systematically elevated loss rates across "
                 "most segments, reflecting pandemic-era underwriting accommodations")
    lines.append("- **2021 vintages** also show above-trend losses, particularly in "
                 "C&I and Consumer segments")
    lines.append("- **Construction & Land 2023 vintage** shows the single highest "
                 "loss rate, attributable to aggressive growth during the period")
    lines.append("- **2024-2025 originations** are still early in their performance "
                 "cycle; initial indications are in line with expectations")
    lines.append("- The residential portfolio demonstrates consistent, low loss rates "
                 "across all vintages, confirming the strength of underwriting standards")
    lines.append("")

    # ── Section 7: Migration Analysis ─────────────────────────────────
    lines.append("## 7. Risk Rating Migration Analysis")
    lines.append("")
    lines.append("### 7.1 Transition Matrix (12-Month Period)")
    lines.append("")
    lines.append("Values represent the percentage of loans in each starting rating "
                 "category that migrated to the destination rating over the review period.")
    lines.append("")
    df_mig, upgrades, downgrades, unchanged = gen_migration_analysis()
    lines.append(make_table(df_mig, index=False))
    lines.append("")
    lines.append("### 7.2 Migration Summary")
    lines.append("")
    lines.append(f"- **Total Rated Relationships:** {upgrades + downgrades + unchanged:,}")
    lines.append(f"- **Upgrades (improved rating):** {upgrades:,} "
                 f"({upgrades / (upgrades + downgrades + unchanged) * 100:.1f}%)")
    lines.append(f"- **Downgrades (deteriorated rating):** {downgrades:,} "
                 f"({downgrades / (upgrades + downgrades + unchanged) * 100:.1f}%)")
    lines.append(f"- **Unchanged:** {unchanged:,} "
                 f"({unchanged / (upgrades + downgrades + unchanged) * 100:.1f}%)")
    lines.append(f"- **Downgrade-to-Upgrade Ratio:** "
                 f"{downgrades / upgrades:.2f}x")
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")
    lines.append("- The downgrade-to-upgrade ratio exceeds 1.0x, indicating net negative "
                 "credit migration during the review period")
    lines.append("- 91.2% of Pass-rated loans retained their rating, in line with "
                 "historical norms")
    lines.append("- Special Mention loans have a 22.1% probability of further downgrade "
                 "to Substandard, highlighting the importance of early intervention")
    lines.append("- Doubtful-rated loans show a 27.5% loss realization rate, consistent "
                 "with regulatory expectations for this classification")
    lines.append("")

    # ── Section 8: Stress Scenario Loss Projections ───────────────────
    lines.append("## 8. Stress Scenario Loss Projections")
    lines.append("")
    df_stress, base_total, adv_total, sev_total = gen_stress_projections(balances)
    lines.append(make_table(df_stress, index=False))
    lines.append("")
    lines.append("### Scenario Assumptions")
    lines.append("")
    lines.append("- **Baseline:** Continued modest economic growth, unemployment "
                 "stable at 4.0%, HPI appreciation of 2.0%")
    lines.append("- **Adverse:** Mild recession, unemployment rising to 7.0%, "
                 "HPI decline of -10%, BBB spreads widening to 400 bps")
    lines.append("- **Severely Adverse:** Deep recession, unemployment at 10.0%, "
                 "HPI decline of -20%, BBB spreads at 600+ bps")
    lines.append("")
    lines.append("### Key Observations")
    lines.append("")
    lines.append(f"- **Baseline projected losses:** ${base_total:,.1f}M "
                 f"({base_total / balances.sum() * 100:.2f}% of portfolio)")
    lines.append(f"- **Adverse projected losses:** ${adv_total:,.1f}M "
                 f"({adv_total / balances.sum() * 100:.2f}% of portfolio)")
    lines.append(f"- **Severely Adverse projected losses:** ${sev_total:,.1f}M "
                 f"({sev_total / balances.sum() * 100:.2f}% of portfolio)")
    lines.append("- **Construction & Land** accounts for the largest share of "
                 "severely adverse losses at an 8.1% projected loss rate")
    lines.append("- Under the severely adverse scenario, projected losses would "
                 f"consume approximately {sev_total / (TIER1_CAPITAL_B * 1000) * 100:.1f}% "
                 "of Tier 1 Capital")
    lines.append("- The current ACL would be insufficient to absorb adverse-scenario "
                 "losses without additional provisioning")
    lines.append("")

    # ── Regulatory Findings ───────────────────────────────────────────
    lines.append("## 9. Regulatory Findings and Recommendations")
    lines.append("")
    lines.append("### Matters Requiring Attention (MRA)")
    lines.append("")
    lines.append("1. **Construction & Land Portfolio Oversight:** Classified assets "
                 "in the Construction & Land segment have increased to 16.1% of "
                 "segment balance. Management should implement enhanced monitoring "
                 "procedures, including quarterly site inspections and updated "
                 "appraisals for all relationships exceeding $5M.")
    lines.append("")
    lines.append("2. **CRE Concentration Management:** The combined CRE-to-Tier 1 "
                 f"Capital ratio of {cre_pct_tier1:.1f}% approaches the 300% "
                 "supervisory threshold. A formal concentration risk management "
                 "plan with defined limits, triggers, and board reporting is required.")
    lines.append("")
    lines.append("3. **Vintage-Specific Review:** The elevated loss rates in 2020-2021 "
                 "vintage loans warrant a targeted review of remaining exposures, "
                 "with re-underwriting of relationships exceeding $2M.")
    lines.append("")
    lines.append("### Positive Findings")
    lines.append("")
    lines.append("- Residential and Home Equity portfolios demonstrate strong "
                 "underwriting discipline with consistently low delinquency and loss metrics")
    lines.append("- ACL methodology and governance framework are consistent with "
                 "CECL requirements and FASB ASC 326")
    lines.append("- Single-borrower concentrations are within legal lending limits")
    lines.append("- Industry diversification is adequate with no individual sector "
                 "exceeding the 20% policy limit")
    lines.append("")

    # ── Appendix ──────────────────────────────────────────────────────
    lines.append("## Appendix A: Definitions")
    lines.append("")
    lines.append("- **C&I:** Commercial and Industrial loans")
    lines.append("- **CRE:** Commercial Real Estate")
    lines.append("- **NPL:** Non-Performing Loans (90+ days past due and/or nonaccrual)")
    lines.append("- **ACL:** Allowance for Credit Losses (CECL-based)")
    lines.append("- **Pass:** Risk rating 1-4; acceptable credit quality")
    lines.append("- **Special Mention:** Risk rating 5; potential weakness")
    lines.append("- **Substandard:** Risk rating 6; well-defined weakness")
    lines.append("- **Doubtful:** Risk rating 7; collection in full is questionable")
    lines.append("- **Loss:** Risk rating 8; uncollectible; to be charged off")
    lines.append("- **OCC:** Office of the Comptroller of the Currency")
    lines.append("- **CECL:** Current Expected Credit Losses")
    lines.append("- **HPI:** House Price Index")
    lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    report = build_report()
    out_path = Path(__file__).parent / "credit_portfolio.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Wrote {out_path}")
