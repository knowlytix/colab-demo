#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate a synthetic CCAR/DFAST-style stress test report for a mid-size bank.
Uses numpy with seed=123 for reproducibility.
"""

import numpy as np

np.random.seed(123)

# ── Configuration ──────────────────────────────────────────────────────────
BANK_NAME = "Heartland Commercial Bancorp, Inc."
REPORT_DATE = "December 31, 2025"
TOTAL_ASSETS = 48.7  # $B
QUARTERS = [f"Q{((i)%4)+1} {2026 + (i)//4}" for i in range(9)]
SCENARIOS = ["Baseline", "Adverse", "Severely Adverse"]

# ── Macro variables ────────────────────────────────────────────────────────
def gen_macro():
    """Generate macro scenario paths."""
    base_gdp = np.array([2.1, 2.0, 1.9, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3])
    adv_gdp  = np.array([0.5, -0.8, -1.6, -2.1, -1.4, -0.3, 0.4, 1.0, 1.5])
    sev_gdp  = np.array([-0.3, -2.5, -4.1, -5.3, -4.0, -2.2, -0.8, 0.3, 1.1])

    base_unemp = np.array([4.1, 4.0, 4.0, 3.9, 3.9, 3.8, 3.8, 3.7, 3.7])
    adv_unemp  = np.array([4.5, 5.2, 6.1, 7.0, 7.5, 7.8, 7.6, 7.3, 6.9])
    sev_unemp  = np.array([4.8, 6.0, 7.6, 9.1, 10.0, 10.4, 10.1, 9.5, 8.8])

    base_hpi = np.array([2.5, 2.3, 2.1, 1.8, 1.9, 2.0, 2.2, 2.4, 2.5])
    adv_hpi  = np.array([0.3, -3.5, -7.2, -10.8, -12.1, -10.5, -7.3, -4.1, -1.5])
    sev_hpi  = np.array([-1.2, -6.8, -13.5, -19.2, -22.4, -20.1, -15.3, -9.8, -5.2])

    base_tsy = np.array([4.25, 4.20, 4.15, 4.10, 4.05, 4.00, 3.95, 3.90, 3.85])
    adv_tsy  = np.array([3.80, 3.40, 2.95, 2.50, 2.30, 2.20, 2.35, 2.55, 2.80])
    sev_tsy  = np.array([3.50, 2.80, 2.10, 1.50, 1.10, 0.90, 1.05, 1.30, 1.65])

    base_bbb = np.array([1.45, 1.42, 1.40, 1.38, 1.36, 1.35, 1.33, 1.32, 1.30])
    adv_bbb  = np.array([1.80, 2.45, 3.20, 3.85, 4.10, 3.90, 3.50, 3.05, 2.65])
    sev_bbb  = np.array([2.20, 3.50, 4.80, 5.90, 6.40, 6.10, 5.45, 4.70, 3.95])

    return {
        "GDP Growth (% annualized)": {"Baseline": base_gdp, "Adverse": adv_gdp, "Severely Adverse": sev_gdp},
        "Unemployment Rate (%)": {"Baseline": base_unemp, "Adverse": adv_unemp, "Severely Adverse": sev_unemp},
        "House Price Index (% YoY)": {"Baseline": base_hpi, "Adverse": adv_hpi, "Severely Adverse": sev_hpi},
        "10-Year Treasury Yield (%)": {"Baseline": base_tsy, "Adverse": adv_tsy, "Severely Adverse": sev_tsy},
        "BBB Corporate Spread (%)": {"Baseline": base_bbb, "Adverse": adv_bbb, "Severely Adverse": sev_bbb},
    }

# ── Loan portfolio (starting balances $M) ─────────────────────────────────
SEGMENTS = ["C&I", "CRE", "Residential Mortgage", "Consumer", "Credit Card"]
STARTING_BALANCES = np.array([12400.0, 9800.0, 14200.0, 6300.0, 3100.0])  # $M

def gen_loan_balances():
    """Generate loan balance paths by segment × scenario."""
    growth = {
        "Baseline":        np.array([0.008, 0.006, 0.005, 0.007, 0.009]),
        "Adverse":         np.array([-0.005, -0.012, -0.003, -0.008, -0.002]),
        "Severely Adverse": np.array([-0.012, -0.025, -0.008, -0.015, -0.005]),
    }
    balances = {}
    for sc in SCENARIOS:
        bal = np.zeros((9, 5))
        bal[0] = STARTING_BALANCES * (1 + growth[sc])
        for q in range(1, 9):
            jitter = np.random.normal(0, 0.001, 5)
            bal[q] = bal[q-1] * (1 + growth[sc] + jitter)
        balances[sc] = bal
    return balances

# ── Loss rates by segment × scenario (annualized, convert to quarterly) ───
def gen_loss_rates():
    """Quarterly loss rates by segment × scenario × quarter."""
    # Annualized base loss rates
    ann_base = {
        "C&I":                  np.array([0.35, 0.33, 0.31, 0.30, 0.29, 0.28, 0.27, 0.27, 0.26]),
        "CRE":                  np.array([0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.20, 0.19, 0.19]),
        "Residential Mortgage": np.array([0.12, 0.11, 0.11, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09]),
        "Consumer":             np.array([1.80, 1.75, 1.70, 1.65, 1.60, 1.55, 1.50, 1.48, 1.45]),
        "Credit Card":          np.array([3.20, 3.10, 3.05, 3.00, 2.95, 2.90, 2.85, 2.80, 2.75]),
    }
    adv_mult  = {"C&I": 3.8, "CRE": 5.2, "Residential Mortgage": 4.5, "Consumer": 2.6, "Credit Card": 2.2}
    sev_mult  = {"C&I": 6.5, "CRE": 9.0, "Residential Mortgage": 7.8, "Consumer": 4.1, "Credit Card": 3.5}

    # Shape: stress peaks at Q4 (index 3) then slowly recovers
    stress_shape = np.array([0.55, 0.78, 0.94, 1.00, 0.96, 0.88, 0.76, 0.62, 0.48])

    rates = {}
    for sc in SCENARIOS:
        r = {}
        for seg in SEGMENTS:
            if sc == "Baseline":
                r[seg] = ann_base[seg] / 100 / 4  # quarterly
            elif sc == "Adverse":
                r[seg] = ann_base[seg] / 100 / 4 * adv_mult[seg] * stress_shape + ann_base[seg] / 100 / 4 * (1 - stress_shape)
            else:
                r[seg] = ann_base[seg] / 100 / 4 * sev_mult[seg] * stress_shape + ann_base[seg] / 100 / 4 * (1 - stress_shape)
        rates[sc] = r
    return rates

# ── PPNR components ───────────────────────────────────────────────────────
def gen_ppnr():
    """NII, non-interest income, non-interest expense by scenario × quarter."""
    ppnr = {}
    for sc in SCENARIOS:
        if sc == "Baseline":
            nii = 420.0 + np.cumsum(np.random.normal(2.5, 1.0, 9))
            nii_income = 145.0 + np.cumsum(np.random.normal(1.0, 0.8, 9))
            nie = 340.0 + np.cumsum(np.random.normal(1.5, 0.5, 9))
        elif sc == "Adverse":
            nii = 420.0 + np.cumsum(np.random.normal(-8.0, 2.0, 9))
            nii_income = 145.0 + np.cumsum(np.random.normal(-5.0, 1.5, 9))
            nie = 340.0 + np.cumsum(np.random.normal(3.0, 1.0, 9))
        else:
            nii = 420.0 + np.cumsum(np.random.normal(-15.0, 3.0, 9))
            nii_income = 145.0 + np.cumsum(np.random.normal(-9.0, 2.0, 9))
            nie = 340.0 + np.cumsum(np.random.normal(5.0, 1.5, 9))
        ppnr[sc] = {"NII": nii, "Non-Interest Income": nii_income, "Non-Interest Expense": nie}
    return ppnr

# ── RWA by category ──────────────────────────────────────────────────────
def gen_rwa():
    """RWA by category × scenario × quarter."""
    categories = ["Credit Risk — Loans", "Credit Risk — Securities",
                   "Credit Risk — Other", "Market Risk", "Operational Risk"]
    starting_rwa = np.array([28500.0, 4200.0, 2800.0, 1950.0, 3100.0])  # $M
    rwa = {}
    for sc in SCENARIOS:
        if sc == "Baseline":
            drift = np.array([0.003, 0.001, 0.002, 0.001, 0.002])
        elif sc == "Adverse":
            drift = np.array([0.012, 0.008, 0.006, 0.015, 0.005])
        else:
            drift = np.array([0.022, 0.015, 0.010, 0.028, 0.008])
        vals = np.zeros((9, 5))
        vals[0] = starting_rwa * (1 + drift)
        for q in range(1, 9):
            vals[q] = vals[q-1] * (1 + drift + np.random.normal(0, 0.001, 5))
        rwa[sc] = {cat: vals[:, i] for i, cat in enumerate(categories)}
    return rwa, categories

# ── Capital stack (starting) ─────────────────────────────────────────────
CET1_START = 4350.0   # $M
AT1_START  = 580.0     # additional T1
T2_START   = 720.0     # Tier 2
TOTAL_ASSETS_M = TOTAL_ASSETS * 1000  # for leverage ratio

# Capital actions
QUARTERLY_DIVIDEND = 65.0   # $M
QUARTERLY_BUYBACK_BASE = 40.0  # $M, reduced under stress

def gen_capital(ppnr_data, loss_rates, loan_bals, rwa_data, rwa_cats):
    """
    Project capital ratios. CET1 = prior + PPNR - Provisions - Taxes - Dividends - Buybacks
    """
    results = {}
    for sc in SCENARIOS:
        nii = ppnr_data[sc]["NII"]
        nii_inc = ppnr_data[sc]["Non-Interest Income"]
        nie = ppnr_data[sc]["Non-Interest Expense"]
        ppnr_vec = nii + nii_inc - nie

        # Provisions = sum of losses across segments
        provisions = np.zeros(9)
        seg_losses = {}
        for j, seg in enumerate(SEGMENTS):
            qr = loss_rates[sc][seg]
            bal = loan_bals[sc][:, j]
            seg_losses[seg] = qr * bal
            provisions += seg_losses[seg]

        # Tax rate 21%
        pre_tax = ppnr_vec - provisions
        taxes = np.where(pre_tax > 0, pre_tax * 0.21, 0.0)

        # Capital actions
        if sc == "Baseline":
            buyback = np.full(9, QUARTERLY_BUYBACK_BASE)
        elif sc == "Adverse":
            buyback = np.array([40.0, 30.0, 15.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0])
        else:
            buyback = np.array([25.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])

        dividend = np.full(9, QUARTERLY_DIVIDEND)
        if sc == "Severely Adverse":
            dividend[3:7] = 45.0  # reduced dividend

        cet1 = np.zeros(9)
        cet1[0] = CET1_START + ppnr_vec[0] - provisions[0] - taxes[0] - dividend[0] - buyback[0]
        for q in range(1, 9):
            cet1[q] = cet1[q-1] + ppnr_vec[q] - provisions[q] - taxes[q] - dividend[q] - buyback[q]

        at1 = np.full(9, AT1_START)
        t2  = np.full(9, T2_START)
        tier1 = cet1 + at1
        total_cap = tier1 + t2

        total_rwa = np.zeros(9)
        for cat in rwa_cats:
            total_rwa += rwa_data[sc][cat]

        # Leverage denominator: average total assets, growing slightly
        if sc == "Baseline":
            ta = TOTAL_ASSETS_M * (1 + np.arange(9) * 0.004)
        elif sc == "Adverse":
            ta = TOTAL_ASSETS_M * (1 + np.arange(9) * 0.001 - 0.005 * np.minimum(np.arange(9), 5))
        else:
            ta = TOTAL_ASSETS_M * (1 - np.arange(9) * 0.003 - 0.008 * np.minimum(np.arange(9), 5))

        cet1_ratio = cet1 / total_rwa * 100
        t1_ratio   = tier1 / total_rwa * 100
        tc_ratio   = total_cap / total_rwa * 100
        lev_ratio  = tier1 / ta * 100

        results[sc] = {
            "CET1": cet1, "AT1": at1, "T2": t2, "Tier1": tier1, "TotalCap": total_cap,
            "TotalRWA": total_rwa, "TotalAssets": ta,
            "CET1_ratio": cet1_ratio, "T1_ratio": t1_ratio, "TC_ratio": tc_ratio, "Lev_ratio": lev_ratio,
            "PPNR": ppnr_vec, "Provisions": provisions, "Taxes": taxes,
            "Dividends": dividend, "Buybacks": buyback,
            "SegLosses": seg_losses,
        }
    return results

# ── Generate everything ───────────────────────────────────────────────────
macro = gen_macro()
loan_bals = gen_loan_balances()
loss_rates = gen_loss_rates()
ppnr_data = gen_ppnr()
rwa_data, rwa_cats = gen_rwa()
capital = gen_capital(ppnr_data, loss_rates, loan_bals, rwa_data, rwa_cats)

# ── Markdown rendering helpers ────────────────────────────────────────────
def fmt(x, dp=2):
    return f"{x:,.{dp}f}"

def fmt6(x):
    return f"{x:.6f}"

def fmt4(x):
    return f"{x:.4f}"

def pct(x, dp=2):
    return f"{x:.{dp}f}%"

def pct4(x):
    return f"{x:.4f}%"

def md_table(headers, rows, align=None):
    """Return markdown table string."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    if align is None:
        align = ["---"] * len(headers)
    lines.append("| " + " | ".join(align) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)

# ── Build the report ──────────────────────────────────────────────────────
out = []
o = out.append

o("# Stress Test Results Report")
o("")
o("## CCAR / DFAST Capital Adequacy Assessment")
o("")
o(f"**Institution:** {BANK_NAME}")
o(f"**Total Consolidated Assets:** ${fmt(TOTAL_ASSETS, 1)} billion (as of {REPORT_DATE})")
o(f"**Report Date:** {REPORT_DATE}")
o("**Projection Horizon:** Q1 2026 through Q1 2028 (9 quarters)")
o("**Regulatory Framework:** Dodd-Frank Act Stress Testing (DFAST) / Comprehensive Capital Analysis and Review (CCAR)")
o("**Submitted to:** Board of Governors of the Federal Reserve System")
o("")
o("---")
o("")

# ── Section 1: Institution Overview ──────────────────────────────────────
o("## 1. Institution Overview")
o("")
o(f"{BANK_NAME} is a mid-size bank holding company headquartered in Columbus, Ohio,")
o(f"with total consolidated assets of ${fmt(TOTAL_ASSETS, 1)} billion as of {REPORT_DATE}.")
o("The institution operates 187 branches across seven states in the Midwest and Mid-Atlantic")
o("regions. Primary business lines include commercial banking, retail banking, wealth management,")
o("and treasury services. The loan portfolio totals $45.8 billion, with concentrations in")
o("commercial and industrial (C&I), commercial real estate (CRE), and residential mortgage lending.")
o("")
o("### 1.1 Portfolio Composition (as of December 31, 2025)")
o("")
headers = ["Segment", "Outstanding Balance ($M)", "% of Total Portfolio", "Weighted Avg. LTV / Risk Rating"]
rows = []
total_bal = STARTING_BALANCES.sum()
ltvs = ["N/A (4.2 avg risk rating)", "62.8%", "71.3%", "N/A (consumer FICO 738)", "N/A (consumer FICO 721)"]
for i, seg in enumerate(SEGMENTS):
    rows.append([seg, fmt(STARTING_BALANCES[i], 2), pct(STARTING_BALANCES[i]/total_bal*100, 2), ltvs[i]])
rows.append(["**Total**", f"**{fmt(total_bal, 2)}**", "**100.00%**", "—"])
o(md_table(headers, rows, [":---", "---:", "---:", ":---"]))
o("")
o("---")
o("")

# ── Section 2: Scenario Descriptions ─────────────────────────────────────
o("## 2. Scenario Descriptions")
o("")
o("Three macroeconomic scenarios are considered in accordance with Federal Reserve guidance:")
o("")
o("- **Baseline:** A continuation of current economic trends with modest growth and stable financial conditions.")
o("- **Adverse:** A moderate recession characterized by rising unemployment, declining asset prices, and widening credit spreads.")
o("- **Severely Adverse:** A deep recession with a sharp contraction in GDP, severe unemployment, significant declines in house prices, and a flight-to-quality compressing Treasury yields while corporate spreads widen dramatically.")
o("")

# Table per macro variable
for var_name, scdata in macro.items():
    o(f"### {var_name}")
    o("")
    headers = ["Quarter"] + SCENARIOS
    rows = []
    for q in range(9):
        row = [QUARTERS[q]]
        for sc in SCENARIOS:
            if "Spread" in var_name or "Yield" in var_name or "Unemployment" in var_name:
                row.append(pct4(scdata[sc][q]))
            else:
                row.append(fmt4(scdata[sc][q]))
        rows.append(row)
    o(md_table(headers, rows, [":---"] + ["---:"]*3))
    o("")

o("---")
o("")

# ── Section 3: Capital Ratios by Scenario ────────────────────────────────
o("## 3. Capital Ratios by Scenario")
o("")
o("Regulatory minimum thresholds (including capital conservation buffer where applicable):")
o("")
o("| Ratio | Regulatory Minimum |")
o("| :--- | ---: |")
o("| Common Equity Tier 1 (CET1) | 4.5000% |")
o("| Tier 1 Capital | 6.0000% |")
o("| Total Capital | 8.0000% |")
o("| Tier 1 Leverage | 4.0000% |")
o("")

for sc in SCENARIOS:
    o(f"### 3.{SCENARIOS.index(sc)+1} {sc} Scenario — Capital Ratios")
    o("")
    headers = ["Quarter", "CET1 Ratio", "Tier 1 Ratio", "Total Capital Ratio", "Leverage Ratio",
               "CET1 ($M)", "Total RWA ($M)", "Total Assets ($M)"]
    rows = []
    d = capital[sc]
    for q in range(9):
        rows.append([
            QUARTERS[q],
            pct4(d["CET1_ratio"][q]),
            pct4(d["T1_ratio"][q]),
            pct4(d["TC_ratio"][q]),
            pct4(d["Lev_ratio"][q]),
            fmt(d["CET1"][q], 4),
            fmt(d["TotalRWA"][q], 4),
            fmt(d["TotalAssets"][q], 4),
        ])
    o(md_table(headers, rows, [":---"] + ["---:"]*7))
    o("")

o("---")
o("")

# ── Section 4: Pre-Provision Net Revenue ─────────────────────────────────
o("## 4. Pre-Provision Net Revenue (PPNR)")
o("")
o("PPNR is computed as: Net Interest Income + Non-Interest Income − Non-Interest Expense.")
o("")

for sc in SCENARIOS:
    o(f"### 4.{SCENARIOS.index(sc)+1} {sc} Scenario — PPNR Components ($M)")
    o("")
    headers = ["Quarter", "Net Interest Income", "Non-Interest Income", "Non-Interest Expense", "PPNR"]
    rows = []
    d = ppnr_data[sc]
    ppnr_vec = d["NII"] + d["Non-Interest Income"] - d["Non-Interest Expense"]
    for q in range(9):
        rows.append([
            QUARTERS[q],
            fmt(d["NII"][q], 4),
            fmt(d["Non-Interest Income"][q], 4),
            fmt(d["Non-Interest Expense"][q], 4),
            fmt(ppnr_vec[q], 4),
        ])
    # Cumulative row
    rows.append([
        "**Cumulative**",
        f"**{fmt(d['NII'].sum(), 4)}**",
        f"**{fmt(d['Non-Interest Income'].sum(), 4)}**",
        f"**{fmt(d['Non-Interest Expense'].sum(), 4)}**",
        f"**{fmt(ppnr_vec.sum(), 4)}**",
    ])
    o(md_table(headers, rows, [":---"] + ["---:"]*4))
    o("")

o("---")
o("")

# ── Section 5: Loan Loss Projections ─────────────────────────────────────
o("## 5. Loan Loss Projections by Portfolio Segment")
o("")
o("Projected credit losses ($M) represent charge-offs net of recoveries for each quarter.")
o("")

for sc in SCENARIOS:
    o(f"### 5.{SCENARIOS.index(sc)+1} {sc} Scenario — Quarterly Losses ($M)")
    o("")
    headers = ["Quarter"] + SEGMENTS + ["Total"]
    rows = []
    d = capital[sc]["SegLosses"]
    for q in range(9):
        row = [QUARTERS[q]]
        total_q = 0.0
        for seg in SEGMENTS:
            row.append(fmt(d[seg][q], 6))
            total_q += d[seg][q]
        row.append(fmt(total_q, 4))
        rows.append(row)
    # Cumulative
    cum_row = ["**Cumulative**"]
    grand_total = 0.0
    for seg in SEGMENTS:
        s = d[seg].sum()
        cum_row.append(f"**{fmt(s, 4)}**")
        grand_total += s
    cum_row.append(f"**{fmt(grand_total, 4)}**")
    rows.append(cum_row)
    o(md_table(headers, rows, [":---"] + ["---:"]*6))
    o("")

o("---")
o("")

# ── Section 6: Credit Loss Rates ─────────────────────────────────────────
o("## 6. Cumulative Credit Loss Rates by Segment")
o("")
o("Cumulative loss rate is computed as total projected losses over the 9-quarter horizon")
o("divided by the average outstanding balance for each segment.")
o("")

headers = ["Segment"] + SCENARIOS + ["Severely Adverse vs. Baseline Multiple"]
rows = []
for j, seg in enumerate(SEGMENTS):
    row = [seg]
    rates_by_sc = []
    for sc in SCENARIOS:
        cum_loss = capital[sc]["SegLosses"][seg].sum()
        avg_bal = loan_bals[sc][:, j].mean()
        rate = cum_loss / avg_bal * 100
        rates_by_sc.append(rate)
        row.append(pct4(rate))
    row.append(f"{rates_by_sc[2]/rates_by_sc[0]:.2f}x")
    rows.append(row)

# Total row
row = ["**Total Portfolio**"]
rates_total = []
for sc in SCENARIOS:
    cum_total = sum(capital[sc]["SegLosses"][seg].sum() for seg in SEGMENTS)
    avg_total = sum(loan_bals[sc][:, j].mean() for j in range(5))
    rate = cum_total / avg_total * 100
    rates_total.append(rate)
    row.append(f"**{pct4(rate)}**")
row.append(f"**{rates_total[2]/rates_total[0]:.2f}x**")
rows.append(row)
o(md_table(headers, rows, [":---"] + ["---:"]*4))
o("")
o("---")
o("")

# ── Section 7: Revenue Projections / NII Sensitivity ─────────────────────
o("## 7. Net Interest Income Sensitivity Analysis")
o("")
o("The table below shows NII under each scenario alongside the change from baseline,")
o("illustrating interest rate sensitivity of the balance sheet.")
o("")
headers = ["Quarter", "Baseline NII ($M)", "Adverse NII ($M)", "Adverse Δ ($M)", "Adverse Δ (%)",
           "Sev. Adverse NII ($M)", "Sev. Adverse Δ ($M)", "Sev. Adverse Δ (%)"]
rows = []
for q in range(9):
    b = ppnr_data["Baseline"]["NII"][q]
    a = ppnr_data["Adverse"]["NII"][q]
    s = ppnr_data["Severely Adverse"]["NII"][q]
    rows.append([
        QUARTERS[q],
        fmt(b, 4), fmt(a, 4), fmt(a - b, 4), pct4((a - b) / b * 100),
        fmt(s, 4), fmt(s - b, 4), pct4((s - b) / b * 100),
    ])
# Cumulative
b_cum = ppnr_data["Baseline"]["NII"].sum()
a_cum = ppnr_data["Adverse"]["NII"].sum()
s_cum = ppnr_data["Severely Adverse"]["NII"].sum()
rows.append([
    "**Cumulative**",
    f"**{fmt(b_cum, 4)}**", f"**{fmt(a_cum, 4)}**", f"**{fmt(a_cum - b_cum, 4)}**",
    f"**{pct4((a_cum - b_cum)/b_cum*100)}**",
    f"**{fmt(s_cum, 4)}**", f"**{fmt(s_cum - b_cum, 4)}**",
    f"**{pct4((s_cum - b_cum)/b_cum*100)}**",
])
o(md_table(headers, rows, [":---"] + ["---:"]*7))
o("")
o("---")
o("")

# ── Section 8: RWA Projections ───────────────────────────────────────────
o("## 8. Risk-Weighted Asset (RWA) Projections")
o("")

for sc in SCENARIOS:
    o(f"### 8.{SCENARIOS.index(sc)+1} {sc} Scenario — RWA by Category ($M)")
    o("")
    headers = ["Quarter"] + rwa_cats + ["Total RWA"]
    rows = []
    for q in range(9):
        row = [QUARTERS[q]]
        total_q = 0.0
        for cat in rwa_cats:
            v = rwa_data[sc][cat][q]
            row.append(fmt(v, 4))
            total_q += v
        row.append(fmt(total_q, 4))
        rows.append(row)
    o(md_table(headers, rows, [":---"] + ["---:"]*6))
    o("")

o("---")
o("")

# ── Section 9: Capital Actions ───────────────────────────────────────────
o("## 9. Capital Actions")
o("")
o("Planned capital distributions over the projection horizon. Under stress scenarios,")
o("the institution reduces or suspends share repurchases and may reduce common dividends")
o("to preserve capital adequacy.")
o("")

for sc in SCENARIOS:
    o(f"### 9.{SCENARIOS.index(sc)+1} {sc} Scenario — Capital Actions ($M)")
    o("")
    d = capital[sc]
    headers = ["Quarter", "Common Dividends", "Share Repurchases", "Total Distributions",
               "PPNR After Tax", "Net Capital Impact"]
    rows = []
    for q in range(9):
        ppnr_at = d["PPNR"][q] - d["Provisions"][q] - d["Taxes"][q]
        dist = d["Dividends"][q] + d["Buybacks"][q]
        net = ppnr_at - dist
        rows.append([
            QUARTERS[q],
            fmt(d["Dividends"][q], 2),
            fmt(d["Buybacks"][q], 2),
            fmt(dist, 2),
            fmt(ppnr_at, 4),
            fmt(net, 4),
        ])
    # Cumulative
    rows.append([
        "**Cumulative**",
        f"**{fmt(d['Dividends'].sum(), 2)}**",
        f"**{fmt(d['Buybacks'].sum(), 2)}**",
        f"**{fmt(d['Dividends'].sum() + d['Buybacks'].sum(), 2)}**",
        f"**{fmt((d['PPNR'] - d['Provisions'] - d['Taxes']).sum(), 4)}**",
        f"**{fmt((d['PPNR'] - d['Provisions'] - d['Taxes'] - d['Dividends'] - d['Buybacks']).sum(), 4)}**",
    ])
    o(md_table(headers, rows, [":---"] + ["---:"]*5))
    o("")

o("---")
o("")

# ── Section 10: Summary — Minimum Capital Ratios ────────────────────────
o("## 10. Summary: Minimum Capital Ratios and Pass/Fail Assessment")
o("")
o("The table below reports the minimum ratio observed during each scenario's 9-quarter projection")
o("horizon, the quarter in which the minimum occurs, and whether the institution maintains")
o("capital above the applicable regulatory minimum.")
o("")

reg_min = {"CET1": 4.5, "Tier 1": 6.0, "Total Capital": 8.0, "Leverage": 4.0}
ratio_keys = {"CET1": "CET1_ratio", "Tier 1": "T1_ratio", "Total Capital": "TC_ratio", "Leverage": "Lev_ratio"}

headers = ["Scenario", "Ratio", "Minimum Value", "Quarter of Minimum", "Regulatory Minimum", "Buffer", "Status"]
rows = []
for sc in SCENARIOS:
    d = capital[sc]
    for ratio_name, key in ratio_keys.items():
        vals = d[key]
        min_idx = np.argmin(vals)
        min_val = vals[min_idx]
        reg = reg_min[ratio_name]
        buffer = min_val - reg
        status = "PASS" if min_val >= reg else "**FAIL**"
        rows.append([
            sc, ratio_name, pct4(min_val), QUARTERS[min_idx], pct4(reg), pct4(buffer), status
        ])

o(md_table(headers, rows, [":---", ":---", "---:", ":---:", "---:", "---:", ":---:"]))
o("")

# Count failures
fail_count = sum(1 for r in rows if r[6] == "**FAIL**")
pass_count = len(rows) - fail_count

o(f"**Overall Assessment:** {pass_count} of {len(rows)} ratio-scenario combinations meet regulatory minimums;")
o(f"{fail_count} combination(s) breach the threshold under stress.")
o("")

# Identify failures for narrative
for r in rows:
    if r[6] == "**FAIL**":
        o(f"- **{r[0]} / {r[1]}:** Minimum of {r[2]} in {r[3]} breaches the {r[4]} floor by {r[5]}.")
o("")

o("---")
o("")

# ── Section 10.1: Detailed waterfall ─────────────────────────────────────
o("### 10.1 Capital Depletion Waterfall — Severely Adverse Scenario ($M)")
o("")
o("Starting CET1 capital through the projection horizon, showing cumulative impacts.")
o("")

d = capital["Severely Adverse"]
headers = ["Component", "Amount ($M)"]
cum_ppnr = d["PPNR"].sum()
cum_prov = d["Provisions"].sum()
cum_tax  = d["Taxes"].sum()
cum_div  = d["Dividends"].sum()
cum_buy  = d["Buybacks"].sum()
ending_cet1 = d["CET1"][-1]

rows = [
    ["Starting CET1 Capital (Q4 2025)", fmt(CET1_START, 4)],
    ["(+) Cumulative PPNR", fmt(cum_ppnr, 4)],
    ["(−) Cumulative Provisions for Loan Losses", fmt(cum_prov, 4)],
    ["(−) Cumulative Tax Expense", fmt(cum_tax, 4)],
    ["(−) Cumulative Common Dividends", fmt(cum_div, 4)],
    ["(−) Cumulative Share Repurchases", fmt(cum_buy, 4)],
    ["**Ending CET1 Capital (Q1 2028)**", f"**{fmt(ending_cet1, 4)}**"],
    ["Ending Total RWA", fmt(d["TotalRWA"][-1], 4)],
    ["**Ending CET1 Ratio**", f"**{pct4(d['CET1_ratio'][-1])}**"],
]
o(md_table(headers, rows, [":---", "---:"]))
o("")

# Verify internal consistency
computed_end = CET1_START + cum_ppnr - cum_prov - cum_tax - cum_div - cum_buy
o(f"*Internal consistency check: Starting CET1 ({fmt(CET1_START, 2)}) + PPNR ({fmt(cum_ppnr, 2)}) − Provisions ({fmt(cum_prov, 2)}) − Taxes ({fmt(cum_tax, 2)}) − Dividends ({fmt(cum_div, 2)}) − Buybacks ({fmt(cum_buy, 2)}) = {fmt(computed_end, 4)}, matching reported ending CET1 of {fmt(ending_cet1, 4)}.*")
o("")
o("---")
o("")

# ── Section 10.2: Cross-scenario comparison ──────────────────────────────
o("### 10.2 Cross-Scenario CET1 Ratio Trajectory")
o("")
headers = ["Quarter", "Baseline", "Adverse", "Severely Adverse"]
rows = []
for q in range(9):
    rows.append([
        QUARTERS[q],
        pct4(capital["Baseline"]["CET1_ratio"][q]),
        pct4(capital["Adverse"]["CET1_ratio"][q]),
        pct4(capital["Severely Adverse"]["CET1_ratio"][q]),
    ])
o(md_table(headers, rows, [":---"] + ["---:"]*3))
o("")

o("---")
o("")
o("### 10.3 Key Risk Findings")
o("")

# Find worst segment in worst quarter of severely adverse
sev = capital["Severely Adverse"]
worst_q = int(np.argmin(sev["CET1_ratio"]))
worst_ratio = sev["CET1_ratio"][worst_q]

# Find the segment with highest loss in that quarter
seg_in_worst_q = {seg: sev["SegLosses"][seg][worst_q] for seg in SEGMENTS}
worst_seg = max(seg_in_worst_q, key=seg_in_worst_q.get)
worst_seg_loss = seg_in_worst_q[worst_seg]

o(f"1. **Worst capital position** occurs in **{QUARTERS[worst_q]}** under the Severely Adverse scenario,")
o(f"   where CET1 falls to **{pct4(worst_ratio)}** (minimum threshold: 4.5000%).")
o("")
o(f"2. In that quarter ({QUARTERS[worst_q]}), the highest-loss portfolio segment is **{worst_seg}**")
o(f"   with losses of **${fmt(worst_seg_loss, 4)}M**, representing")
o(f"   {pct4(worst_seg_loss / sev['Provisions'][worst_q] * 100)} of total provisions for that quarter.")
o("")

# CRE cumulative loss analysis
cre_cum_base = capital["Baseline"]["SegLosses"]["CRE"].sum()
cre_cum_sev  = capital["Severely Adverse"]["SegLosses"]["CRE"].sum()
o(f"3. **CRE concentration risk:** Cumulative CRE losses increase from ${fmt(cre_cum_base, 2)}M (Baseline)")
o(f"   to ${fmt(cre_cum_sev, 2)}M (Severely Adverse), a **{cre_cum_sev/cre_cum_base:.1f}x** multiplier,")
o("   reflecting the portfolio's sensitivity to commercial property devaluations and rising vacancy rates.")
o("")

# NII erosion
nii_base_cum = ppnr_data["Baseline"]["NII"].sum()
nii_sev_cum  = ppnr_data["Severely Adverse"]["NII"].sum()
o(f"4. **NII erosion under rate compression:** Cumulative NII drops from ${fmt(nii_base_cum, 2)}M (Baseline)")
o(f"   to ${fmt(nii_sev_cum, 2)}M (Severely Adverse), a decline of ${fmt(nii_base_cum - nii_sev_cum, 2)}M")
o(f"   ({pct4((nii_base_cum - nii_sev_cum)/nii_base_cum*100)}), driven by the 10-Year Treasury yield")
o(f"   declining from {pct4(macro['10-Year Treasury Yield (%)']['Baseline'][0])} to a trough of")
o(f"   {pct4(min(macro['10-Year Treasury Yield (%)']['Severely Adverse']))}.")
o("")
o("5. **Capital action flexibility:** Under the Severely Adverse scenario, the institution")
o(f"   suspends buybacks for 6 quarters and reduces dividends from ${fmt(QUARTERLY_DIVIDEND, 0)}M to $45M/quarter")
o(f"   for 4 quarters, freeing ${fmt(QUARTERLY_BUYBACK_BASE*6 + 20*4, 0)}M in capital that partially offsets")
o("   elevated provisions.")
o("")
o("---")
o("")
o("*This report was prepared by the Enterprise Risk Management division of Heartland Commercial Bancorp, Inc.*")
o(f"*All projections are based on models validated as of {REPORT_DATE} and are subject to model risk.*")
o("*For questions, contact the Chief Risk Officer at cro@heartlandbancorp.com.*")
o("")

# ── Write output ─────────────────────────────────────────────────────────
report = "\n".join(out)

with open("/home/asudjianto/jupyterlab/kg-memory/finstructbench/instances/stress_test.md", "w") as f:
    f.write(report)

print(f"Report written: {len(report)} characters, {len(out)} lines")
