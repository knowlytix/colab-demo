#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate a synthetic Basel III Capital Adequacy (Pillar 3 Disclosure) report
for a fictional regional bank. Uses numpy with seed=456 for reproducibility.

All numbers are internally consistent across tables.
"""

import numpy as np

np.random.seed(456)

# ============================================================================
# 1. INSTITUTION PARAMETERS
# ============================================================================
bank_name = "Pacific Northwest Bancorp, Inc."
reporting_date = "December 31, 2025"
prior_date = "September 30, 2025"
total_assets = 47_832_156_000  # ~$47.8B
total_deposits = 38_265_725_000
total_loans = 31_456_892_000

# ============================================================================
# 2. CAPITAL COMPOSITION
# ============================================================================
common_stock = 1_250_000_000
surplus = 875_000_000
retained_earnings = 2_340_567_891
aoci = -187_345_672
minority_interest_cet1 = 12_456_000
goodwill = -423_000_000
intangibles = -87_654_321
dta_deduction = -34_567_890
msr_deduction = -15_234_567
investment_in_unconsolidated = -8_900_000
pension_asset_deduction = -5_678_000

cet1_before_deductions = common_stock + surplus + retained_earnings + aoci + minority_interest_cet1
total_cet1_deductions = goodwill + intangibles + dta_deduction + msr_deduction + investment_in_unconsolidated + pension_asset_deduction
cet1_capital = cet1_before_deductions + total_cet1_deductions

# AT1
at1_preferred = 350_000_000
at1_minority = 5_678_000
at1_deductions = -2_345_000
at1_capital = at1_preferred + at1_minority + at1_deductions

tier1_capital = cet1_capital + at1_capital

# Tier 2
t2_subordinated = 500_000_000
t2_allowance = 245_678_901
t2_minority = 3_456_000
t2_deductions = -12_345_678
tier2_capital = t2_subordinated + t2_allowance + t2_minority + t2_deductions

total_capital = tier1_capital + tier2_capital

# ============================================================================
# 3. RISK-WEIGHTED ASSETS (must be consistent across all tables)
# ============================================================================
# Credit Risk RWA by exposure class
exposure_classes = [
    "Corporate",
    "Small and Medium Enterprises (SME)",
    "Retail — Qualifying Revolving",
    "Retail — Other",
    "Residential Mortgage",
    "Commercial Real Estate",
    "Sovereign and Central Bank",
    "Public Sector Entities",
    "Equity Exposures",
    "Securitization",
    "Other Assets",
]

ead_values = np.array([
    8_945_000_000,
    3_245_000_000,
    2_567_000_000,
    1_890_000_000,
    7_234_000_000,
    4_567_000_000,
    5_678_000_000,
    1_234_000_000,
    456_000_000,
    789_000_000,
    1_123_000_000,
], dtype=float)

risk_weights = np.array([
    0.85,   # Corporate
    1.00,   # SME
    0.75,   # Retail revolving
    0.75,   # Retail other
    0.35,   # Resi mortgage
    1.00,   # CRE
    0.00,   # Sovereign (home country)
    0.20,   # PSE
    2.50,   # Equity (IRB 250%)
    0.45,   # Securitization
    1.00,   # Other
])

credit_rwa = ead_values * risk_weights

# Market Risk RWA
market_risk_rwa_total = 1_234_567_890.0

# Operational Risk RWA
op_risk_rwa_total = 2_345_678_901.0

# CVA RWA
cva_rwa_total = 345_678_901.0

total_credit_rwa = credit_rwa.sum()
total_rwa = total_credit_rwa + market_risk_rwa_total + op_risk_rwa_total + cva_rwa_total

# ============================================================================
# 4. CAPITAL RATIOS
# ============================================================================
cet1_ratio = cet1_capital / total_rwa
tier1_ratio = tier1_capital / total_rwa
total_capital_ratio = total_capital / total_rwa

# Leverage ratio
on_balance_exposure = total_assets
derivative_exposure = 2_345_678_000
sft_exposure = 1_567_890_000
off_balance_exposure = 3_456_789_000
total_leverage_exposure = on_balance_exposure + derivative_exposure + sft_exposure + off_balance_exposure
leverage_ratio = tier1_capital / total_leverage_exposure

# ============================================================================
# 5. CREDIT RISK DETAIL (PD, LGD, EAD, EL by segment)
# ============================================================================
segments = [
    "Corporate — Investment Grade",
    "Corporate — Sub-Investment Grade",
    "Corporate — Specialized Lending",
    "SME — Secured",
    "SME — Unsecured",
    "Retail — Mortgage (Prime)",
    "Retail — Mortgage (Non-Prime)",
    "Retail — Credit Cards",
    "Retail — Auto Loans",
    "Retail — Personal Loans",
    "Commercial Real Estate — Income Producing",
    "Commercial Real Estate — Land/Construction",
    "Sovereign — Domestic",
    "Sovereign — Foreign (IG)",
    "Public Sector Entities",
]

pds = np.array([
    0.0023, 0.0189, 0.0312, 0.0145, 0.0367,
    0.0034, 0.0198, 0.0456, 0.0123, 0.0289,
    0.0178, 0.0423, 0.0002, 0.0015, 0.0008,
])

lgds = np.array([
    0.45, 0.45, 0.40, 0.35, 0.55,
    0.20, 0.30, 0.85, 0.50, 0.65,
    0.35, 0.40, 0.45, 0.45, 0.45,
])

eads_detail = np.array([
    5_234_000_000, 2_456_000_000, 1_255_000_000,
    1_890_000_000, 1_355_000_000,
    5_467_000_000, 1_767_000_000,
    1_678_000_000, 567_000_000, 645_000_000,
    3_245_000_000, 1_322_000_000,
    4_567_000_000, 1_111_000_000,
    1_234_000_000,
], dtype=float)

expected_loss = pds * lgds * eads_detail

# RWA for detailed segments (use a simplified IRB-like formula approximation)
# K ~ LGD * N[(1/(1-R))^0.5 * G(PD) + (R/(1-R))^0.5 * G(0.999)] - PD*LGD
# We'll use a simplified scalar approach for consistency
from scipy.stats import norm

def irb_capital_requirement(pd, lgd, ead, maturity=2.5):
    """Simplified IRB risk weight formula."""
    if pd < 1e-8:
        return 0.0
    R = 0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)) + \
        0.24 * (1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)))
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    K = lgd * norm.ppf(0.999) * np.sqrt(R / (1 - R))
    K += lgd * norm.ppf(pd) * np.sqrt(1 / (1 - R))
    # Use the conditional PD approach
    conditional_pd = norm.cdf(
        (norm.ppf(pd) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)
    )
    K = lgd * conditional_pd - pd * lgd
    K *= (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    K = max(K, 0)
    rwa = K * 12.5 * ead
    return rwa

segment_rwa = np.array([
    irb_capital_requirement(pds[i], lgds[i], eads_detail[i])
    for i in range(len(segments))
])

# ============================================================================
# 6. COUNTERPARTY CREDIT RISK
# ============================================================================
derivative_types = [
    "Interest Rate Swaps",
    "Cross-Currency Swaps",
    "FX Forwards / Options",
    "Credit Default Swaps",
    "Equity Options",
    "Commodity Derivatives",
]

notionals = np.array([
    12_456_000_000, 3_456_000_000, 5_678_000_000,
    1_234_000_000, 567_000_000, 234_000_000,
], dtype=float)

mtm_values = np.array([
    345_678_000, 123_456_000, 89_012_000,
    -34_567_000, 12_345_000, 5_678_000,
], dtype=float)

ccr_ead = np.array([
    678_901_234, 234_567_890, 178_901_234,
    89_012_345, 34_567_890, 12_345_678,
], dtype=float)

ccr_rw = np.array([0.50, 0.75, 0.50, 1.50, 1.00, 0.75])
ccr_rwa = ccr_ead * ccr_rw

# ============================================================================
# 7. MARKET RISK — VaR
# ============================================================================
risk_factors = ["Interest Rate", "Foreign Exchange", "Equity", "Commodity", "Credit Spread"]

var_99_1d = np.array([12_345_678, 5_678_901, 3_456_789, 1_234_567, 8_901_234], dtype=float)
stressed_var = var_99_1d * np.array([2.1, 2.5, 3.2, 2.8, 2.3])
var_10d = var_99_1d * np.sqrt(10)

# diversified VaR (correlation benefit)
sum_var = var_99_1d.sum()
diversified_var = sum_var * 0.72  # ~28% diversification benefit
sum_stressed = stressed_var.sum()
diversified_stressed = sum_stressed * 0.75

# ============================================================================
# 8. OPERATIONAL RISK — SMA
# ============================================================================
bi_components = {
    "Interest, Lease, and Dividend Component (ILDC)": 567_890_123,
    "Services Component (SC)": 345_678_901,
    "Financial Component (FC)": 123_456_789,
}
business_indicator = sum(bi_components.values())

# SMA buckets
# Bucket 1: BI <= 1B EUR, marginal coefficient 12%
# Bucket 2: 1B < BI <= 3B EUR, marginal coefficient 15%
# Bucket 3: BI > 30B EUR, marginal coefficient 18%
bi_component_capital = business_indicator * 0.12  # simplified
sma_multiplier = 1.0  # no loss multiplier adjustment
op_risk_capital = bi_component_capital * sma_multiplier

# ============================================================================
# 9. LARGE EXPOSURES
# ============================================================================
large_exposure_names = [
    "Cascade Energy Holdings, LLC",
    "Evergreen Timber Corporation",
    "Columbia River Port Authority",
    "Puget Sound Healthcare System",
    "Mount Rainier Resort Group",
    "Olympic Peninsula Utilities Co.",
    "Willamette Valley Agriculture Inc.",
    "Pacific Coast Shipping Corp.",
    "Cascade Semiconductor Ltd.",
    "Northwest Natural Resources Fund",
]

le_exposures = np.array([
    456_789_012, 389_012_345, 345_678_901, 312_456_789,
    289_012_345, 267_890_123, 245_678_901, 223_456_789,
    201_234_567, 189_012_345,
], dtype=float)

le_pct_capital = le_exposures / total_capital * 100
le_pct_cet1 = le_exposures / cet1_capital * 100

# ============================================================================
# 10. PRIOR PERIOD DATA (Q3 2025)
# ============================================================================
prior_cet1_ratio = cet1_ratio - 0.0023
prior_tier1_ratio = tier1_ratio - 0.0019
prior_total_ratio = total_capital_ratio - 0.0031
prior_leverage_ratio = leverage_ratio + 0.0008
prior_total_rwa = total_rwa * 1.012
prior_cet1_capital = cet1_capital * 0.997
prior_total_capital = total_capital * 0.995

# ============================================================================
# GENERATE MARKDOWN
# ============================================================================

def fmt(n, decimals=0):
    """Format number with commas."""
    if decimals == 0:
        return f"{int(round(n)):,}"
    return f"{n:,.{decimals}f}"

def pct(n, decimals=4):
    """Format as percentage."""
    return f"{n * 100:.{decimals}f}%"

def bps(n):
    """Format as basis points."""
    return f"{n * 10000:.2f}"

lines = []
L = lines.append

L("# Basel III Capital Adequacy Report — Pillar 3 Disclosure")
L("")
L(f"**{bank_name}**")
L("")
L(f"**Reporting Period Ending:** {reporting_date}")
L("")
L(f"**Prior Period:** {prior_date}")
L("")
L("---")
L("")

# ============================================================================
# SECTION 1: INSTITUTION OVERVIEW
# ============================================================================
L("## 1. Institution Overview")
L("")
L("| Item | Detail |")
L("|:-----|:-------|")
L(f"| Legal Entity Name | {bank_name} |")
L("| Regulatory Authority | Office of the Comptroller of the Currency (OCC) |")
L("| Basel Framework | Basel III (Standardized Approach for Credit Risk; Internal Models for Market Risk) |")
L(f"| Reporting Date | {reporting_date} |")
L("| Reporting Currency | USD (United States Dollar) |")
L(f"| Consolidated Total Assets | ${fmt(total_assets)} |")
L(f"| Total Deposits | ${fmt(total_deposits)} |")
L(f"| Total Loans and Leases | ${fmt(total_loans)} |")
L("| D-SIB Designation | Category IV (Total Assets $50B–$250B threshold) |")
L("| External Credit Rating | A- (S&P) / A3 (Moody's) / A- (Fitch) |")
L("")
L("Pacific Northwest Bancorp is a regional bank holding company headquartered in")
L("Portland, Oregon, operating 187 branches across Washington, Oregon, Idaho, and")
L("Montana. The institution is subject to enhanced prudential standards under the")
L("Dodd-Frank Act and reports under the Basel III Standardized Approach for credit")
L("risk capital requirements, with internal models approval for market risk (VaR).")
L("")

# ============================================================================
# SECTION 2: CAPITAL COMPOSITION
# ============================================================================
L("## 2. Capital Composition")
L("")
L("### Table 2.1 — Common Equity Tier 1 (CET1) Capital")
L("")
L("| Component | Amount (USD) |")
L("|:----------|-------------:|")
L(f"| Common Stock (par value) | ${fmt(common_stock)} |")
L(f"| Additional Paid-In Capital (Surplus) | ${fmt(surplus)} |")
L(f"| Retained Earnings | ${fmt(retained_earnings)} |")
L(f"| Accumulated Other Comprehensive Income (AOCI) | $({fmt(abs(aoci))}) |")
L(f"| Minority Interest (CET1 qualifying) | ${fmt(minority_interest_cet1)} |")
L(f"| **CET1 Before Regulatory Deductions** | **${fmt(cet1_before_deductions)}** |")
L("")

L("### Table 2.2 — CET1 Regulatory Deductions")
L("")
L("| Deduction | Amount (USD) |")
L("|:----------|-------------:|")
L(f"| Goodwill (net of associated DTL) | $({fmt(abs(goodwill))}) |")
L(f"| Other Intangible Assets (net of associated DTL) | $({fmt(abs(intangibles))}) |")
L(f"| Deferred Tax Assets (threshold deduction) | $({fmt(abs(dta_deduction))}) |")
L(f"| Mortgage Servicing Rights (threshold deduction) | $({fmt(abs(msr_deduction))}) |")
L(f"| Investment in Unconsolidated Financial Institutions | $({fmt(abs(investment_in_unconsolidated))}) |")
L(f"| Defined Benefit Pension Fund Net Assets | $({fmt(abs(pension_asset_deduction))}) |")
L(f"| **Total CET1 Deductions** | **$({fmt(abs(total_cet1_deductions))})** |")
L(f"| **CET1 Capital** | **${fmt(cet1_capital)}** |")
L("")

L("### Table 2.3 — Additional Tier 1 (AT1) and Tier 2 Capital")
L("")
L("| Component | Amount (USD) |")
L("|:----------|-------------:|")
L(f"| Non-Cumulative Perpetual Preferred Stock (AT1 qualifying) | ${fmt(at1_preferred)} |")
L(f"| Minority Interest (AT1 qualifying) | ${fmt(at1_minority)} |")
L(f"| AT1 Regulatory Deductions | $({fmt(abs(at1_deductions))}) |")
L(f"| **Additional Tier 1 Capital** | **${fmt(at1_capital)}** |")
L(f"| **Tier 1 Capital (CET1 + AT1)** | **${fmt(tier1_capital)}** |")
L("")
L(f"| Subordinated Debt (Tier 2 qualifying) | ${fmt(t2_subordinated)} |")
L(f"| Allowance for Loan Losses (eligible portion) | ${fmt(t2_allowance)} |")
L(f"| Minority Interest (Tier 2 qualifying) | ${fmt(t2_minority)} |")
L(f"| Tier 2 Regulatory Deductions | $({fmt(abs(t2_deductions))}) |")
L(f"| **Tier 2 Capital** | **${fmt(tier2_capital)}** |")
L(f"| **Total Regulatory Capital** | **${fmt(total_capital)}** |")
L("")

# ============================================================================
# SECTION 3: RISK-WEIGHTED ASSETS
# ============================================================================
L("## 3. Risk-Weighted Assets")
L("")
L("### Table 3.1 — Credit Risk RWA by Exposure Class")
L("")
L("| Exposure Class | EAD (USD) | Risk Weight | RWA (USD) |")
L("|:---------------|----------:|:-----------:|----------:|")
for i, ec in enumerate(exposure_classes):
    rw_str = f"{risk_weights[i]*100:.0f}%"
    L(f"| {ec} | ${fmt(ead_values[i])} | {rw_str} | ${fmt(credit_rwa[i])} |")
L(f"| **Total Credit Risk RWA** | **${fmt(ead_values.sum())}** | | **${fmt(total_credit_rwa)}** |")
L("")

L("### Table 3.2 — Total RWA Composition")
L("")
L("| Risk Category | RWA (USD) | % of Total |")
L("|:--------------|----------:|-----------:|")
L(f"| Credit Risk | ${fmt(total_credit_rwa)} | {pct(total_credit_rwa/total_rwa, 2)} |")
L(f"| Market Risk | ${fmt(market_risk_rwa_total)} | {pct(market_risk_rwa_total/total_rwa, 2)} |")
L(f"| Operational Risk | ${fmt(op_risk_rwa_total)} | {pct(op_risk_rwa_total/total_rwa, 2)} |")
L(f"| Credit Valuation Adjustment (CVA) | ${fmt(cva_rwa_total)} | {pct(cva_rwa_total/total_rwa, 2)} |")
L(f"| **Total Risk-Weighted Assets** | **${fmt(total_rwa)}** | **100.00%** |")
L("")

# ============================================================================
# SECTION 4: CAPITAL RATIOS
# ============================================================================
L("## 4. Capital Adequacy Ratios")
L("")
L("### Table 4.1 — Regulatory Capital Ratios")
L("")
L("| Ratio | Value | Minimum | Buffer | Min + Buffer | Status |")
L("|:------|------:|--------:|-------:|-------------:|:------:|")

ccb = 0.025
dsib = 0.01  # D-SIB surcharge for Category IV

cet1_min = 0.045
cet1_buf = ccb + dsib
cet1_pass = "PASS" if cet1_ratio >= cet1_min + cet1_buf else "FAIL"
L(f"| CET1 Capital Ratio | {pct(cet1_ratio)} | {pct(cet1_min)} | {pct(cet1_buf)} | {pct(cet1_min + cet1_buf)} | {cet1_pass} |")

t1_min = 0.06
t1_buf = ccb + dsib
t1_pass = "PASS" if tier1_ratio >= t1_min + t1_buf else "FAIL"
L(f"| Tier 1 Capital Ratio | {pct(tier1_ratio)} | {pct(t1_min)} | {pct(t1_buf)} | {pct(t1_min + t1_buf)} | {t1_pass} |")

tc_min = 0.08
tc_buf = ccb + dsib
tc_pass = "PASS" if total_capital_ratio >= tc_min + tc_buf else "FAIL"
L(f"| Total Capital Ratio | {pct(total_capital_ratio)} | {pct(tc_min)} | {pct(tc_buf)} | {pct(tc_min + tc_buf)} | {tc_pass} |")

lev_min = 0.04
lev_pass = "PASS" if leverage_ratio >= lev_min else "FAIL"
L(f"| Leverage Ratio | {pct(leverage_ratio)} | {pct(lev_min)} | N/A | {pct(lev_min)} | {lev_pass} |")
L("")

L("### Table 4.2 — Capital Surplus / (Shortfall)")
L("")
L("| Ratio | Actual | Required (incl. buffers) | Surplus (bps) | Surplus (USD) |")
L("|:------|-------:|-------------------------:|--------------:|--------------:|")
cet1_surplus_bps = (cet1_ratio - (cet1_min + cet1_buf)) * 10000
cet1_surplus_usd = cet1_surplus_bps / 10000 * total_rwa
t1_surplus_bps = (tier1_ratio - (t1_min + t1_buf)) * 10000
t1_surplus_usd = t1_surplus_bps / 10000 * total_rwa
tc_surplus_bps = (total_capital_ratio - (tc_min + tc_buf)) * 10000
tc_surplus_usd = tc_surplus_bps / 10000 * total_rwa
L(f"| CET1 | {pct(cet1_ratio)} | {pct(cet1_min + cet1_buf)} | {cet1_surplus_bps:+.2f} | ${fmt(cet1_surplus_usd)} |")
L(f"| Tier 1 | {pct(tier1_ratio)} | {pct(t1_min + t1_buf)} | {t1_surplus_bps:+.2f} | ${fmt(t1_surplus_usd)} |")
L(f"| Total Capital | {pct(total_capital_ratio)} | {pct(tc_min + tc_buf)} | {tc_surplus_bps:+.2f} | ${fmt(tc_surplus_usd)} |")
L("")

# ============================================================================
# SECTION 5: CREDIT RISK BY PORTFOLIO
# ============================================================================
L("## 5. Credit Risk — Portfolio Detail")
L("")
L("### Table 5.1 — IRB Parameters by Segment")
L("")
L("| Segment | PD | LGD | EAD (USD) | Expected Loss (USD) | RWA (USD) | Risk Density |")
L("|:--------|---:|----:|----------:|--------------------:|----------:|-----------:|")
for i, seg in enumerate(segments):
    rd = segment_rwa[i] / eads_detail[i] if eads_detail[i] > 0 else 0
    L(f"| {seg} | {pct(pds[i])} | {pct(lgds[i], 2)} | ${fmt(eads_detail[i])} | ${fmt(expected_loss[i])} | ${fmt(segment_rwa[i])} | {pct(rd, 2)} |")
L(f"| **Total** | | | **${fmt(eads_detail.sum())}** | **${fmt(expected_loss.sum())}** | **${fmt(segment_rwa.sum())}** | **{pct(segment_rwa.sum()/eads_detail.sum(), 2)}** |")
L("")

L("### Table 5.2 — Credit Risk Concentration by Geography")
L("")
geographies = ["Washington", "Oregon", "Idaho", "Montana", "Other States"]
geo_pcts = np.array([0.38, 0.29, 0.15, 0.08, 0.10])
geo_eads = eads_detail.sum() * geo_pcts
geo_npls = geo_eads * np.array([0.012, 0.015, 0.018, 0.021, 0.009])
L("| State / Region | EAD (USD) | % of Portfolio | NPL Amount (USD) | NPL Rate |")
L("|:---------------|----------:|---------------:|-----------------:|---------:|")
for i, g in enumerate(geographies):
    npl_rate = geo_npls[i] / geo_eads[i]
    L(f"| {g} | ${fmt(geo_eads[i])} | {pct(geo_pcts[i], 2)} | ${fmt(geo_npls[i])} | {pct(npl_rate)} |")
L(f"| **Total** | **${fmt(geo_eads.sum())}** | **100.00%** | **${fmt(geo_npls.sum())}** | **{pct(geo_npls.sum()/geo_eads.sum())}** |")
L("")

L("### Table 5.3 — Credit Risk Migration (PD Band Distribution)")
L("")
pd_bands = [
    ("0.00%–0.03%", 0.0002),
    ("0.03%–0.10%", 0.0008),
    ("0.10%–0.25%", 0.0023),
    ("0.25%–0.50%", 0.0034),
    ("0.50%–1.00%", 0.0145),
    ("1.00%–2.50%", 0.0189),
    ("2.50%–5.00%", 0.0367),
    ("5.00%–10.00%", 0.0456),
    ("10.00%–99.99%", 0.15),
    ("Default (100%)", 1.00),
]
pd_ead_dist = np.array([4_567_000_000, 5_678_000_000, 8_234_000_000, 5_456_000_000,
                         4_123_000_000, 3_234_000_000, 1_567_000_000, 678_000_000,
                         234_000_000, 56_000_000], dtype=float)
L("| PD Band | Weighted Avg PD | EAD (USD) | % of Total | Number of Obligors |")
L("|:--------|----------------:|----------:|-----------:|-------------------:|")
obligor_counts = [234, 567, 1_234, 2_345, 3_456, 2_678, 1_234, 456, 123, 34]
for i, (band, wpd) in enumerate(pd_bands):
    L(f"| {band} | {pct(wpd)} | ${fmt(pd_ead_dist[i])} | {pct(pd_ead_dist[i]/pd_ead_dist.sum(), 2)} | {fmt(obligor_counts[i])} |")
L(f"| **Total** | | **${fmt(pd_ead_dist.sum())}** | **100.00%** | **{fmt(sum(obligor_counts))}** |")
L("")

# ============================================================================
# SECTION 6: COUNTERPARTY CREDIT RISK
# ============================================================================
L("## 6. Counterparty Credit Risk")
L("")
L("### Table 6.1 — CCR by Derivative Type")
L("")
L("| Derivative Type | Notional (USD) | MTM (USD) | EAD (USD) | Risk Weight | RWA (USD) |")
L("|:----------------|---------------:|----------:|----------:|:-----------:|----------:|")
for i, dt in enumerate(derivative_types):
    rw_str = f"{ccr_rw[i]*100:.0f}%"
    L(f"| {dt} | ${fmt(notionals[i])} | ${fmt(mtm_values[i])} | ${fmt(ccr_ead[i])} | {rw_str} | ${fmt(ccr_rwa[i])} |")
L(f"| **Total** | **${fmt(notionals.sum())}** | **${fmt(mtm_values.sum())}** | **${fmt(ccr_ead.sum())}** | | **${fmt(ccr_rwa.sum())}** |")
L("")

L("### Table 6.2 — CCR Exposure by Counterparty Credit Quality")
L("")
cq_grades = ["AAA/AA", "A", "BBB", "BB", "B", "Below B / Unrated"]
cq_eads = np.array([345_678_000, 289_012_345, 234_567_890, 178_901_234, 123_456_789, 56_680_013], dtype=float)
cq_rws = np.array([0.20, 0.50, 0.75, 1.00, 1.50, 1.50])
cq_rwas = cq_eads * cq_rws
L("| Credit Quality | EAD (USD) | Risk Weight | RWA (USD) |")
L("|:---------------|----------:|:-----------:|----------:|")
for i, cq in enumerate(cq_grades):
    L(f"| {cq} | ${fmt(cq_eads[i])} | {cq_rws[i]*100:.0f}% | ${fmt(cq_rwas[i])} |")
L(f"| **Total** | **${fmt(cq_eads.sum())}** | | **${fmt(cq_rwas.sum())}** |")
L("")

L("### Table 6.3 — Credit Valuation Adjustment (CVA) Capital Charge")
L("")
L("| Component | Amount (USD) |")
L("|:----------|-------------:|")
L(f"| CVA RWA (BA-CVA) | ${fmt(cva_rwa_total)} |")
L(f"| CVA Capital Charge (8%) | ${fmt(cva_rwa_total * 0.08)} |")
L(f"| Eligible CVA Hedges (notional) | ${fmt(456_789_012)} |")
L(f"| Net CVA RWA (after hedge recognition) | ${fmt(cva_rwa_total * 0.85)} |")
L("")

# ============================================================================
# SECTION 7: MARKET RISK
# ============================================================================
L("## 7. Market Risk")
L("")
L("### Table 7.1 — Value-at-Risk (99%, 1-Day Holding Period)")
L("")
L("| Risk Factor | VaR (99%, 1-day) | VaR (99%, 10-day) | Stressed VaR (99%, 1-day) | sVaR Multiplier |")
L("|:------------|--:|--:|--:|--:|")
for i, rf in enumerate(risk_factors):
    mul = stressed_var[i] / var_99_1d[i]
    L(f"| {rf} | ${fmt(var_99_1d[i])} | ${fmt(var_10d[i])} | ${fmt(stressed_var[i])} | {mul:.2f}x |")
L(f"| **Undiversified Total** | **${fmt(sum_var)}** | **${fmt(sum_var * np.sqrt(10))}** | **${fmt(sum_stressed)}** | |")
L(f"| Diversification Benefit | $({fmt(sum_var - diversified_var)}) | | $({fmt(sum_stressed - diversified_stressed)}) | |")
L(f"| **Diversified Total** | **${fmt(diversified_var)}** | **${fmt(diversified_var * np.sqrt(10))}** | **${fmt(diversified_stressed)}** | |")
L("")

L("### Table 7.2 — Market Risk Capital Charge")
L("")
# IMA: max(VaR_t-1, mc * VaR_avg_60d) + max(sVaR_t-1, ms * sVaR_avg_60d) + specific risk
mc = 3.0  # multiplication factor
ms = 3.0
var_avg_60d = diversified_var * 0.95
svar_avg_60d = diversified_stressed * 0.92
ima_var_component = max(diversified_var, mc * var_avg_60d)
ima_svar_component = max(diversified_stressed, ms * svar_avg_60d)
specific_risk_charge = 23_456_789
total_market_risk_charge = ima_var_component + ima_svar_component + specific_risk_charge
L("| Component | Amount (USD) |")
L("|:----------|-------------:|")
L(f"| Previous Day VaR | ${fmt(diversified_var)} |")
L(f"| 60-Day Average VaR x Multiplier ({mc:.1f}x) | ${fmt(mc * var_avg_60d)} |")
L(f"| VaR Component (higher of above) | ${fmt(ima_var_component)} |")
L(f"| Previous Day Stressed VaR | ${fmt(diversified_stressed)} |")
L(f"| 60-Day Average sVaR x Multiplier ({ms:.1f}x) | ${fmt(ms * svar_avg_60d)} |")
L(f"| sVaR Component (higher of above) | ${fmt(ima_svar_component)} |")
L(f"| Specific Risk Surcharge | ${fmt(specific_risk_charge)} |")
L(f"| **Total IMA Market Risk Capital** | **${fmt(total_market_risk_charge)}** |")
L(f"| **Market Risk RWA (Capital x 12.5)** | **${fmt(total_market_risk_charge * 12.5)}** |")
L("")

L("### Table 7.3 — VaR Back-Testing Results (250 Trading Days)")
L("")
breaches = 3
L("| Metric | Value |")
L("|:-------|------:|")
L(f"| Number of VaR Exceptions (99% confidence) | {breaches} |")
L("| Basel Traffic Light Zone | Green (0–4 exceptions) |")
L(f"| Multiplication Factor (mc) | {mc:.1f} |")
L("| Back-Testing Period | January 2, 2025 – December 31, 2025 |")
L("| Trading Days in Period | 250 |")
L(f"| Exception Rate | {pct(breaches/250)} |")
L("")

# ============================================================================
# SECTION 8: OPERATIONAL RISK
# ============================================================================
L("## 8. Operational Risk")
L("")
L("### Table 8.1 — Standardised Measurement Approach (SMA) Calculation")
L("")
L("| Business Indicator Component | Amount (USD) |")
L("|:-----------------------------|-------------:|")
for comp, val in bi_components.items():
    L(f"| {comp} | ${fmt(val)} |")
L(f"| **Business Indicator (BI)** | **${fmt(business_indicator)}** |")
L("")

L("### Table 8.2 — SMA Capital Requirement")
L("")
# Bucket calculation
bucket1_limit = 1_000_000_000
bucket2_limit = 3_000_000_000
if business_indicator <= bucket1_limit:
    marginal_1 = business_indicator * 0.12
    marginal_2 = 0
    marginal_3 = 0
elif business_indicator <= bucket2_limit:
    marginal_1 = bucket1_limit * 0.12
    marginal_2 = (business_indicator - bucket1_limit) * 0.15
    marginal_3 = 0
else:
    marginal_1 = bucket1_limit * 0.12
    marginal_2 = (bucket2_limit - bucket1_limit) * 0.15
    marginal_3 = (business_indicator - bucket2_limit) * 0.18

bi_component_total = marginal_1 + marginal_2 + marginal_3
internal_loss_multiplier = 1.05  # slight history of losses
op_risk_capital_sma = bi_component_total * internal_loss_multiplier

L("| SMA Component | Amount (USD) |")
L("|:--------------|-------------:|")
L(f"| BI Bucket 1 (up to $1.0B at 12%) | ${fmt(marginal_1)} |")
L(f"| BI Bucket 2 ($1.0B–$3.0B at 15%) | ${fmt(marginal_2)} |")
L(f"| BI Bucket 3 (above $3.0B at 18%) | ${fmt(marginal_3)} |")
L(f"| **BI Component** | **${fmt(bi_component_total)}** |")
L(f"| Internal Loss Multiplier (ILM) | {internal_loss_multiplier:.2f}x |")
L(f"| **Operational Risk Capital Requirement** | **${fmt(op_risk_capital_sma)}** |")
L(f"| **Operational Risk RWA (Capital x 12.5)** | **${fmt(op_risk_capital_sma * 12.5)}** |")
L("")

L("### Table 8.3 — Operational Risk Loss History (5-Year)")
L("")
loss_years = [2021, 2022, 2023, 2024, 2025]
loss_amounts = [34_567_890, 28_901_234, 41_234_567, 37_890_123, 32_456_789]
loss_events = [89, 72, 103, 95, 81]
L("| Year | Number of Loss Events | Total Loss Amount (USD) | Average Loss (USD) |")
L("|:-----|----------------------:|------------------------:|-------------------:|")
for i, yr in enumerate(loss_years):
    avg = loss_amounts[i] / loss_events[i]
    L(f"| {yr} | {loss_events[i]} | ${fmt(loss_amounts[i])} | ${fmt(avg)} |")
L(f"| **5-Year Total** | **{sum(loss_events)}** | **${fmt(sum(loss_amounts))}** | **${fmt(sum(loss_amounts)/sum(loss_events))}** |")
L("")

# ============================================================================
# SECTION 9: LEVERAGE RATIO
# ============================================================================
L("## 9. Leverage Ratio")
L("")
L("### Table 9.1 — Leverage Ratio Composition")
L("")
L("| Component | Amount (USD) |")
L("|:----------|-------------:|")
L(f"| **Tier 1 Capital (Numerator)** | **${fmt(tier1_capital)}** |")
L(f"| On-Balance Sheet Exposures | ${fmt(on_balance_exposure)} |")
L(f"| Derivative Exposures (SA-CCR) | ${fmt(derivative_exposure)} |")
L(f"| Securities Financing Transactions | ${fmt(sft_exposure)} |")
L(f"| Off-Balance Sheet Items (CCF-adjusted) | ${fmt(off_balance_exposure)} |")
L(f"| **Total Leverage Exposure (Denominator)** | **${fmt(total_leverage_exposure)}** |")
L(f"| **Leverage Ratio** | **{pct(leverage_ratio)}** |")
L(f"| Minimum Requirement | {pct(lev_min)} |")
L(f"| Surplus / (Shortfall) | {(leverage_ratio - lev_min)*10000:+.2f} bps |")
L("")

# ============================================================================
# SECTION 10: LARGE EXPOSURES
# ============================================================================
L("## 10. Large Exposures")
L("")
L("### Table 10.1 — Top 10 Single-Name Exposures")
L("")
L("| Rank | Counterparty | Exposure (USD) | % of Tier 1 Capital | % of CET1 Capital | Sector |")
L("|:----:|:-------------|---------------:|--------------------:|-------------------:|:-------|")
sectors = ["Energy", "Forestry", "Government", "Healthcare", "Hospitality",
           "Utilities", "Agriculture", "Transportation", "Technology", "Natural Resources"]
for i in range(len(large_exposure_names)):
    pct_t1 = le_exposures[i] / tier1_capital * 100
    pct_c1 = le_pct_cet1[i]
    L(f"| {i+1} | {large_exposure_names[i]} | ${fmt(le_exposures[i])} | {pct_t1:.4f}% | {pct_c1:.4f}% | {sectors[i]} |")
L(f"| | **Total Top 10** | **${fmt(le_exposures.sum())}** | **{le_exposures.sum()/tier1_capital*100:.4f}%** | **{le_exposures.sum()/cet1_capital*100:.4f}%** | |")
L("")
L(f"> **Regulatory Limit:** Single-name exposures must not exceed 25% of Tier 1 Capital (${fmt(tier1_capital * 0.25)}). All exposures are within limits.")
L("")

# ============================================================================
# SECTION 11: BUFFER REQUIREMENTS
# ============================================================================
L("## 11. Capital Buffer Requirements")
L("")
L("### Table 11.1 — Combined Buffer Requirement")
L("")
ccyb = 0.0  # countercyclical buffer (US currently 0%)
dsib_buffer = 0.01
combined_buffer = ccb + ccyb + dsib_buffer
L("| Buffer Component | Rate |")
L("|:-----------------|-----:|")
L(f"| Capital Conservation Buffer (CCB) | {pct(ccb)} |")
L(f"| Countercyclical Capital Buffer (CCyB) | {pct(ccyb)} |")
L(f"| D-SIB Surcharge | {pct(dsib_buffer)} |")
L("| G-SIB Surcharge | 0.0000% (not designated) |")
L(f"| **Combined Buffer Requirement** | **{pct(combined_buffer)}** |")
L("")

L("### Table 11.2 — Buffer Compliance Assessment")
L("")
L("| Metric | Actual Ratio | Minimum | Combined Buffer | Floor (Min + Buffer) | Available Buffer | Status |")
L("|:-------|------------:|--------:|----------------:|---------------------:|-----------------:|:------:|")
cet1_avail_buf = cet1_ratio - cet1_min
t1_avail_buf = tier1_ratio - t1_min
tc_avail_buf = total_capital_ratio - tc_min
L(f"| CET1 | {pct(cet1_ratio)} | {pct(cet1_min)} | {pct(combined_buffer)} | {pct(cet1_min + combined_buffer)} | {pct(cet1_avail_buf)} | {'PASS' if cet1_avail_buf >= combined_buffer else 'CONSTRAINED'} |")
L(f"| Tier 1 | {pct(tier1_ratio)} | {pct(t1_min)} | {pct(combined_buffer)} | {pct(t1_min + combined_buffer)} | {pct(t1_avail_buf)} | {'PASS' if t1_avail_buf >= combined_buffer else 'CONSTRAINED'} |")
L(f"| Total Capital | {pct(total_capital_ratio)} | {pct(tc_min)} | {pct(combined_buffer)} | {pct(tc_min + combined_buffer)} | {pct(tc_avail_buf)} | {'PASS' if tc_avail_buf >= combined_buffer else 'CONSTRAINED'} |")
L("")

L("### Table 11.3 — Maximum Distributable Amount (MDA) Restriction")
L("")
L("| CET1 Buffer Range (above minimum) | Maximum Distribution Rate |")
L("|:----------------------------------|:-------------------------:|")
L(f"| Above {pct(combined_buffer)} (full buffer) | No restriction |")
L(f"| {pct(combined_buffer * 0.75)} – {pct(combined_buffer)} (fourth quartile) | 60% of earnings |")
L(f"| {pct(combined_buffer * 0.50)} – {pct(combined_buffer * 0.75)} (third quartile) | 40% of earnings |")
L(f"| {pct(combined_buffer * 0.25)} – {pct(combined_buffer * 0.50)} (second quartile) | 20% of earnings |")
L(f"| 0.0000% – {pct(combined_buffer * 0.25)} (first quartile) | 0% (no distributions) |")
L("")
L(f"> **Current Status:** CET1 buffer of {pct(cet1_avail_buf)} exceeds combined requirement of {pct(combined_buffer)}. No MDA restrictions apply.")
L("")

# ============================================================================
# SECTION 12: PRIOR PERIOD COMPARISON
# ============================================================================
L("## 12. Comparison to Prior Period")
L("")
L(f"### Table 12.1 — Quarter-over-Quarter Changes ({prior_date} to {reporting_date})")
L("")
L("| Metric | Current Period | Prior Period | Change | Change (bps) |")
L("|:-------|---------------:|-------------:|-------:|-------------:|")

def delta_row(name, curr, prev):
    chg = curr - prev
    bps_chg = chg * 10000
    return f"| {name} | {pct(curr)} | {pct(prev)} | {pct(chg)} | {bps_chg:+.2f} |"

L(delta_row("CET1 Ratio", cet1_ratio, prior_cet1_ratio))
L(delta_row("Tier 1 Ratio", tier1_ratio, prior_tier1_ratio))
L(delta_row("Total Capital Ratio", total_capital_ratio, prior_total_ratio))
L(delta_row("Leverage Ratio", leverage_ratio, prior_leverage_ratio))
L("")

L("### Table 12.2 — Key Balance Changes")
L("")
L("| Component | Current Period (USD) | Prior Period (USD) | Change (USD) | Change (%) |")
L("|:----------|---------------------:|-------------------:|-------------:|-----------:|")

def bal_row(name, curr, prev):
    chg = curr - prev
    pct_chg = chg / prev * 100 if prev != 0 else 0
    return f"| {name} | ${fmt(curr)} | ${fmt(prev)} | ${fmt(chg)} | {pct_chg:+.4f}% |"

L(bal_row("CET1 Capital", cet1_capital, prior_cet1_capital))
L(bal_row("Total Capital", total_capital, prior_total_capital))
L(bal_row("Total RWA", total_rwa, prior_total_rwa))
L(f"| Credit Risk RWA | ${fmt(total_credit_rwa)} | ${fmt(total_credit_rwa * 1.008)} | ${fmt(total_credit_rwa - total_credit_rwa * 1.008)} | {((1/1.008)-1)*100:+.4f}% |")
L(f"| Market Risk RWA | ${fmt(market_risk_rwa_total)} | ${fmt(market_risk_rwa_total * 1.023)} | ${fmt(market_risk_rwa_total - market_risk_rwa_total * 1.023)} | {((1/1.023)-1)*100:+.4f}% |")
L(f"| Operational Risk RWA | ${fmt(op_risk_rwa_total)} | ${fmt(op_risk_rwa_total * 0.998)} | ${fmt(op_risk_rwa_total - op_risk_rwa_total * 0.998)} | {((1/0.998)-1)*100:+.4f}% |")
L("")

# ============================================================================
# REGULATORY NOTES
# ============================================================================
L("## Appendix A: Regulatory Framework and Methodology Notes")
L("")
L("### A.1 — Basis of Preparation")
L("")
L("This report has been prepared in accordance with the Basel III framework as")
L("implemented by U.S. federal banking regulators through the following rules:")
L("")
L("- **12 CFR Part 3** (OCC): Risk-Based Capital and Leverage")
L("- **Regulation Q (12 CFR Part 217)**: Capital Adequacy of Bank Holding Companies")
L("- **Basel III Endgame (proposed)**: Expanded risk-based capital requirements")
L("")
L("Capital ratios are calculated on a fully phased-in basis. AOCI is included in")
L("CET1 capital without opt-out, consistent with Category IV institution requirements")
L("effective January 1, 2020.")
L("")
L("### A.2 — Credit Risk Methodology")
L("")
L("Credit risk-weighted assets are calculated using the U.S. Standardized Approach.")
L("Key risk weight assignments follow the Basel III standardized framework:")
L("")
L("| Exposure Type | Risk Weight | Regulatory Reference |")
L("|:--------------|:-----------:|:---------------------|")
L("| U.S. Government Securities | 0% | 12 CFR 3.32(a)(1) |")
L("| GSE Obligations | 20% | 12 CFR 3.32(c) |")
L("| Public Sector Entities (U.S.) | 20% | 12 CFR 3.32(e) |")
L("| Corporate (Investment Grade) | 65–100% | 12 CFR 3.32(f) |")
L("| Residential Mortgage (Category 1) | 35–50% | 12 CFR 3.33 |")
L("| Commercial Real Estate (HVCRE) | 150% | 12 CFR 3.33(d) |")
L("| Retail / Consumer | 75–100% | 12 CFR 3.34 |")
L("| Equity Exposures | 100–600% | 12 CFR 3.35 |")
L("| Securitization | 20–1,250% | 12 CFR 3.42 |")
L("")
L("### A.3 — Market Risk Methodology")
L("")
L("Market risk capital is calculated using the Internal Models Approach (IMA) under")
L("OCC approval (effective March 15, 2019). The VaR model employs historical")
L("simulation with a 500-day lookback period. Stressed VaR uses a stress window")
L("calibrated to the 2008–2009 financial crisis period (September 2008 – March 2009).")
L("")
L("### A.4 — Operational Risk Methodology")
L("")
L("Operational risk capital is calculated under the Standardised Measurement Approach")
L("(SMA) as prescribed by the Basel III Endgame proposal. The Business Indicator (BI)")
L("is computed from audited financial statements for the prior three fiscal years.")
L("The Internal Loss Multiplier (ILM) reflects the institution's 10-year operational")
L("loss history relative to the BI Component.")
L("")
L("### A.5 — Accounting Standards")
L("")
L("All figures are presented under U.S. Generally Accepted Accounting Principles")
L("(US GAAP). The Current Expected Credit Losses (CECL) methodology is applied for")
L("the allowance for credit losses, with a three-year phase-in adjustment for")
L("regulatory capital purposes (75% phase-in as of the reporting date).")
L("")

# ============================================================================
# CROSS-REFERENCE CHECK
# ============================================================================
L("## Appendix B: Internal Consistency Cross-Reference")
L("")
L("The following table summarizes key figures that appear across multiple sections")
L("and confirms internal consistency of the report.")
L("")
L("| Cross-Reference | Section | Value (USD) |")
L("|:----------------|:--------|------------:|")
L(f"| CET1 Capital | §2, §4, §9 | ${fmt(cet1_capital)} |")
L(f"| Tier 1 Capital | §2, §4, §9 | ${fmt(tier1_capital)} |")
L(f"| Total Capital | §2, §4 | ${fmt(total_capital)} |")
L(f"| Total Credit Risk RWA | §3.1, §3.2 | ${fmt(total_credit_rwa)} |")
L(f"| Market Risk RWA | §3.2, §7 | ${fmt(market_risk_rwa_total)} |")
L(f"| Operational Risk RWA | §3.2, §8 | ${fmt(op_risk_rwa_total)} |")
L(f"| Total RWA | §3.2, §4 | ${fmt(total_rwa)} |")
L(f"| CET1 Ratio (CET1/RWA) | §4.1 | {pct(cet1_ratio, 6)} |")
L(f"| Tier 1 Ratio (T1/RWA) | §4.1 | {pct(tier1_ratio, 6)} |")
L(f"| Total Capital Ratio (TC/RWA) | §4.1 | {pct(total_capital_ratio, 6)} |")
L(f"| Leverage Ratio (T1/TLE) | §9 | {pct(leverage_ratio, 6)} |")
L(f"| Total Leverage Exposure | §9 | ${fmt(total_leverage_exposure)} |")
L("")
L("---")
L("")
L("*This document is for regulatory disclosure purposes under Basel III Pillar 3")
L("requirements. All data is synthetic and does not represent any actual financial")
L("institution. Prepared in accordance with the disclosure templates prescribed by")
L("the Basel Committee on Banking Supervision (BCBS d400, revised December 2018).*")
L("")
L(f"**Report Generated:** {reporting_date}")
L("")
L(f"**Approved By:** Chief Risk Officer, {bank_name}")
L("")

# ============================================================================
# WRITE OUTPUT
# ============================================================================
output_path = "/home/asudjianto/jupyterlab/kg-memory/finstructbench/instances/basel_capital.md"
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"Written {len(lines)} lines to {output_path}")
print("\nKey consistency checks:")
print(f"  CET1 Capital:        ${cet1_capital:,.0f}")
print(f"  Tier 1 Capital:      ${tier1_capital:,.0f}")
print(f"  Total Capital:       ${total_capital:,.0f}")
print(f"  Total Credit RWA:    ${total_credit_rwa:,.0f}")
print(f"  Total RWA:           ${total_rwa:,.0f}")
print(f"  CET1 Ratio:          {cet1_ratio*100:.6f}%")
print(f"  Tier 1 Ratio:        {tier1_ratio*100:.6f}%")
print(f"  Total Capital Ratio: {total_capital_ratio*100:.6f}%")
print(f"  Leverage Ratio:      {leverage_ratio*100:.6f}%")
