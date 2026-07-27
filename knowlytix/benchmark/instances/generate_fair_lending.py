#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate a synthetic Fair Lending Analysis Report (ECOA/HMDA compliance)
for a mid-size bank. Uses numpy with seed=789 for reproducibility.
"""

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(789)

# ── Configuration ──────────────────────────────────────────────────────────
BANK_NAME = "Prairie National Bank & Trust"
EXAM_PERIOD = "January 1, 2025 – December 31, 2025"
REPORT_DATE = "March 15, 2026"
TOTAL_ASSETS_B = 14.3  # $B
REGULATOR = "Office of the Comptroller of the Currency (OCC)"
ASSESSMENT_AREA = "Kansas City – Overland Park – Olathe MSA (29820)"

# ── Protected class segments ──────────────────────────────────────────────
RACE_ETHNICITY = [
    "White (Non-Hispanic)",
    "Black or African American",
    "Hispanic or Latino",
    "Asian",
    "American Indian / Alaska Native",
    "Native Hawaiian / Pacific Islander",
    "Two or More Races",
]
GENDERS = ["Male", "Female", "Joint (Male/Female)"]
AGE_GROUPS = ["18–29", "30–44", "45–59", "60–74", "75+"]

# ── Loan products ─────────────────────────────────────────────────────────
PRODUCTS = [
    "Conventional Mortgage",
    "FHA Mortgage",
    "Home Equity Line",
    "Auto Loan",
    "Small Business Loan",
]

# ── Census tracts for redlining analysis ──────────────────────────────────
TRACT_IDS = [
    "29095010100", "29095010200", "29095010300", "29095010400",
    "29095020100", "29095020200", "29095020300", "29095020400",
    "29095030100", "29095030200", "29095030300", "29095030400",
    "20091040100", "20091040200", "20091040300", "20091040400",
]
TRACT_INCOME_LEVELS = [
    "Upper", "Middle", "Middle", "Moderate",
    "Middle", "Lower", "Moderate", "Lower",
    "Upper", "Middle", "Moderate", "Lower",
    "Upper", "Middle", "Moderate", "Lower",
]
TRACT_MINORITY_PCT = np.array([
    12.3, 24.7, 31.5, 58.2,
    19.8, 72.4, 46.1, 81.3,
    8.5, 22.1, 53.9, 74.6,
    10.1, 27.3, 49.8, 68.7,
])


# ── Helper: generate application counts with realistic skew ───────────────
def gen_application_counts_by_race():
    """Return (applications, approvals, denials, withdrawals) per race."""
    # Base application volumes (White dominant in market area)
    base_apps = np.array([18420, 3210, 2870, 1640, 310, 120, 580])
    noise = np.random.randint(-80, 80, size=len(base_apps))
    apps = base_apps + noise

    # Approval rates — some segments deliberately below 0.80 AIR
    approval_rates = np.array([0.72, 0.54, 0.58, 0.74, 0.49, 0.61, 0.63])
    withdrawal_rates = np.array([0.08, 0.10, 0.09, 0.07, 0.12, 0.11, 0.09])

    approvals = np.round(apps * approval_rates).astype(int)
    withdrawals = np.round(apps * withdrawal_rates).astype(int)
    denials = apps - approvals - withdrawals

    return apps, approvals, denials, withdrawals


def gen_application_counts_by_gender():
    """Return (applications, approvals, denials, withdrawals) per gender."""
    apps = np.array([14830, 9410, 2910]) + np.random.randint(-50, 50, 3)
    approval_rates = np.array([0.71, 0.67, 0.76])
    withdrawal_rates = np.array([0.08, 0.09, 0.06])

    approvals = np.round(apps * approval_rates).astype(int)
    withdrawals = np.round(apps * withdrawal_rates).astype(int)
    denials = apps - approvals - withdrawals

    return apps, approvals, denials, withdrawals


def gen_application_counts_by_age():
    """Return (applications, approvals, denials, withdrawals) per age group."""
    apps = np.array([4120, 9540, 7810, 4360, 1320]) + np.random.randint(-40, 40, 5)
    approval_rates = np.array([0.59, 0.72, 0.74, 0.70, 0.63])
    withdrawal_rates = np.array([0.11, 0.08, 0.07, 0.08, 0.10])

    approvals = np.round(apps * approval_rates).astype(int)
    withdrawals = np.round(apps * withdrawal_rates).astype(int)
    denials = apps - approvals - withdrawals

    return apps, approvals, denials, withdrawals


def compute_air(approval_rates, majority_idx=0):
    """Adverse Impact Ratio = minority_rate / majority_rate."""
    majority_rate = approval_rates[majority_idx]
    return np.round(approval_rates / majority_rate, 4)


# ── Pricing analysis ──────────────────────────────────────────────────────
def gen_pricing_by_race():
    """Average APR and standard errors by race for conventional mortgages."""
    base_aprs = np.array([6.42, 6.89, 6.78, 6.38, 7.14, 6.95, 6.71])
    sds = np.array([0.81, 0.94, 0.88, 0.79, 1.12, 1.05, 0.90])
    n_loans = np.array([13262, 1733, 1665, 1214, 152, 73, 365])
    se = sds / np.sqrt(n_loans)

    # Two-sample t-test vs White group
    t_stats = (base_aprs[1:] - base_aprs[0]) / np.sqrt(se[1:]**2 + se[0]**2)
    df_approx = n_loans[1:] + n_loans[0] - 2
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_approx))

    return base_aprs, sds, n_loans, se, t_stats, p_values


# ── Geographic / redlining ────────────────────────────────────────────────
def gen_tract_data():
    """Generate HMDA-style data per census tract."""
    n_tracts = len(TRACT_IDS)
    apps = np.random.randint(80, 450, n_tracts)
    # Lower approval rates in high-minority, low-income tracts
    base_rate = 0.73
    minority_penalty = -0.0028 * TRACT_MINORITY_PCT
    income_bonus = np.array([
        0.06, 0.02, 0.02, -0.02,
        0.02, -0.06, -0.02, -0.06,
        0.06, 0.02, -0.02, -0.06,
        0.06, 0.02, -0.02, -0.06,
    ])
    rates = np.clip(base_rate + minority_penalty + income_bonus + np.random.normal(0, 0.015, n_tracts), 0.35, 0.92)
    approvals = np.round(apps * rates).astype(int)
    denials = apps - approvals
    avg_loan_k = np.round(np.random.uniform(120, 380, n_tracts), 1)
    return apps, approvals, denials, rates, avg_loan_k


# ── Model fairness metrics ────────────────────────────────────────────────
def gen_model_fairness():
    """Generate fairness metrics for the credit scoring model."""
    segments = ["White (Non-Hispanic)", "Black or African American",
                "Hispanic or Latino", "Asian"]
    # Demographic parity (selection rate)
    selection_rates = np.array([0.72, 0.55, 0.59, 0.73])
    # True positive rates (equalized odds)
    tpr = np.array([0.84, 0.76, 0.78, 0.85])
    # False positive rates
    fpr = np.array([0.09, 0.14, 0.12, 0.08])
    # Positive predictive value (predictive parity)
    ppv = np.array([0.91, 0.83, 0.86, 0.92])
    # AUC per group
    auc = np.array([0.88, 0.81, 0.83, 0.89])

    return segments, selection_rates, tpr, fpr, ppv, auc


# ── Matched-pair testing ──────────────────────────────────────────────────
def gen_matched_pairs():
    """Simulated matched-pair test results."""
    test_types = [
        "Pre-Application Inquiry",
        "Application Assistance",
        "Loan Terms Offered",
        "Underwriting Outcome",
        "Post-Approval Servicing",
    ]
    n_pairs = np.array([40, 35, 50, 45, 30])
    # Count of pairs where minority received less favorable treatment
    disparate = np.array([14, 8, 21, 18, 7])
    favorable = np.array([6, 10, 8, 5, 9])
    equal = n_pairs - disparate - favorable
    # Statistical significance (binomial test p-value)
    p_vals = []
    for i in range(len(test_types)):
        # One-sided binomial test: is disparate > 50% of non-equal outcomes?
        non_equal = disparate[i] + favorable[i]
        if non_equal > 0:
            # Use scipy.stats.binom_test (works across scipy versions)
            p = 0.0
            for k in range(disparate[i], non_equal + 1):
                p += stats.binom.pmf(k, non_equal, 0.5)
            p_vals.append(p)
        else:
            p_vals.append(1.0)
    p_vals = np.array(p_vals)
    return test_types, n_pairs, disparate, equal, favorable, p_vals


# ── Product-level denial reasons ──────────────────────────────────────────
def gen_denial_reasons():
    """Top denial reasons by product."""
    reasons = {
        "Conventional Mortgage": [
            ("Debt-to-Income Ratio", 34.2),
            ("Credit History", 24.8),
            ("Insufficient Collateral", 18.1),
            ("Employment History", 12.5),
            ("Incomplete Application", 10.4),
        ],
        "FHA Mortgage": [
            ("Credit History", 31.7),
            ("Debt-to-Income Ratio", 28.3),
            ("Employment History", 16.9),
            ("Insufficient Cash", 13.4),
            ("Incomplete Application", 9.7),
        ],
        "Small Business Loan": [
            ("Insufficient Cash Flow", 29.5),
            ("Credit History", 22.1),
            ("Insufficient Collateral", 20.8),
            ("Time in Business", 15.3),
            ("Incomplete Application", 12.3),
        ],
    }
    return reasons


# ══════════════════════════════════════════════════════════════════════════
#  BUILD REPORT
# ══════════════════════════════════════════════════════════════════════════

lines = []


# ── Title & metadata ──────────────────────────────────────────────────────
lines.append("# Fair Lending Analysis Report")
lines.append("")
lines.append(f"## Institution: {BANK_NAME}")
lines.append("")
lines.append("### Report Metadata")
lines.append("")
lines.append(f"- **Report Date:** {REPORT_DATE}")
lines.append(f"- **Examination Period:** {EXAM_PERIOD}")
lines.append(f"- **Supervisory Agency:** {REGULATOR}")
lines.append(f"- **Assessment Area:** {ASSESSMENT_AREA}")
lines.append(f"- **Total Consolidated Assets:** ${TOTAL_ASSETS_B:.1f} billion")
lines.append("- **CRA Rating (Most Recent):** Satisfactory (2024)")
lines.append("- **Regulatory Framework:** Equal Credit Opportunity Act (ECOA), Home Mortgage Disclosure Act (HMDA), Fair Housing Act")
lines.append("- **Analysis Methodology:** Statistical regression, Adverse Impact Ratio (AIR), matched-pair testing, geographic analysis")
lines.append("")

# ── Section 1: Executive Summary ──────────────────────────────────────────
lines.append("## 1. Executive Summary")
lines.append("")
lines.append("This report presents the results of the annual fair lending analysis conducted for "
             f"{BANK_NAME} covering the examination period {EXAM_PERIOD}. The analysis evaluates "
             "compliance with ECOA, HMDA, and the Fair Housing Act across all consumer and mortgage "
             "lending products. The assessment encompasses application disposition analysis, pricing "
             "analysis, geographic distribution, credit model fairness evaluation, and matched-pair "
             "testing.")
lines.append("")
lines.append("### Key Findings")
lines.append("")
lines.append("- Adverse Impact Ratios for Black or African American and American Indian / Alaska Native applicants fall **below** the 0.80 threshold for conventional mortgage products, indicating potential disparate impact in underwriting.")
lines.append("- Pricing disparities for Black or African American borrowers are statistically significant (p < 0.001), with an average APR spread of 47 basis points above the White (Non-Hispanic) reference group after controlling for creditworthiness factors.")
lines.append("- Asian applicants show approval rates and pricing comparable to the majority group; no adverse findings.")
lines.append("- Geographic analysis identifies three census tracts classified as Lower-income with minority concentration above 70% where the bank's lending penetration is significantly below peer benchmarks.")
lines.append("- The credit scoring model exhibits equalized-odds gaps exceeding tolerance for the Black or African American segment.")
lines.append("- Matched-pair testing reveals statistically significant disparate treatment in loan terms offered (p = 0.0122).")
lines.append("")

# ── Section 2: Application Summary by Race/Ethnicity ─────────────────────
lines.append("## 2. Loan Application Disposition Analysis")
lines.append("")
lines.append("### 2.1 Applications by Race / Ethnicity")
lines.append("")

apps_r, approv_r, den_r, wd_r = gen_application_counts_by_race()
total_apps = apps_r.sum()
approval_rates_r = np.round(approv_r / apps_r, 4)
denial_rates_r = np.round(den_r / apps_r, 4)
air_r = compute_air(approval_rates_r, majority_idx=0)

df_race = pd.DataFrame({
    "Race / Ethnicity": RACE_ETHNICITY,
    "Applications": apps_r,
    "Approved": approv_r,
    "Denied": den_r,
    "Withdrawn": wd_r,
    "Approval Rate": [f"{r:.2%}" for r in approval_rates_r],
    "Denial Rate": [f"{r:.2%}" for r in denial_rates_r],
})
lines.append(df_race.to_markdown(index=False))
lines.append("")
lines.append(f"- **Total Applications (All Segments):** {total_apps:,}")
lines.append(f"- **Overall Approval Rate:** {approv_r.sum() / total_apps:.2%}")
lines.append(f"- **Overall Denial Rate:** {den_r.sum() / total_apps:.2%}")
lines.append("")

# AIR table
lines.append("### 2.2 Adverse Impact Ratios (Race / Ethnicity)")
lines.append("")
lines.append("The Adverse Impact Ratio (AIR) measures the ratio of the minority group's approval rate "
             "to the majority (White Non-Hispanic) group's approval rate. An AIR below 0.80 is the "
             "standard threshold indicating potential disparate impact under the four-fifths rule.")
lines.append("")

air_flag = ["Pass" if a >= 0.80 else "**FAIL**" for a in air_r]
df_air = pd.DataFrame({
    "Race / Ethnicity": RACE_ETHNICITY,
    "Approval Rate": [f"{r:.2%}" for r in approval_rates_r],
    "AIR vs. White": [f"{a:.4f}" for a in air_r],
    "Four-Fifths Test": air_flag,
})
lines.append(df_air.to_markdown(index=False))
lines.append("")

# ── Section 2.3: By Gender ───────────────────────────────────────────────
lines.append("### 2.3 Applications by Gender")
lines.append("")

apps_g, approv_g, den_g, wd_g = gen_application_counts_by_gender()
approval_rates_g = np.round(approv_g / apps_g, 4)
denial_rates_g = np.round(den_g / apps_g, 4)
air_g = compute_air(approval_rates_g, majority_idx=0)

df_gender = pd.DataFrame({
    "Gender": GENDERS,
    "Applications": apps_g,
    "Approved": approv_g,
    "Denied": den_g,
    "Withdrawn": wd_g,
    "Approval Rate": [f"{r:.2%}" for r in approval_rates_g],
    "AIR vs. Male": [f"{a:.4f}" for a in air_g],
    "Four-Fifths Test": ["Pass" if a >= 0.80 else "**FAIL**" for a in air_g],
})
lines.append(df_gender.to_markdown(index=False))
lines.append("")

# ── Section 2.4: By Age Group ────────────────────────────────────────────
lines.append("### 2.4 Applications by Age Group")
lines.append("")

apps_a, approv_a, den_a, wd_a = gen_application_counts_by_age()
approval_rates_a = np.round(approv_a / apps_a, 4)
denial_rates_a = np.round(den_a / apps_a, 4)
# Use 30-44 as reference (highest volume)
air_a = compute_air(approval_rates_a, majority_idx=1)

df_age = pd.DataFrame({
    "Age Group": AGE_GROUPS,
    "Applications": apps_a,
    "Approved": approv_a,
    "Denied": den_a,
    "Withdrawn": wd_a,
    "Approval Rate": [f"{r:.2%}" for r in approval_rates_a],
    "AIR vs. 30–44": [f"{a:.4f}" for a in air_a],
    "Four-Fifths Test": ["Pass" if a >= 0.80 else "**FAIL**" for a in air_a],
})
lines.append(df_age.to_markdown(index=False))
lines.append("")
lines.append("- **Reference Group (Age):** 30–44 (highest application volume)")
lines.append("")

# ── Section 3: Pricing Analysis ──────────────────────────────────────────
lines.append("## 3. Pricing Analysis — Conventional Mortgage")
lines.append("")
lines.append("Average Annual Percentage Rate (APR) by race/ethnicity for originated conventional "
             "mortgage loans during the examination period. Statistical significance assessed via "
             "two-sample t-test against the White (Non-Hispanic) reference group.")
lines.append("")

base_aprs, sds, n_loans, se, t_stats, p_values = gen_pricing_by_race()

sig_labels = []
for p in p_values:
    if p < 0.001:
        sig_labels.append("***")
    elif p < 0.01:
        sig_labels.append("**")
    elif p < 0.05:
        sig_labels.append("*")
    else:
        sig_labels.append("n.s.")

df_pricing = pd.DataFrame({
    "Race / Ethnicity": RACE_ETHNICITY,
    "N (Originated)": n_loans,
    "Mean APR (%)": [f"{a:.2f}" for a in base_aprs],
    "Std Dev": [f"{s:.2f}" for s in sds],
    "Spread vs. White (bps)": ["—"] + [f"{(base_aprs[i+1] - base_aprs[0]) * 100:+.0f}" for i in range(len(base_aprs)-1)],
    "t-Statistic": ["—"] + [f"{t:.2f}" for t in t_stats],
    "p-Value": ["—"] + [f"{p:.4f}" for p in p_values],
    "Significance": ["(ref)"] + sig_labels,
})
lines.append(df_pricing.to_markdown(index=False))
lines.append("")
lines.append("- Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05, n.s. = not significant")
lines.append("- Spread is measured in basis points (bps) relative to the White (Non-Hispanic) group")
lines.append("")

# ── Section 4: Geographic Distribution / Redlining ───────────────────────
lines.append("## 4. Geographic Distribution and Redlining Analysis")
lines.append("")
lines.append("HMDA data aggregated by census tract within the assessment area. Tracts are classified by "
             "HUD income level (Upper / Middle / Moderate / Lower) and minority population percentage.")
lines.append("")

t_apps, t_approv, t_den, t_rates, t_loan_k = gen_tract_data()

df_geo = pd.DataFrame({
    "Census Tract": TRACT_IDS,
    "Income Level": TRACT_INCOME_LEVELS,
    "Minority %": [f"{m:.1f}" for m in TRACT_MINORITY_PCT],
    "Applications": t_apps,
    "Approved": t_approv,
    "Denied": t_den,
    "Approval Rate": [f"{r:.2%}" for r in t_rates],
    "Avg Loan ($K)": [f"{v:.1f}" for v in t_loan_k],
})
lines.append(df_geo.to_markdown(index=False))
lines.append("")

# Summary stats by income level
lines.append("### 4.1 Aggregated by Tract Income Level")
lines.append("")
income_levels_unique = ["Upper", "Middle", "Moderate", "Lower"]
agg_rows = []
for lvl in income_levels_unique:
    mask = [il == lvl for il in TRACT_INCOME_LEVELS]
    mask = np.array(mask)
    total_a = t_apps[mask].sum()
    total_ap = t_approv[mask].sum()
    total_d = t_den[mask].sum()
    avg_min = TRACT_MINORITY_PCT[mask].mean()
    agg_rows.append({
        "Income Level": lvl,
        "Tracts": mask.sum(),
        "Total Applications": total_a,
        "Total Approved": total_ap,
        "Total Denied": total_d,
        "Aggregate Approval Rate": f"{total_ap / total_a:.2%}",
        "Mean Minority %": f"{avg_min:.1f}",
    })

df_geo_agg = pd.DataFrame(agg_rows)
lines.append(df_geo_agg.to_markdown(index=False))
lines.append("")
lines.append("- Lower-income tracts show a combined approval rate significantly below the bank-wide average, with disproportionately high minority concentration.")
lines.append("- Three Lower-income tracts (29095020200, 29095020400, 29095030400) have minority populations exceeding 70% and approval rates below 55%, warranting further review for potential redlining.")
lines.append("")

# ── Section 5: Model Fairness Metrics ─────────────────────────────────────
lines.append("## 5. Credit Model Fairness Evaluation")
lines.append("")
lines.append("Fairness metrics computed on the production credit scoring model (Model ID: CS-2024-v3.1) "
             "used for automated underwriting decisions. Metrics are disaggregated by race/ethnicity for "
             "the four largest applicant groups.")
lines.append("")

segments_f, sel_rates, tpr, fpr, ppv, auc = gen_model_fairness()

# Demographic parity ratio
dp_ratio = sel_rates / sel_rates[0]
# Equalized odds gap (TPR difference)
tpr_gap = tpr - tpr[0]
# FPR gap
fpr_gap = fpr - fpr[0]
# Predictive parity ratio
pp_ratio = ppv / ppv[0]

df_fairness = pd.DataFrame({
    "Segment": segments_f,
    "Selection Rate": [f"{s:.2%}" for s in sel_rates],
    "Demographic Parity Ratio": [f"{d:.4f}" for d in dp_ratio],
    "TPR (Recall)": [f"{t:.2%}" for t in tpr],
    "TPR Gap vs. White": [f"{g:+.2%}" for g in tpr_gap],
    "FPR": [f"{f:.2%}" for f in fpr],
    "FPR Gap vs. White": [f"{g:+.2%}" for g in fpr_gap],
    "PPV (Precision)": [f"{p:.2%}" for p in ppv],
    "Predictive Parity Ratio": [f"{r:.4f}" for r in pp_ratio],
    "AUC": [f"{a:.2f}" for a in auc],
})
lines.append(df_fairness.to_markdown(index=False))
lines.append("")

# Tolerance table
lines.append("### 5.1 Fairness Tolerance Thresholds and Results")
lines.append("")
metrics_summary = [
    ("Demographic Parity Ratio", ">= 0.80", [f"{d:.4f}" for d in dp_ratio[1:]],
     ["Pass" if d >= 0.80 else "**FAIL**" for d in dp_ratio[1:]]),
    ("Equalized Odds — TPR Gap", "<= 0.05", [f"{abs(g):.4f}" for g in tpr_gap[1:]],
     ["Pass" if abs(g) <= 0.05 else "**FAIL**" for g in tpr_gap[1:]]),
    ("Equalized Odds — FPR Gap", "<= 0.05", [f"{abs(g):.4f}" for g in fpr_gap[1:]],
     ["Pass" if abs(g) <= 0.05 else "**FAIL**" for g in fpr_gap[1:]]),
    ("Predictive Parity Ratio", ">= 0.90", [f"{r:.4f}" for r in pp_ratio[1:]],
     ["Pass" if r >= 0.90 else "**FAIL**" for r in pp_ratio[1:]]),
]

threshold_rows = []
for metric, threshold, vals, results in metrics_summary:
    for i, seg in enumerate(segments_f[1:]):
        threshold_rows.append({
            "Metric": metric,
            "Threshold": threshold,
            "Segment": seg,
            "Value": vals[i],
            "Result": results[i],
        })
df_thresh = pd.DataFrame(threshold_rows)
lines.append(df_thresh.to_markdown(index=False))
lines.append("")
lines.append("- The Black or African American segment fails the Demographic Parity Ratio threshold (0.7639 < 0.80) and the Equalized Odds TPR gap threshold (0.08 > 0.05).")
lines.append("- Hispanic or Latino segment is marginally below the Demographic Parity threshold but passes equalized odds criteria.")
lines.append("- Asian segment passes all fairness thresholds.")
lines.append("")

# ── Section 6: Matched-Pair Testing ──────────────────────────────────────
lines.append("## 6. Matched-Pair Testing Results")
lines.append("")
lines.append("Matched-pair (mystery shopper) tests conducted by an independent third party during "
             "Q2–Q3 2025. Each test pair consisted of one White (Non-Hispanic) tester and one Black or "
             "African American tester with equivalent credit profiles, income, and loan request parameters.")
lines.append("")

test_types, n_pairs, disparate, equal, favorable, p_vals_mp = gen_matched_pairs()

df_mp = pd.DataFrame({
    "Test Category": test_types,
    "N Pairs": n_pairs,
    "Disparate (Minority Less Favorable)": disparate,
    "Equal Treatment": equal,
    "Favorable (Minority More Favorable)": favorable,
    "Disparate Treatment Rate": [f"{d/n:.2%}" for d, n in zip(disparate, n_pairs)],
    "p-Value (One-Sided Binomial)": [f"{p:.4f}" for p in p_vals_mp],
    "Significant (p < 0.05)": ["**Yes**" if p < 0.05 else "No" for p in p_vals_mp],
})
lines.append(df_mp.to_markdown(index=False))
lines.append("")
lines.append("- Loan Terms Offered and Underwriting Outcome categories show statistically significant disparate treatment.")
lines.append("- Pre-Application Inquiry results approach significance (p = "
             f"{p_vals_mp[0]:.4f}) and should be monitored.")
lines.append("")

# ── Section 7: Denial Reason Analysis ─────────────────────────────────────
lines.append("## 7. Denial Reason Analysis by Product")
lines.append("")
lines.append("Top denial reasons (percentage of total denials) by loan product.")
lines.append("")

denial_reasons = gen_denial_reasons()
for product, reasons in denial_reasons.items():
    lines.append(f"### {product}")
    lines.append("")
    df_dr = pd.DataFrame(reasons, columns=["Denial Reason", "Share of Denials (%)"])
    lines.append(df_dr.to_markdown(index=False))
    lines.append("")

# ── Section 8: Remediation & Recommendations ─────────────────────────────
lines.append("## 8. Remediation Actions and Recommendations")
lines.append("")
lines.append("| # | Finding | Risk Level | Recommended Action | Target Date |")
lines.append("|---|---------|------------|-------------------|-------------|")
lines.append("| 1 | AIR below 0.80 for Black or African American applicants | High | Conduct second-look program for denied applicants; review underwriting overlays | Q2 2026 |")
lines.append("| 2 | AIR below 0.80 for American Indian / Alaska Native applicants | High | Enhance outreach and special-purpose credit programs in tribal areas | Q3 2026 |")
lines.append("| 3 | Statistically significant APR disparities (Black or African American) | High | Audit pricing exceptions and discretionary adjustments; implement rate-lock controls | Q2 2026 |")
lines.append("| 4 | Model equalized-odds gap (TPR) for Black or African American segment | Medium | Retrain model with bias mitigation constraints; validate on hold-out set | Q3 2026 |")
lines.append("| 5 | Lending penetration deficit in three Lower-income/high-minority tracts | Medium | Develop Community Development Lending Plan; partner with CDFIs | Q4 2026 |")
lines.append("| 6 | Disparate treatment in loan-terms matched-pair testing | High | Mandatory fair lending refresher training for all loan officers; implement offer standardization | Q2 2026 |")
lines.append("| 7 | Hispanic or Latino Demographic Parity Ratio approaching threshold | Low | Monitor quarterly; conduct root-cause analysis if ratio drops below 0.80 | Ongoing |")
lines.append("")

# ── Section 9: Appendix ──────────────────────────────────────────────────
lines.append("## 9. Appendix")
lines.append("")
lines.append("### 9.1 Data Sources")
lines.append("")
lines.append("- HMDA Loan Application Register (LAR) — 2025 filing")
lines.append("- Internal Loan Origination System (LOS) extracts")
lines.append("- Credit bureau data (Experian, TransUnion, Equifax)")
lines.append("- U.S. Census Bureau ACS 5-Year Estimates (2020–2024)")
lines.append("- FFIEC Census Tract demographic data")
lines.append("")
lines.append("### 9.2 Methodology Notes")
lines.append("")
lines.append("- Adverse Impact Ratio (AIR) calculated as: (minority approval rate) / (majority approval rate). Threshold: AIR >= 0.80.")
lines.append("- Pricing regression controls for: credit score, LTV, DTI, loan amount, property type, occupancy status, lock period.")
lines.append("- Matched-pair significance assessed using one-sided binomial test (H0: P(disparate) = 0.50).")
lines.append("- Model fairness metrics follow definitions in Barocas, Hardt & Narayanan (2019) and the CFPB fair lending supervision framework.")
lines.append("")
lines.append("### 9.3 Glossary")
lines.append("")
lines.append("| Abbreviation | Definition |")
lines.append("|-------------|-----------|")
lines.append("| AIR | Adverse Impact Ratio |")
lines.append("| APR | Annual Percentage Rate |")
lines.append("| AUC | Area Under the ROC Curve |")
lines.append("| CDFI | Community Development Financial Institution |")
lines.append("| CRA | Community Reinvestment Act |")
lines.append("| DTI | Debt-to-Income Ratio |")
lines.append("| ECOA | Equal Credit Opportunity Act |")
lines.append("| FPR | False Positive Rate |")
lines.append("| HMDA | Home Mortgage Disclosure Act |")
lines.append("| LTV | Loan-to-Value Ratio |")
lines.append("| MSA | Metropolitan Statistical Area |")
lines.append("| PPV | Positive Predictive Value |")
lines.append("| TPR | True Positive Rate |")
lines.append("")

# ── Write output ──────────────────────────────────────────────────────────
output_path = "/home/asudjianto/jupyterlab/kg-memory/finstructbench/instances/fair_lending.md"
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote {output_path}")
