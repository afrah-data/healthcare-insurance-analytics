# Healthcare Insurance Fraud, Waste & Abuse (FWA) Detection — Stakeholder-Driven Analysis

Exploratory analysis of a synthetic health insurance claims dataset (10,000 claims, 300
providers, 2021–2025), framed around five internal stakeholders who each need a different
answer from the same data.

## Why stakeholder-driven, not just "find the fraud"

Most fraud-detection tutorials treat this as a single binary classification problem. In
practice, a real insurer has multiple teams making different decisions off the same claims
data, on different timescales and at different levels of granularity:

| Stakeholder | Core Question | Grain |
|---|---|---|
| **SIU** (Special Investigations Unit) | Which providers look like outliers worth investigating? | Provider |
| **Claims Operations** | Should this claim auto-approve, or get held for review? | Claim |
| **Finance / Compliance** | How much dollar exposure are we carrying from fraud? | Portfolio |
| **Provider Network Management** | Are there specialty or volume patterns affecting network risk? | Specialty / Provider |
| **Product / Underwriting** | Do certain insurance products carry structurally higher risk? | Product |

This project builds a separate, purpose-fit answer for each — the same underlying columns
(`Provider_ID`, `Provider_Specialty`, `Insurance_Type`, `Claim_Amount`,
`Days_Between_Service_and_Claim`, `Is_Fraud`) get reshaped differently depending on who's
asking.

## What's in this repo

- `FWA_Stakeholder_Analysis.ipynb` — the full analysis, with markdown explaining the
  reasoning behind each step (not just the code), run end-to-end against the dataset below
- `healthcare_fraud_detection.csv` — synthetic dataset (see Data section)

## Approach

1. **Missing data diagnosis** — checked whether missingness was random or informative
   (e.g. did claims with a missing `Insurance_Type` also skew toward fraud?) before choosing
   an imputation strategy per column (explicit `'Unknown'` category for categoricals, median
   for numerics, and a provider-level lookup fill for `Provider_Specialty`).
2. **SIU — provider outlier detection**: ranked providers by fraud rate, filtered out
   low-volume providers (a provider with 2 claims and 1 flagged isn't a "50% fraud rate"
   pattern) before trusting the ranking.
3. **Claims Ops — filing-delay risk buckets**: bucketed the gap between service date and
   claim filing (`pd.cut`) as a fast, claim-level triage signal.
4. **Provider Network Management — specialty & volume risk**: aggregated fraud rate by
   specialty against the portfolio baseline, and separately flagged providers who are volume
   outliers regardless of fraud rate (a credentialing/monitoring concern in its own right).
5. **Product / Underwriting — portfolio risk by product**: aggregated fraud rate and average
   claim size by `Insurance_Type` to see whether risk is structurally uneven across products.
6. **Finance / Compliance — dollar exposure**: quantified what share of total claim dollars
   is tied to fraud, and how much of that sits specifically with the SIU-flagged providers.

## Key techniques demonstrated

- `groupby().agg()` with named aggregations
- Leakage-aware feature engineering (provider/specialty statistics computed with an eye
  toward recomputing on train-only data before modeling)
- `pd.cut()` for interpretable, business-driven bucketing
- Missingness diagnosis as a signal, not just a nuisance to fill
- Multi-perspective framing of one dataset for different business decisions

## Key findings

- **Baseline fraud rate: 8.29%** of claims (829 of 10,000).
- **SIU:** top flagged provider ran a **42.1%** fraud rate vs. the 8.29% baseline, backed by
  a sufficient claim volume (38 claims) to trust the number.
- **Claims Ops:** filing delay turned out to be an almost perfect fraud predictor in this
  dataset (100% fraud same/next-day, ~25-28% at days 2-6, 0% at 7+ days) -- a strong enough
  pattern that it's flagged in the notebook as likely synthetic-label leakage rather than a
  realistic behavioral signal (see write-up for the reasoning).
- **Provider Network Mgmt:** specialty-level fraud rates were fairly flat (7.1%-9.7%); 18
  providers were flagged purely on claim volume (95th percentile+) as a separate
  monitoring list.
- **Underwriting:** fraud rate was essentially flat across insurance products (8.2%-8.5%) --
  no single product stood out as structurally riskier here.
- **Finance:** fraud claims represent **14.3%** of total dollar volume ($821K of $5.73M) --
  notably higher than their 8.3% share of claim *count*, meaning fraud skews toward larger
  claims. The SIU top-10 watchlist alone accounts for **$242K** in exposure.

## Limitations & next steps

`Is_Fraud` is a synthetic label provided for learning purposes, and the filing-delay pattern
above is a good example of why that matters. The natural next step is turning the remaining
signals (`Provider_Specialty`, claim volume, `Insurance_Type`) into features for a
classification model, deliberately **excluding** `Days_Between_Service_and_Claim` given the
leakage concern, with a train/test split that groups by provider so the same provider never
appears in both sets.

## Data

`healthcare_fraud_detection.csv` is a fully synthetic dataset generated for educational
purposes — no real patient or claims data is used.
