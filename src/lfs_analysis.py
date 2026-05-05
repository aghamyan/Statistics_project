#!/usr/bin/env python3
"""Armenia LFS 2021-2024 analysis pipeline.

This script builds two analytical datasets:
- period == "2024"
- period == "2021_2023" (pooled)

It performs cleaning, harmonization, descriptive summaries, hypothesis testing,
ANOVA, and regression, then exports tables/figures for a LaTeX report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]

DATA_FILES = {
    2021: ROOT / "LFS_Year_2021 Dataset (residents).xlsx",
    2022: ROOT / "LFS_Year_2022 Dataset (residents).xlsx",
    2023: ROOT / "LFS_2023 Dataset (residents).xlsx",
    2024: ROOT / "LFS_2024 Dataset (residents).xlsx",
}

OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_DATA = ROOT / "outputs" / "data"


# IMPORTANT:
# These names must match real dataset/codebook column names.
# If results look empty, this VARMAP is the first place to check.
VARMAP = {
    "person_id": ["person_id", "pid", "ID", "ID_mem", "IDmem"],
    "weight": ["weight", "wgt", "final_weight", "WeightsCalib_year"],

    "age": ["age", "AGE", "B4"],
    "sex": ["sex", "gender", "SEX", "B3"],
    "region": ["region", "marz", "REGION", "A2"],

    "employment_status_raw": [
        "empl_stat",
        "labour_status",
        "ECON_STATUS",
        "POLF",
        "LF",
    ],

    "nace_section": [
        "nace",
        "sector_code",
        "NACE",
        "E4_21group_NACE_rev_2.2",
        "F3_21groups_NACE_rev_2.2",
    ],

    "disability_raw": [
        "disability",
        "disabled",
        "DISAB",
        "D5",
    ],

    "marital_raw": [
        "marital",
        "marital_status",
        "MARITAL",
        "B11",
    ],

    "monthly_income": [
        "income_monthly",
        "wage_month",
        "MONTH_INC",
        "G1_3Total",
        "G2_3Total",
    ],

    "paid_hours_week": [
        "paid_hours_week",
        "work_hours",
        "HOURS_WEEK",
        "E13",
        "E14_Res",
    ],

    "unpaid_dom_hours": [
        "unpaid_dom_hours",
        "domestic_hours",
        "UNPAID_H",
        "Kd1_Res",
        "Kb1_Res",
    ],
}


# Common missing/special codes in social surveys.
SPECIAL_MISSINGS = {
    -9, -8, -7, -6, -5,
    96, 97, 98, 99,
    996, 997, 998, 999,
}


def choose_col(cols: List[str], candidates: List[str]) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of {candidates} found in columns")


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    rename = {}

    for target, candidates in VARMAP.items():
        try:
            source_col = choose_col(cols, candidates)
            rename[source_col] = target
        except KeyError:
            pass

    out = df.rename(columns=rename).copy()

    required = [
        "age",
        "sex",
        "region",
        "employment_status_raw",
        "nace_section",
        "disability_raw",
        "marital_raw",
        "monthly_income",
        "paid_hours_week",
        "unpaid_dom_hours",
    ]

    missing = [c for c in required if c not in out.columns]

    if missing:
        print("\nAvailable columns in this file:")
        print(out.columns.tolist())
        raise ValueError(
            f"Missing harmonized variables: {missing}. "
            f"Update VARMAP using the official codebook/questionnaire."
        )

    return out


def recode(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    numeric_cols = [
        "age",
        "sex",
        "employment_status_raw",
        "nace_section",
        "disability_raw",
        "marital_raw",
        "monthly_income",
        "paid_hours_week",
        "unpaid_dom_hours",
    ]

    for c in numeric_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[d[c].isin(SPECIAL_MISSINGS), c] = np.nan

    # WARNING:
    # These mappings assume the dataset uses these exact codes.
    # Verify using official Armstat codebook.
    d["gender"] = d["sex"].map({
        1: "Male",
        2: "Female",
    })

    d["disability_status"] = d["disability_raw"].map({
        1: "Has disability",
        2: "No disability",
    })

    d["marital_status"] = d["marital_raw"].map({
        1: "Never married",
        2: "Married",
        3: "Divorced/Separated",
        4: "Widowed",
    })

    d["employment_status"] = d["employment_status_raw"].map({
        1: "Employed",
        2: "Unemployed",
        3: "Out of labor force",
    })

    # Sector recoding from 21 NACE groups.
    d["sector"] = pd.Series(
        np.select(
            [
                d["nace_section"].between(1, 3),
                d["nace_section"].between(4, 8),
                d["nace_section"].between(9, 21),
            ],
            [
                "Agriculture",
                "Industry",
                "Services",
            ],
            default=None,
        ),
        index=d.index,
        dtype="string",
    )

    d["age_group"] = pd.cut(
        d["age"],
        bins=[14, 24, 34, 44, 54, 64, 120],
        labels=["15-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    )

    return d


def clean_ranges(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d.loc[
        (d["monthly_income"] < 0) | (d["monthly_income"] > 10_000_000),
        "monthly_income",
    ] = np.nan

    d.loc[
        (d["paid_hours_week"] < 0) | (d["paid_hours_week"] > 112),
        "paid_hours_week",
    ] = np.nan

    d.loc[
        (d["unpaid_dom_hours"] < 0) | (d["unpaid_dom_hours"] > 112),
        "unpaid_dom_hours",
    ] = np.nan

    return d


def weighted_mean_ci(x, w):
    ok = (~pd.isna(x)) & (~pd.isna(w))
    x = np.asarray(x[ok])
    w = np.asarray(w[ok])

    if len(x) == 0 or w.sum() == 0:
        return np.nan, np.nan, np.nan, 0

    m = np.average(x, weights=w)
    neff = (w.sum() ** 2) / (w ** 2).sum()
    var = np.average((x - m) ** 2, weights=w)
    se = np.sqrt(var / neff)
    ci = (m - 1.96 * se, m + 1.96 * se)

    return m, ci[0], ci[1], neff


def run_chi2(df: pd.DataFrame, row: str, col: str, name: str) -> pd.DataFrame:
    sub = df[[row, col]].dropna()
    tab = pd.crosstab(sub[row], sub[col])

    if tab.empty or tab.shape[0] < 2 or tab.shape[1] < 2:
        print(f"\nWARNING: Skipping {name}")
        print("Reason: not enough valid data for chi-square test.")
        print(f"Rows before dropna: {len(df)}")
        print(f"Rows after dropna: {len(sub)}")

        print(f"\n{row} values:")
        print(df[row].value_counts(dropna=False).head(30))

        print(f"\n{col} values:")
        print(df[col].value_counts(dropna=False).head(30))

        return pd.DataFrame([{
            "test": name,
            "chi2": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "n": len(sub),
            "status": "skipped_empty_or_insufficient_table",
        }])

    chi2, p, dof, exp = stats.chi2_contingency(tab)

    tab.to_csv(OUT_TABLES / f"{name}_crosstab.csv")

    return pd.DataFrame([{
        "test": name,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "n": int(tab.values.sum()),
        "status": "ok",
    }])


def run_anova(df: pd.DataFrame, y: str, x: str, name: str) -> pd.DataFrame:
    sub = df[[y, x]].dropna()

    if sub.empty or sub[x].nunique() < 2:
        print(f"\nWARNING: Skipping ANOVA {name}")
        print("Reason: not enough valid groups.")
        print(f"Rows after dropna: {len(sub)}")
        print(f"{x} values:")
        print(df[x].value_counts(dropna=False).head(30))

        return pd.DataFrame([{
            "test": name,
            "term": x,
            "sum_sq": np.nan,
            "df": np.nan,
            "F": np.nan,
            "PR(>F)": np.nan,
            "status": "skipped_insufficient_groups",
        }])

    model = smf.ols(f"{y} ~ C({x})", data=sub).fit()

    a = sm.stats.anova_lm(model, typ=2).reset_index()
    a = a.rename(columns={"index": "term"})
    a.insert(0, "test", name)
    a["status"] = "ok"

    return a


def debug_prints(all_df: pd.DataFrame, emp: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("DEBUG SUMMARY")
    print("=" * 70)

    print("\nall_df shape:", all_df.shape)
    print("emp shape:", emp.shape)

    print("\nYear counts:")
    print(all_df["year"].value_counts(dropna=False).sort_index())

    print("\nPeriod counts:")
    print(all_df["period"].value_counts(dropna=False))

    print("\nEmployment status raw values:")
    print(all_df["employment_status_raw"].value_counts(dropna=False).head(30))

    print("\nEmployment status recoded values:")
    print(all_df["employment_status"].value_counts(dropna=False).head(30))

    print("\nSex raw values:")
    print(all_df["sex"].value_counts(dropna=False).head(30))

    print("\nGender recoded values:")
    print(all_df["gender"].value_counts(dropna=False).head(30))

    print("\nNACE section raw values:")
    print(all_df["nace_section"].value_counts(dropna=False).head(40))

    print("\nSector recoded values:")
    print(all_df["sector"].value_counts(dropna=False).head(30))

    print("\nDisability raw values:")
    print(all_df["disability_raw"].value_counts(dropna=False).head(30))

    print("\nDisability recoded values:")
    print(all_df["disability_status"].value_counts(dropna=False).head(30))

    print("\nMarital raw values:")
    print(all_df["marital_raw"].value_counts(dropna=False).head(30))

    print("\nMarital recoded values:")
    print(all_df["marital_status"].value_counts(dropna=False).head(30))

    print("\nEmployed period counts:")
    print(emp["period"].value_counts(dropna=False))

    print("=" * 70 + "\n")


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    frames = []
    cleaning_log = []

    for year, path in DATA_FILES.items():
        print(f"\nReading {year}: {path.name}")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_excel(path)
        n0 = len(df)

        d = harmonize_columns(df)
        d = recode(d)
        d = clean_ranges(d)

        d["year"] = year

        frames.append(d)

        cleaning_log.append({
            "year": year,
            "raw_n": n0,
            "post_harmonize_n": len(d),
        })

    all_df = pd.concat(frames, ignore_index=True).copy()

    all_df["period"] = np.where(
        all_df["year"] == 2024,
        "2024",
        "2021_2023",
    )

    all_df.to_parquet(OUT_DATA / "lfs_cleaned.parquet", index=False)
    pd.DataFrame(cleaning_log).to_csv(
        OUT_TABLES / "cleaning_log.csv",
        index=False,
    )

    emp = all_df[all_df["employment_status"] == "Employed"].copy()

    debug_prints(all_df, emp)

    # Q1: Sector x gender and region among employed people.
    chi_tables = []

    for p in ["2024", "2021_2023"]:
        sub = emp[emp["period"] == p].copy()

        chi_tables.append(
            run_chi2(
                sub,
                "sector",
                "gender",
                f"q1_sector_gender_{p}",
            )
        )

        chi_tables.append(
            run_chi2(
                sub,
                "sector",
                "region",
                f"q1_sector_region_{p}",
            )
        )

    pd.concat(chi_tables, ignore_index=True).to_csv(
        OUT_TABLES / "q1_chi2_summary.csv",
        index=False,
    )

    # Q2: Disability x labor force status.
    q2 = []

    for p in ["2024", "2021_2023"]:
        sub = all_df[all_df["period"] == p].copy()

        q2.append(
            run_chi2(
                sub,
                "disability_status",
                "employment_status",
                f"q2_disability_status_{p}",
            )
        )

    pd.concat(q2, ignore_index=True).to_csv(
        OUT_TABLES / "q2_chi2_summary.csv",
        index=False,
    )

    # Q3: ANOVA income by marital status among employed.
    q3 = []

    for p in ["2024", "2021_2023"]:
        sub = emp[
            (emp["period"] == p)
            & emp["monthly_income"].notna()
        ].copy()

        q3.append(
            run_anova(
                sub,
                "monthly_income",
                "marital_status",
                f"q3_income_marital_{p}",
            )
        )

    pd.concat(q3, ignore_index=True).to_csv(
        OUT_TABLES / "q3_anova_income.csv",
        index=False,
    )

    # Q4: ANOVA paid hours by age group among employed.
    q4 = []

    for p in ["2024", "2021_2023"]:
        sub = emp[
            (emp["period"] == p)
            & emp["paid_hours_week"].notna()
        ].copy()

        q4.append(
            run_anova(
                sub,
                "paid_hours_week",
                "age_group",
                f"q4_hours_age_{p}",
            )
        )

    pd.concat(q4, ignore_index=True).to_csv(
        OUT_TABLES / "q4_anova_hours.csv",
        index=False,
    )

    # Q5: Regression: paid hours on unpaid domestic work.
    reg_rows = []

    for p in ["2024", "2021_2023"]:
        sub = emp[emp["period"] == p].dropna(
            subset=[
                "paid_hours_week",
                "unpaid_dom_hours",
                "age",
                "gender",
                "marital_status",
            ]
        ).copy()

        if len(sub) < 10:
            print(f"\nWARNING: Skipping regression for {p}")
            print("Reason: not enough valid observations.")
            print("Rows:", len(sub))

            reg_rows.append({
                "period": p,
                "model": "simple",
                "n": len(sub),
                "r2": np.nan,
                "coef_unpaid": np.nan,
                "p_unpaid": np.nan,
                "status": "skipped_insufficient_observations",
            })

            reg_rows.append({
                "period": p,
                "model": "adjusted",
                "n": len(sub),
                "r2": np.nan,
                "coef_unpaid": np.nan,
                "p_unpaid": np.nan,
                "status": "skipped_insufficient_observations",
            })

            continue

        m1 = smf.ols(
            "paid_hours_week ~ unpaid_dom_hours",
            data=sub,
        ).fit(cov_type="HC3")

        m2 = smf.ols(
            "paid_hours_week ~ unpaid_dom_hours + age + C(gender) + C(marital_status)",
            data=sub,
        ).fit(cov_type="HC3")

        for mname, m in [("simple", m1), ("adjusted", m2)]:
            reg_rows.append({
                "period": p,
                "model": mname,
                "n": int(m.nobs),
                "r2": m.rsquared,
                "coef_unpaid": m.params.get("unpaid_dom_hours", np.nan),
                "p_unpaid": m.pvalues.get("unpaid_dom_hours", np.nan),
                "status": "ok",
            })

        with open(OUT_TABLES / f"q5_regression_{p}.txt", "w") as f:
            f.write(m2.summary().as_text())

    pd.DataFrame(reg_rows).to_csv(
        OUT_TABLES / "q5_regression_summary.csv",
        index=False,
    )

    # Figures.
    sns.set_theme(style="whitegrid")

    if not emp.empty:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=emp.dropna(subset=["sector", "gender"]), x="sector", hue="gender")
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "sector_by_gender.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, 5))
        sns.countplot(data=emp.dropna(subset=["region", "sector"]), x="region", hue="sector")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "sector_by_region.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.boxplot(
            data=emp.dropna(subset=["marital_status", "monthly_income", "period"]),
            x="marital_status",
            y="monthly_income",
            hue="period",
        )
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "income_by_marital.png", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.pointplot(
            data=emp.dropna(subset=["age_group", "paid_hours_week", "period"]),
            x="age_group",
            y="paid_hours_week",
            hue="period",
            errorbar=("ci", 95),
        )
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "hours_by_agegroup_ci.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.regplot(
            data=emp.dropna(subset=["unpaid_dom_hours", "paid_hours_week"]),
            x="unpaid_dom_hours",
            y="paid_hours_week",
            scatter_kws={"alpha": 0.2},
        )
        plt.tight_layout()
        plt.savefig(OUT_FIGS / "paid_vs_unpaid_hours_reg.png", dpi=300)
        plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=all_df.dropna(subset=["employment_status", "disability_status"]),
        x="employment_status",
        hue="disability_status",
        multiple="fill",
        stat="probability",
    )
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "employment_by_disability_stacked.png", dpi=300)
    plt.close()

    # Descriptive statistics.
    desc = all_df.groupby("period")[
        [
            "age",
            "monthly_income",
            "paid_hours_week",
            "unpaid_dom_hours",
        ]
    ].describe().T

    desc.to_csv(OUT_TABLES / "descriptive_stats.csv")

    metadata = {
        "note": (
            "Update VARMAP and category maps using official questionnaire "
            "before final inference."
        ),
        "period_definition": {
            "2024": "year == 2024",
            "2021_2023": "year in [2021, 2022, 2023]",
        },
    }

    (OUT_DATA / "pipeline_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nDONE.")
    print(f"Tables saved to: {OUT_TABLES}")
    print(f"Figures saved to: {OUT_FIGS}")
    print(f"Cleaned data saved to: {OUT_DATA}")


if __name__ == "__main__":
    main()