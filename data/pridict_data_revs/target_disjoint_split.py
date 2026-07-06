"""
Target-disjoint data splitting for PRIDICT Library-1 pegRNA data.

Splits pegRNAs into train/validation/test sets such that all pegRNAs targeting
the same mutation remain in the same partition. The mutation identifier is the
`Name` column from the original PRIDICT supplementary spreadsheet, joined to the
cleaned CSV by the wide initial and mutated target sequences.

This avoids leakage from related pegRNAs that share a target mutation but differ
in spacer choice, RTT overhang, or other pegRNA design parameters.

Usage:
    python data/pridict_data_revs/target_disjoint_split.py \
        --excel data/pridict_data_revs/41587_2022_1613_MOESM5_ESM.xlsx \
        --output_dir data/pridict_data_revs/ \
        --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 --seed 42

If `--csv` is provided, the script uses that pre-cleaned crispAIPE-format CSV.
If `--csv` is omitted, the script rebuilds the crispAIPE-format Library-1 table
directly from the PRIDICT supplementary Excel file using the PE2 readout columns.
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


CRISPAIPE_COLUMNS = [
    "total_read_count",
    "edited_percentage",
    "unedited_percentage",
    "indel_percentage",
    "initial_sequence",
    "mutated_sequence",
    "protospacer_location",
    "pbs_location",
    "rt_initial_location",
    "rt_mutated_location",
]


def extract_protospacer_seq(row):
    seq = row["initial_sequence"]
    start, end = map(int, row["protospacer_location"].strip("[]").split(","))
    return seq[start:end + 1]


def _shift_location_end(location, offset=-1):
    """Convert PRIDICT's PBS end coordinate to the cleaned CSV convention."""
    start, end = map(int, str(location).strip("[]").split(","))
    return f"[{start}, {end + offset}]"


def build_crispaipe_library1_from_excel(xl_df):
    """Create the crispAIPE Library-1 input schema from the PRIDICT workbook."""
    required_cols = [
        "PE2df_totalreads",
        "PE2df_percentageedited",
        "PE2df_percentageunedited",
        "PE2df_percentageindel",
        "wide_initial_target",
        "wide_mutated_target",
        "protospacerlocation_only_initial",
        "PBSlocation",
        "RT_initial_location",
        "RT_mutated_location",
    ]
    missing = sorted(set(required_cols) - set(xl_df.columns))
    if missing:
        raise ValueError(f"Missing required PRIDICT Library_1 columns: {missing}")

    csv_df = pd.DataFrame(
        {
            "total_read_count": xl_df["PE2df_totalreads"],
            "edited_percentage": xl_df["PE2df_percentageedited"],
            "unedited_percentage": xl_df["PE2df_percentageunedited"],
            "indel_percentage": xl_df["PE2df_percentageindel"],
            "initial_sequence": xl_df["wide_initial_target"],
            "mutated_sequence": xl_df["wide_mutated_target"],
            "protospacer_location": xl_df["protospacerlocation_only_initial"],
            "pbs_location": xl_df["PBSlocation"].map(_shift_location_end),
            "rt_initial_location": xl_df["RT_initial_location"],
            "rt_mutated_location": xl_df["RT_mutated_location"],
        }
    )

    missing_values = csv_df[CRISPAIPE_COLUMNS].isna().any(axis=1).sum()
    if missing_values:
        raise ValueError(
            f"{missing_values} Library-1 rows contain missing crispAIPE input values"
        )
    return csv_df[CRISPAIPE_COLUMNS]


def target_disjoint_split(
    csv_path,
    excel_path,
    output_dir,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    seed=42,
):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, (
        f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"
    )

    xl_df = pd.read_excel(excel_path, sheet_name="Library_1")

    if csv_path:
        csv_df = pd.read_csv(csv_path)
    else:
        csv_df = build_crispaipe_library1_from_excel(xl_df)

    xl_subset = xl_df[
        ["wide_initial_target", "wide_mutated_target", "Name", "Gene"]
    ].copy()
    merged = csv_df.merge(
        xl_subset,
        left_on=["initial_sequence", "mutated_sequence"],
        right_on=["wide_initial_target", "wide_mutated_target"],
        how="left",
    )

    missing_names = merged["Name"].isna().sum()
    assert missing_names == 0, (
        f"{missing_names} rows could not be mapped to PRIDICT mutation names"
    )

    print(f"Total samples: {len(merged)}")
    print(f"Unique mutations (Name): {merged['Name'].nunique()}")

    mutation_stats = merged.groupby("Name").agg(
        mean_edited=("edited_percentage", "mean"),
        count=("edited_percentage", "size"),
    ).reset_index()

    mutation_stats["eff_quartile"] = pd.qcut(
        mutation_stats["mean_edited"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )

    remaining_frac = val_frac + test_frac
    train_mutations, remaining_mutations = train_test_split(
        mutation_stats["Name"].values,
        test_size=remaining_frac,
        random_state=seed,
        stratify=mutation_stats["eff_quartile"].values,
    )

    remaining_stats = mutation_stats[
        mutation_stats["Name"].isin(remaining_mutations)
    ]
    relative_test_frac = test_frac / remaining_frac
    val_mutations, test_mutations = train_test_split(
        remaining_stats["Name"].values,
        test_size=relative_test_frac,
        random_state=seed,
        stratify=remaining_stats["eff_quartile"].values,
    )

    train_mutations = set(train_mutations)
    val_mutations = set(val_mutations)
    test_mutations = set(test_mutations)

    assert len(train_mutations & val_mutations) == 0
    assert len(train_mutations & test_mutations) == 0
    assert len(val_mutations & test_mutations) == 0

    train_df = merged[merged["Name"].isin(train_mutations)].copy()
    val_df = merged[merged["Name"].isin(val_mutations)].copy()
    test_df = merged[merged["Name"].isin(test_mutations)].copy()

    original_cols = csv_df.columns.tolist()
    train_df = train_df[original_cols]
    val_df = val_df[original_cols]
    test_df = test_df[original_cols]

    print("\n--- Split Summary ---")
    print(
        f"Train: {len(train_df)} samples ({len(train_df) / len(csv_df) * 100:.1f}%), "
        f"{len(train_mutations)} mutations"
    )
    print(
        f"Val:   {len(val_df)} samples ({len(val_df) / len(csv_df) * 100:.1f}%), "
        f"{len(val_mutations)} mutations"
    )
    print(
        f"Test:  {len(test_df)} samples ({len(test_df) / len(csv_df) * 100:.1f}%), "
        f"{len(test_mutations)} mutations"
    )
    print(f"Total: {len(train_df) + len(val_df) + len(test_df)} samples")

    def get_protospacers(df):
        return set(df.apply(extract_protospacer_seq, axis=1))

    train_proto = get_protospacers(train_df)
    val_proto = get_protospacers(val_df)
    test_proto = get_protospacers(test_df)

    print("\n--- Leakage Check ---")
    print(f"Train-Val mutation overlap: {len(train_mutations & val_mutations)}")
    print(f"Train-Test mutation overlap: {len(train_mutations & test_mutations)}")
    print(f"Val-Test mutation overlap: {len(val_mutations & test_mutations)}")
    print(f"Train-Val protospacer overlap: {len(train_proto & val_proto)}")
    print(f"Train-Test protospacer overlap: {len(train_proto & test_proto)}")
    print(f"Val-Test protospacer overlap: {len(val_proto & test_proto)}")

    print("\n--- Efficiency Distribution ---")
    print(f"Train mean edited%: {train_df['edited_percentage'].mean():.2f}")
    print(f"Val   mean edited%: {val_df['edited_percentage'].mean():.2f}")
    print(f"Test  mean edited%: {test_df['edited_percentage'].mean():.2f}")

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "pridict-train_revs.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "pridict-val_revs.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "pridict-test_revs.csv"), index=False)
    print(f"\nSaved to {output_dir}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Optional path to pridict-90k-cleaned.csv. If omitted, the cleaned "
            "crispAIPE Library-1 table is rebuilt from --excel."
        ),
    )
    parser.add_argument(
        "--excel",
        required=True,
        help="Path to original PRIDICT supplementary Excel file",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_disjoint_split(
        args.csv,
        args.excel,
        args.output_dir,
        args.train_frac,
        args.val_frac,
        args.test_frac,
        args.seed,
    )
