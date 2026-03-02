import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


"""Utility to aggregate v4 TSV datasets into a single standardized file.

This module is intentionally conservative about what it treats as the main
`sentence` field: it only maps columns that contain the **original/full**
sentence, never shortened or summarized versions. Shortened sentences stay in
their own metadata columns.
"""

# Mapping from various column name variants (lowercased) to our standard names
STANDARD_COL_MAP: Dict[str, str] = {
    # lemma
    "lemma": "lemma",
    "lemmas": "lemma",
    "lemma ": "lemma",
    # sentence / context (only original, not shortened/resumed)
    "sentence": "sentence",  # NS DaFig: original sentence
    "ddo-citat": "sentence",  # SN files: original DDO citation
    "politiken (eller ddo-bakspejlet)-citat (særligt henrik palle), morten mønster og niels krause kjær": "sentence",  # BSP file: full Politiken/DDO-Bakspejlet quote
    # explanations
    "exp1": "exp1",
    "exp1 (true)": "exp1",
    "exp1 ( true)": "exp1",
    "exp2": "exp2",
    "exp2 (concrete/false)": "exp2",
    "exp3": "exp3",
    "exp3 (abstract/false)": "exp3",
    "exp4": "exp4",
    "exp4 (antonym or random)": "exp4",
}


def shorten_dataset_name(name: str) -> str:
    """Return a short code for a verbose v4 dataset name.

    The long names are preserved in the `dataset` column; this is only for
    plotting / compact legends.
    """
    mapping = {
        "BSP adhoc-metaforer fra Politikens anmeldelser": "BSP_pol_ad_hoc",
        "NS DaFig Korpusdata": "NS_DaFig",
        "SN Metaforer DDO emnebaseret": "SN_DDO",
        "SN ad hoc-metaf. fra ofø-citater i DDO (mest type2": "SN_DDO_ad_hoc",
        "SO Unikke danske metaforer fra NODALIDA-data og korpus.dk": "SO_unik",
    }
    return mapping.get(name, name)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns in a DataFrame to standard names where applicable.

    Only known core columns (lemma, sentence, exp1-4) are renamed; all
    other columns are preserved as-is.
    """
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower()
        std = STANDARD_COL_MAP.get(key)
        if std is not None:
            rename_map[col] = std
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def infer_dataset_name(path: Path) -> str:
    """Infer a human-readable dataset name from the filename.

    If the filename contains a " - ", we take the part after it;
    otherwise we use the stem.
    """
    name = path.stem
    if " - " in name:
        # e.g. "Danish_metaphor_benchmark - NS DaFig Korpusdata"
        return name.split(" - ", 1)[1].strip()
    return name


def load_and_standardize_file(path: Path) -> pd.DataFrame:
    """Load a single TSV file and standardize its core columns.

    - Uses tab separator.
    - Normalizes core column names.
    - Adds a "dataset" column from the filename.
    - Keeps all metadata columns as-is.
    """
    df = pd.read_csv(path, sep="\t")
    df = normalize_columns(df)

    # Add dataset identifier
    dataset_name = infer_dataset_name(path)
    df["dataset"] = dataset_name
    df["dataset_short"] = shorten_dataset_name(dataset_name)

    # Ensure core columns exist, even if missing in some files
    for col in ["lemma", "sentence", "exp1", "exp2", "exp3", "exp4"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Remove rows where any explanation is missing
    df = df.dropna(subset=["exp1", "exp2", "exp3", "exp4"], how="any")

    return df


def aggregate_v4(
    input_dir: str = "data/v4",
    output_path: str = "data/v4/combined_v4.tsv",
) -> Path:
    """Aggregate all TSV files in data/v4 into a single standardized file.

    The output will contain at least the following columns:
    - lemma
    - sentence
    - exp1, exp2, exp3, exp4
    - dataset (name inferred from filename)

    All other columns from the source files are preserved as metadata
    (e.g. type, emne, topic, etc.).
    """
    input_dir_path = Path(input_dir)
    output_path_path = Path(output_path)

    # Use all original TSVs in the folder, but skip previously aggregated files
    tsv_files: List[Path] = [
        p for p in sorted(input_dir_path.glob("*.tsv"))
        if not p.stem.startswith("combined_v4")
    ]
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in {input_dir_path}")

    frames: List[pd.DataFrame] = []
    for f in tsv_files:
        df = load_and_standardize_file(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # ---- Canonicalize and coalesce duplicate / semantic metadata columns ----

    # Helper for simple rename-or-coalesce between two columns
    def _merge_two(src: str, dst: str) -> None:
        nonlocal combined
        if src in combined.columns and dst in combined.columns:
            combined[dst] = combined[dst].fillna(combined[src])
            combined.drop(columns=[src], inplace=True)
        elif src in combined.columns and dst not in combined.columns:
            combined.rename(columns={src: dst}, inplace=True)

    # Clear-cut duplicates: same meaning, different spelling/casing
    _merge_two("DDO entry", "DDO_entry")
    _merge_two("Uniqueness", "uniqueness")
    _merge_two("Nats_id", "nats_id")

    # Semantic merges
    def _coalesce_many(sources: List[str], target: str) -> None:
        nonlocal combined
        present = [c for c in sources if c in combined.columns]
        if not present:
            return
        if target not in combined.columns:
            combined[target] = pd.NA
        for c in present:
            if c == target:
                # Already the canonical column; just make sure we use it as a source
                continue
            combined[target] = combined[target].fillna(combined[c])
        # Drop all but the target
        for c in present:
            if c != target and c in combined.columns:
                combined.drop(columns=[c], inplace=True)

    # Shortened sentence variants -> short_sentence
    _coalesce_many(
        [
            "Shortened sentence",
            "citat forkortet/resumeret",
            "DDO-citat forkortet/resumeret",
        ],
        target="short_sentence",
    )

    # Comments / notes -> comment
    _coalesce_many(
        [
            "Comment",
            "comments",
            "Bemærkninger",
        ],
        target="comment",
    )

    # Annotator identifiers -> annotator
    _coalesce_many(
        [
            "Annotator",
            "Annotør",
        ],
        target="annotator",
    )

    # Reorder columns: core columns first, then metadata (alphabetically).
    # We put the short dataset code first, and move the long dataset name last
    # for convenience in plotting/inspection.
    core_cols = ["dataset_short", "lemma", "sentence", "exp1", "exp2", "exp3", "exp4"]
    existing_core = [c for c in core_cols if c in combined.columns]
    other_cols = [c for c in combined.columns if c not in existing_core]
    ordered_cols = existing_core + sorted(other_cols)

    # If we have a long-form dataset column, move it to the very end
    if "dataset" in ordered_cols:
        ordered_cols = [c for c in ordered_cols if c != "dataset"] + ["dataset"]

    combined = combined[ordered_cols]

    # Ensure parent directory exists
    output_path_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path_path, sep="\t", index=False)

    # ---- Basic statistics for sanity checking ----
    total_rows = len(combined)
    datasets = combined["dataset"].unique().tolist() if "dataset" in combined.columns else []

    print("=== Aggregated v4 dataset statistics ===")
    print(f"Output file: {output_path_path}")
    print(f"Total rows: {total_rows}")
    if datasets:
        print(f"Number of datasets: {len(datasets)}")
        print("Datasets:")
        for name in datasets:
            print(f"  - {name}")
        print("\nRows per dataset:")
        print(combined["dataset"].value_counts())

    return output_path_path


if __name__ == "__main__":
    out = aggregate_v4()
    print(f"Wrote aggregated file to: {out}")
