# data parsing helpers

# will want RKO MK-2206/Dinaciclib single & combo
# and ZR751 Palbociclib/Ridaforolimus single & combo

import pandas as pd
from pathlib import Path

data_dir = Path("data")


def parse_data(filename):
    """
    Parses input data from Excel and returns a dict with single and combination data.
    Assumes that response data has already been normalized.

    Args:
    filename: Name of the input file (.xlsx or .xls).

    Returns:
        Dictionary with single drug and combination data (concentrations & responses).
    """
    filepath = Path(data_dir / filename)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found at path: {filepath}")

    # read in data from XLSX, checking that header is correct
    df = pd.read_excel(filepath, sheet_name="data", header=0)
    expected_header = ["drug1_conc", "drug2_conc", "response"]
    if not set(expected_header).issubset(df.columns):
        missing = set(expected_header) - set(df.columns)
        raise ValueError(f"Missing expected data columns: {missing}")

    # read in assay information (drug names, cell line)
    info = pd.read_excel(filepath, sheet_name="assay_info", header=0)
    expected_info = ["drug1_name", "drug2_name", "cell_line", "conc_units"]
    if not set(expected_info).issubset(info.columns):
        missing = set(expected_info) - set(info.columns)
        raise ValueError(f"Missing expected assay information columns: {missing}")

    # drop rows with missing data
    initial_row_count = len(df)
    df_clean = df.dropna()
    dropped_count = initial_row_count - len(df_clean)

    if dropped_count > 0:
        print(
            f"Warning: Dropped {dropped_count} rows with missing values."
        )

    # extract drug 1 only data
    df_drug1 = df_clean[(df_clean["drug2_conc"] == 0) & (df_clean["drug1_conc"] > 0)]
    doses_single_1 = df_drug1["drug1_conc"].to_numpy()
    resps_single_1 = df_drug1["response"].to_numpy()

    # extract drug 2 only data
    df_drug2 = df_clean[(df_clean["drug1_conc"] == 0) & (df_clean["drug2_conc"] > 0)]
    doses_single_2 = df_drug2["drug2_conc"].to_numpy()
    resps_single_2 = df_drug2["response"].to_numpy()

    # extract combination data
    # average replicates to allow pivot to 2D matrix inside Synergy class
    df_combo_raw = df_clean[(df_clean["drug1_conc"] > 0) & (df_clean["drug2_conc"] > 0)]
    df_combo_avg = df_combo_raw.groupby(["drug1_conc", "drug2_conc"], as_index=False)[
        "response"
    ].mean()

    return {
        "drug1_single": (doses_single_1, resps_single_1),
        "drug2_single": (doses_single_2, resps_single_2),
        "combo_data": df_combo_avg,
        "assay_info": info.iloc[0].to_dict(),
    }
