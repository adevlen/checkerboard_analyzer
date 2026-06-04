"""
Code to model single drug response and calculate synergy scores for two examples:
    RKO cell line: MK-2206 & Dinaciclib combination
    ZR751 cell line: Palbociclib & Ridaforolimus combination
"""

import argparse
import numpy as np
from pathlib import Path
from checkerboard_analyzer.utils import parse_data
from checkerboard_analyzer.fit_curve import DoseResponseCurve
from checkerboard_analyzer.calc_synergy import Synergy


def check_fit_quality(drug_curve):
    """
    Returns False if Hill slope is flat or very steep.
    This indicates an unreliable EC50 and a mathematically invalid Loewe model.

    Args:
        drug_curve: Instance of the DoseResponseCurve class.

    Returns:
        Boolean value indicating whether hill slope is within the range (0.5, 4.0).
    """
    h_val = float(drug_curve.params["h"])
    if h_val >= 0.5 or h_val <= 4.0:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Synergy Analyzer Pipeline")

    # parser args: input data file, concentration units, drug names
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_matrix.xlsx",
        help="Path to the input Excel dataset (default: data/sample_matrix.csv)",
    )

    args = parser.parse_args()
    input_file = args.input

    data = parse_data(input_file)

    assay_info = data["assay_info"]
    drug1_name = assay_info["drug1_name"]
    drug2_name = assay_info["drug2_name"]

    # specify output directory to save plots/tables
    output_dir = Path(f"outputs/{assay_info['cell_line']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # get single drug curves
    doses_1, resps_1 = data["drug1_single"]
    drug1 = DoseResponseCurve(doses_1, resps_1)
    drug1.curve_fit()

    doses_2, resps_2 = data["drug2_single"]
    drug2 = DoseResponseCurve(doses_2, resps_2)
    drug2.curve_fit()

    # save curve fit params to Excel file
    drug1.save_params(drug1_name, output_dir)
    drug2.save_params(drug2_name, output_dir)

    # plot dose-reponse curves
    drug1.plot_curve(drug1_name, output_dir)
    drug2.plot_curve(drug2_name, output_dir)

    # get synergy scores and heatmaps
    analyzer = Synergy(data, drug1, drug2)

    if not check_fit_quality(drug1):
        print(
            f"Hill slope for {drug1_name} fit is unstable: {drug1.params['h']}. \
            Interpret EC50 value with caution."
        )
        print(
            "Loewe is invalid for this combination. outputs/ will include Bliss only."
        )
        analyzer.calc_bliss()
        analyzer.plot_synergy_heatmaps(assay_info, output_dir, synergy_model="bliss")

    if not check_fit_quality(drug2):
        print(
            f"Hill slope for {drug2_name} fit is unstable: {drug2.params['h']}. \
                Interpret EC50 value with caution."
        )
        print(
            "Loewe is invalid for this combination. outputs/ will include Bliss only."
        )
        analyzer.calc_bliss()
        analyzer.plot_synergy_heatmaps(assay_info, output_dir, synergy_model="bliss")

    if check_fit_quality(drug1) and check_fit_quality(drug2):
        analyzer.calc_bliss()
        analyzer.calc_loewe()
        analyzer.plot_synergy_heatmaps(assay_info, output_dir, synergy_model="both")

    print(
        f"\nSynergy analysis complete! \
            Dose-response curves and synergy heatmaps are located in {output_dir}\n"
    )


if __name__ == "__main__":
    main()
