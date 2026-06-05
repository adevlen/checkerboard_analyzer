"""
Code to calculate Bliss and Loewe synergy scores and plot the results as heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Synergy:
    def __init__(self, parsed_data, drug1_baseline, drug2_baseline):
        """
        Initializes an instance of the Synergy class.

        Args:
            parsed_data: dictionary containing single drug and combination matrices.
            drug1_baseline: instance of DoseResponseCurve class for drug 1.
            drug2_baseline: instance of DoseResponseCurve class for drug 2.
        """
        self.data = parsed_data
        self.curve_drug1 = drug1_baseline
        self.curve_drug2 = drug2_baseline

    def _get_2D_matrices(self):
        """
        Converts combination DataFrame into 2D NumPy matrixes.

        Returns:
            Tuple of matrices containing drug concentrations and combination responses.
        """
        combo_pivot = self.data["combo_data"].pivot(
            index="drug2_conc", columns="drug1_conc", values="response"
        )

        Eobs_matrix = combo_pivot.to_numpy()
        drug1_conc = combo_pivot.columns.to_numpy()
        drug2_conc = combo_pivot.index.to_numpy()

        # create meshgrid to store all drug combo conc (same shape as response matrix)
        # drug 1 conc down col, drug2 conc across rows
        drug1_conc_2D, drug2_conc_2D = np.meshgrid(drug1_conc, drug2_conc)
        return drug1_conc_2D, drug2_conc_2D, Eobs_matrix

    def calc_bliss(self):
        """
        Calculates the Bliss synergy score for each drug combination.

        Returns:
            NumPy array of Bliss synergy scores.
        """
        drug1_conc_2D, drug2_conc_2D, Eobs_matrix = self._get_2D_matrices()

        # get 2D responses using predict()
        E1 = self.curve_drug1.predict(drug1_conc_2D)
        E2 = self.curve_drug2.predict(drug2_conc_2D)

        expected_bliss = (
            E1 + E2 - (E1 * E2)
        )  # Ebliss = E1 + E2 - (E1*E2) -> reference value
        raw_bliss = Eobs_matrix - expected_bliss

        self.bliss_scores = raw_bliss

    def calc_loewe(self):
        """
        Calculates the Loewe additivity score for each drug combination.

        Returns:
            loewe_additivity: NumPy array of Loewe additivity scores.
        """

        drug1_conc_2D, drug2_conc_2D, Eobs_matrix = (
            self._get_2D_matrices()
        )  # combination conc matrices and responses

        assay_info = self.data["assay_info"]
        # predict baseline conc based on response (solve inverted Hill eqn)
        D1 = self.curve_drug1.inverse_predict(
            Eobs_matrix, label=assay_info["drug1_name"]
        )
        D2 = self.curve_drug2.inverse_predict(
            Eobs_matrix, label=assay_info["drug2_name"]
        )

        # eval Loewe eqn to get combination matrix
        # CI = d1/D1 + d2/D2
        CI = (drug1_conc_2D / D1) + (drug2_conc_2D / D2)
        self.loewe_scores = 1 - CI  # Loewe synergy score

    @staticmethod
    def _format_heatmap_annotations(matrix, fmt=".2f"):
        """
        Build cell labels for a synergy heatmap, marking NaN scores explicitly.

        Args:
            matrix: NumPy array containing synergy scores.
            fmt: specifies how to display floating-point numbers (default = 2 decimal places).
        
        Returns:
            labels: labels to display on synergy heatmap.
        """
        labels = np.empty(matrix.shape, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                labels[i, j] = "NaN" if np.isnan(value) else f"{value:{fmt}}"
        return labels

    @staticmethod
    def _heatmap_display_values(matrix, nan_color=0.0):
        """Return a copy of matrix with NaN replaced for seaborn color mapping.

        Seaborn skips annotations on masked/NaN cells, so callers should pass this
        array as heatmap data while supplying labels from the original matrix.

        Args:
            matrix: NumPy array containing synergy scores.
            nan_color: specifies color for NaN cells (default=color like score of 0).

        Returns:
            Copy of matrix with NaN replaced for seaborn color mapping.
        """
        return np.where(np.isnan(matrix), nan_color, matrix)

    def plot_synergy_heatmaps(self, assay_info, output_dir, synergy_model="both"):
        """
        Plots synergy scores as a heatmap.
        Saves heatmaps as PNG files in the given output directory.

        Args:
            assay_info: dictionary containing drug names, units, cell line.
            output_dir: Path variable specifying the location to save the heatmaps.
            synergy_model: str specifying the synergy model (Bliss, Loewe, or both).
        """

        drug1_name = assay_info["drug1_name"]
        drug2_name = assay_info["drug2_name"]
        units = assay_info["conc_units"]

        if synergy_model == "both":
            matrices_to_plot = {
                "Bliss Independence": self.bliss_scores,
                "Loewe Additivity": self.loewe_scores,
            }
        elif synergy_model == "bliss":
            matrices_to_plot = {"Bliss Independence": self.bliss_scores}
        elif synergy_model == "loewe":
            matrices_to_plot = {"Loewe Additivity": self.loewe_scores}
        else:
            return "Synergy model must be 'bliss , 'loewe', or 'both'"

        pivot_df = self.data["combo_data"].pivot(
            index="drug2_conc", columns="drug1_conc", values="response"
        )
        d1_labels = [f"{c:.1e}" if c > 0 else "0" for c in pivot_df.columns]
        d2_labels = [f"{r:.1e}" if r > 0 else "0" for r in pivot_df.index]

        for name, matrix in matrices_to_plot.items():
            if matrix is None:
                print(f"Skipping {name} plot: matrix has not been calculated yet.")
                continue

            # clip finite values for display; preserve NaN for undefined Loewe scores
            plot_matrix = np.array(matrix, dtype=float, copy=True)
            finite = np.isfinite(plot_matrix)
            if np.any(finite & ((plot_matrix > 1.0) | (plot_matrix < -1.0))):
                plot_matrix[finite] = np.clip(plot_matrix[finite], -1.0, 1.0)
                print(
                    f"Warning: {name} scores clipped to (-1.0, 1.0)."
                )

            annot_matrix = self._format_heatmap_annotations(plot_matrix)
            color_matrix = self._heatmap_display_values(plot_matrix)

            plt.figure(figsize=(7, 6))

            sns.heatmap(
                color_matrix,
                annot=annot_matrix,
                fmt="",
                cmap="coolwarm",
                center=0.0,  # color neutral point is zero
                vmin=-1.0,  # symmetric bounds for color intensity
                vmax=1.0,
                xticklabels=d1_labels,  # ticks are drug conc
                yticklabels=d2_labels,
                cbar_kws={"label": "Synergy Score"},
            )

            # invert y-axis so lowest concentrations start at bottom-left corner
            plt.gca().invert_yaxis()

            plt.title(f"{name} Synergy Spectrum")
            plt.xlabel(f"{drug1_name} Concentration ({units})")
            plt.ylabel(f"{drug2_name} Concentration ({units})")
            plt.tight_layout()

            heatmap_dir = output_dir / "synergy_heatmaps"
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(heatmap_dir / f"{drug1_name}_{drug2_name}_{name}_heatmap.png")
