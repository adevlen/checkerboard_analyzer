"""
Code to model single drug response using the 4PL Hill equation.
Plots a dose-response curve and saves to output directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class DoseResponseCurve:

    def __init__(self, concentrations, responses, units="μM"):
        """
        Initializes an instance of the DoseResponseCurve class.

        Args:
            concentrations: NumPy array with dose concentrations in uM.
            responses: NumPy array with normalized response values.
            units: str specifying concentration units for the given drug.
        """
        self.concentrations = np.asarray(concentrations)
        self.responses = np.asarray(responses)
        self.units = units
        self.params = {}  # curve fit parameters (Emin, Emax, EC50, h)

    def initial_guesses(self):
        """
        Populates params member variable with initial guesses for Emin, Emax, EC50, h.

        Returns:
            The initial parameter values for Emin, Emax, EC50, and h as a list.
        """
        # sort by conc
        sort_idx = np.argsort(self.concentrations)
        c_sorted = self.concentrations[sort_idx]
        r_sorted = self.responses[sort_idx]

        # remove zero doses for log space scaling calculations
        non_zero_mask = c_sorted > 0
        c_numeric = c_sorted[non_zero_mask]
        r_numeric = r_sorted[non_zero_mask]

        # Emax: take average of two lowest tested doses
        guessed_emax = np.mean(r_sorted[:2]) if len(r_sorted) >= 2 else 1.0

        # Emin: take average of two highest tested doses
        guessed_emin = np.mean(r_sorted[-2:]) if len(r_sorted) >= 2 else 0.1

        # EC50: find midpoint between Emin and Emax
        midpoint = guessed_emin + (guessed_emax - guessed_emin) / 2.0

        # find tested concentration closest to response midpoint
        if len(r_numeric) > 0:
            closest_idx = np.argmin(np.abs(r_numeric - midpoint))
            guessed_ec50 = c_numeric[closest_idx]
        else:
            guessed_ec50 = 1e-3

        guessed_h = 1.0  # standard slope

        # ensure guesses are within bounds established in curve_fit (synergy E_bounds)
        safe_emin = np.clip(guessed_emin, 0.0 + 1e-3, 1.2 - 1e-3)
        safe_emax = np.clip(guessed_emax, 0.0 + 1e-3, 1.2 - 1e-3)

        max_tested = np.max(self.concentrations)
        safe_ec50 = np.clip(guessed_ec50, 1e-6, max_tested * 2)
        safe_h = np.clip(guessed_h, 0.2, 5.0)

        return [safe_emin, safe_emax, safe_ec50, safe_h]

    @staticmethod
    def _hill_equation(x, Emin, Emax, EC50, h):
        """
        Defines 4PL Hill equation for given parameters and returns drug response.

        Args:
            x: NumPy array of concentration values.
            Emin: the minimum drug response.
            Emax: the maximum drug response.
            EC50: drug concentration that gives response halfway between Emin and Emax.
            h: the steepness of the dose-response curve (Hill slope).

        Returns:
            The drug response for the given concentrations and parameters.
        """
        return Emin + (Emax - Emin) / (1.0 + (x / EC50) ** h)

    def curve_fit(self):
        """
        Populates params dict and runs non-linear least squares optimization.
        """
        p0 = self.initial_guesses()

        min_tested = np.min(self.concentrations[self.concentrations > 0])

        # prevent EC50 and h from dropping below zero (noise)
        # lower bounds format: [lower_emin, lower_emax, lower_ec50, lower_h]
        # upper bounds format: [upper_emin, upper_emax, upper_ec50, upper_h])
        lower_bounds = [0.0, 0.0, min_tested * 0.01, 0.2]
        upper_bounds = [1.2, 1.2, np.max(self.concentrations) * 10, 5.0]

        # ordinary least squares regression
        try:
            popt, _ = curve_fit(
                self._hill_equation,
                self.concentrations,
                self.responses,
                p0,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                maxfev=10000,
            )  # run optimization

            self.params = {  # store values in params
                "Emin": popt[0],
                "Emax": popt[1],
                "EC50": popt[2],
                "h": popt[3],
            }
        except RuntimeError as e:
            print(f"Curve fitting failed to converge. Error: {e}")
            self.params = None

    def predict(self, x):
        """
        Predict responses for given concentrations using the fitted 4PL model.

        Args:
            x: the concentration(s) to predict a response for.

        Returns:
            The drug response for the given concentrations and parameters.
        """
        if self.params is None:
            raise ValueError("DoseResponseCurve instance has not yet been fit.")
        else:
            return self._hill_equation(
                x,
                self.params["Emin"],
                self.params["Emax"],
                self.params["EC50"],
                self.params["h"],
            )

    def inverse_predict(self, response, label=None):
        """
        Calculates drug concentration based on response.

        Args:
            response: NumPy array of response values as percentages.
            label: Optional drug name; when set, warnings refer to Loewe synergy scores.

        Returns:
            precdicted doses: the drug concentrations for the given responses.
        """
        if self.params is None:
            raise ValueError("DoseResponseCurve instance has not yet been fit.")

        if not isinstance(response, np.ndarray):
            response = np.asarray(response)

        emin = self.params["Emin"]
        emax = self.params["Emax"]
        ec50 = self.params["EC50"]
        h = self.params["h"]

        # print warning before clipping response values
        outside = (response <= emin) | (response >= emax)
        n_out = int(np.sum(outside))
        if n_out > 0:
            name = f" for {label}" if label else ""
            print(
                f"Warning: {n_out} response(s) outside the fitted Hill range{name} "
                f"({emin:.4g}–{emax:.4g}); values will be set to NaN for inversion."
            )
            if label is not None:
                print("Loewe synergy scores for affected combination cells are undefined.")
            else:
                print("Inferred doses for those values are undefined.")

        # if outside [Emin, Emax], set to NaN
        # otherwise, calculate Hill inverse
        predicted_doses = np.full(np.shape(response), np.nan, dtype=float)
        valid = ~outside

        if np.any(valid):
            y = response[valid]
            numerator = np.abs(emax - y)
            denominator = np.abs(y - emin)
            denominator[denominator == 0] = 1e-8

            second_term = numerator / denominator
            h_val = np.abs(h) or 0.1  # if h==0

            predicted_doses[valid] = ec50 * (second_term ** (1.0 / h_val))
        return predicted_doses

    def get_stats(self):
        """
        Groups technical replicates ahead of plotting dose-response curve.
        Allows response data to be plotted as mean +/- standard deviation.

        Returns:
            stats_df: DataFrame with mean & SD of response mapped to concentration.
        """

        df_stats = (
            pd.DataFrame(
                {"concentration": self.concentrations, "response": self.responses}
            )
            .groupby("concentration")
            .agg(mean_response=("response", "mean"), sd_response=("response", "std"))
            .reset_index()
        )

        # fill NaN values with 0.0 (conc has fewer than max replicates)
        df_stats["sd_response"] = df_stats["sd_response"].fillna(0.0)
        return df_stats

    def plot_curve(self, drug_name, output_dir):
        """
        Plots the dose-response curve (% response vs log concentration).
        Reports the EC50 in the title.

        Args:
            drug_name: name of the drug to create plot for.
            output_dir: Path variable specifying the location to save the plots.
        """
        if self.params is None:
            raise ValueError("DoseResponseCurve instance has not yet been fit.")

        positive_conc = self.concentrations[self.concentrations > 0]
        if positive_conc.size == 0:
            raise ValueError(
                "Cannot plot dose-response curve: no positive concentrations."
            )
        conc_min = float(np.min(positive_conc))
        conc_max = float(np.max(positive_conc))
        # extend curve slightly below/above tested range on log scale
        log_conc = np.logspace(
            np.log10(conc_min) - 0.5,
            np.log10(conc_max) + 0.5,
            100,
        )
        pred_responses = self.predict(log_conc)
        df_stats = self.get_stats()

        plt.figure(figsize=(8, 5))

        # plot data as mean +/- standard deviation of technical replicates
        plt.errorbar(
            x=df_stats["concentration"],
            y=df_stats["mean_response"],
            yerr=df_stats["sd_response"],
            fmt="o",
            color="blue",
            ecolor="blue",
            capsize=4,
            markersize=6,
            label="Data Points (Mean ± SD)",
        )

        plt.plot(log_conc, pred_responses, label=None, color="black")  # add 4PL fit

        plt.xscale("log")
        plt.xlabel(f"Log Concentration ({self.units})", fontsize=14)
        plt.ylabel("Response (%)", fontsize=14)
        plt.tick_params(axis="both", labelsize=12)
        plt.suptitle(f"Dose-Response Curve for Drug {drug_name}", fontsize=16)
        plt.title(f'EC50: {self.params["EC50"]:.2e}{self.units}', fontsize=14)

        # plot lines to indicate location of EC50
        midpoint = (
            self.params["Emin"] + (self.params["Emax"] - self.params["Emin"]) / 2.0
        )
        plt.axvline(
            x=self.params["EC50"], color="red", alpha=0.5, linestyle="--", label=None
        )
        plt.axhline(y=midpoint, color="red", alpha=0.5, linestyle="--", label=None)

        curve_dir = output_dir / "dose_response_curves"
        curve_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(curve_dir / f"{drug_name}_dose_response.png")

    def save_params(self, drug_name, output_dir):
        """
        Writes curve parameters to an Excel file for the given drug.

        Args:
            drug_name: name of the drug to save curve fit parameters for.
            output_dir: Path variable specifying the location to save the Excel file.
        """
        params_dir = output_dir / "parameters"
        params_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame.from_dict(self.params, orient="index").to_excel(
            params_dir / f"{drug_name}_parameters.xlsx"
        )
