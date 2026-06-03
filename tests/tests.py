"""Unit tests for checkerboard_analyzer package."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
import matplotlib

matplotlib.use("Agg")

from checkerboard_analyzer.fit_curve import DoseResponseCurve
from checkerboard_analyzer.calc_synergy import Synergy
from checkerboard_analyzer import utils


def hill(x, emin=0.05, emax=0.95, ec50=1e-6, h=1.5):
    """Generate synthetic dose-response values using the 4PL Hill equation.

    Independent reference implementation used to build test fixtures; not a
    substitute for ``DoseResponseCurve._hill_equation`` under test.

    Args:
        x: Concentration values.
        emin: Minimum response asymptote.
        emax: Maximum response asymptote.
        ec50: Half-maximal effective concentration.
        h: Hill slope.

    Returns:
        Array of normalized response values.
    """
    return emin + (emax - emin) / (1.0 + (x / ec50) ** h)


def make_concentrations():
    """Build a standard log-spaced concentration series.

    Returns:
        NumPy array of concentrations from 0 through 1e-4.
    """
    return np.array([0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])


def make_responses(concentrations, **hill_kwargs):
    """Generate responses for a concentration series.

    Args:
        concentrations: Dose concentrations corresponding to each response.
        **hill_kwargs: Keyword arguments forwarded to ``hill``.

    Returns:
        NumPy array of response values; zero-dose wells receive ``emax``.
    """
    responses = hill(concentrations, **hill_kwargs)
    responses[concentrations == 0] = hill_kwargs.get("emax", 0.95)
    return responses


def fit_dose_curve(concentrations=None, responses=None, **hill_kwargs):
    """Build and fit a ``DoseResponseCurve`` from synthetic Hill data.

    Args:
        concentrations: Optional concentration array; defaults to ``make_concentrations()``.
        responses: Optional response array; defaults to ``make_responses(concentrations)``.
        **hill_kwargs: Keyword arguments forwarded to ``make_responses``.

    Returns:
        A fitted ``DoseResponseCurve`` instance.
    """
    if concentrations is None:
        concentrations = make_concentrations()
    if responses is None:
        responses = make_responses(concentrations, **hill_kwargs)
    curve = DoseResponseCurve(concentrations, responses)
    curve.curve_fit()
    return curve


def make_combo_dataframe(drug1_concs, drug2_concs):
    """Build combo assay rows for a full checkerboard matrix.

    Args:
        drug1_concs: Drug 1 concentration values.
        drug2_concs: Drug 2 concentration values.

    Returns:
        DataFrame with ``drug1_conc``, ``drug2_conc``, and ``response`` columns.
    """
    rows = []
    for d1 in drug1_concs:
        for d2 in drug2_concs:
            if d1 == 0 and d2 == 0:
                continue
            r1 = hill(d1) if d1 > 0 else 1.0
            r2 = hill(d2) if d2 > 0 else 1.0
            combo_response = r1 + r2 - r1 * r2
            rows.append({"drug1_conc": d1, "drug2_conc": d2, "response": combo_response})
    return pd.DataFrame(rows)


def write_test_excel(path, combo_df=None, extra_data_rows=None, assay_info=None):
    """Write a minimal valid Excel workbook for ``parse_data``.

    Args:
        path: Destination file path for the workbook.
        combo_df: Optional combination data as a DataFrame or list of row dicts.
        extra_data_rows: Optional additional rows appended to the data sheet.
        assay_info: Optional assay metadata dict; defaults to generic test values.

    Returns:
        None
    """
    drug1_concs = make_concentrations()
    drug2_concs = make_concentrations()

    data_rows = []
    for c in drug1_concs:
        if c > 0:
            data_rows.append({"drug1_conc": c, "drug2_conc": 0.0, "response": hill(c)})
    for c in drug2_concs:
        if c > 0:
            data_rows.append({"drug1_conc": 0.0, "drug2_conc": c, "response": hill(c)})

    if combo_df is not None:
        if isinstance(combo_df, pd.DataFrame):
            data_rows.extend(combo_df.to_dict("records"))
        else:
            data_rows.extend(combo_df)
    else:
        data_rows.extend(make_combo_dataframe(drug1_concs[1:], drug2_concs[1:]).to_dict("records"))

    if extra_data_rows:
        data_rows.extend(extra_data_rows)

    df = pd.DataFrame(data_rows)
    info = pd.DataFrame([assay_info or {
        "drug1_name": "DrugA",
        "drug2_name": "DrugB",
        "cell_line": "TestCell",
        "conc_units": "μM",
    }])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="data", index=False)
        info.to_excel(writer, sheet_name="assay_info", index=False)


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect ``utils.data_dir`` to a temporary directory.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Path to the temporary data directory.
    """
    monkeypatch.setattr(utils, "data_dir", tmp_path)
    return tmp_path


@pytest.fixture
def sample_excel(tmp_data_dir):
    """Create a valid sample Excel file in the temporary data directory.

    Args:
        tmp_data_dir: Temporary data directory fixture.

    Returns:
        Path to the generated ``sample_matrix.xlsx`` file.
    """
    path = tmp_data_dir / "sample_matrix.xlsx"
    write_test_excel(path)
    return path


@pytest.fixture
def fitted_curves():
    """Return two fitted single-drug dose-response curves.

    Returns:
        Tuple of two fitted ``DoseResponseCurve`` instances.
    """
    conc = make_concentrations()
    drug1 = fit_dose_curve(conc, make_responses(conc, ec50=1e-7))
    drug2 = fit_dose_curve(conc, make_responses(conc, ec50=5e-7))
    return drug1, drug2


@pytest.fixture
def synergy_data(fitted_curves):
    """Build a parsed-data dictionary suitable for ``Synergy``.

    Args:
        fitted_curves: Tuple of fitted single-drug curve fixtures.

    Returns:
        Dictionary matching the structure returned by ``parse_data``.
    """
    drug1, drug2 = fitted_curves
    conc = make_concentrations()
    combo_df = make_combo_dataframe(conc[1:], conc[1:])
    return {
        "drug1_single": (drug1.concentrations, drug1.responses),
        "drug2_single": (drug2.concentrations, drug2.responses),
        "combo_data": combo_df,
        "assay_info": {
            "drug1_name": "DrugA",
            "drug2_name": "DrugB",
            "cell_line": "TestCell",
            "conc_units": "μM",
        },
    }


class TestDoseResponseCurve:
    """Tests for the ``DoseResponseCurve`` class."""

    def test_init_stores_arrays_and_units(self):
        """Verify constructor stores concentrations, responses, and units."""
        conc = np.array([1e-9, 1e-6])
        resp = np.array([0.9, 0.2])
        curve = DoseResponseCurve(conc, resp, units="nM")
        np.testing.assert_array_equal(curve.concentrations, conc)
        np.testing.assert_array_equal(curve.responses, resp)
        assert curve.units == "nM"
        assert curve.params == {}

    def test_hill_equation(self):
        """Verify the static Hill equation returns expected values at key doses."""
        x = np.array([0.0, 1e-6, 1e-5])
        y = DoseResponseCurve._hill_equation(x, 0.1, 0.9, 1e-6, 1.0)
        assert y[0] == pytest.approx(0.9)  # zero dose -> Emax
        assert y[1] == pytest.approx(0.5)
        assert y[2] < 0.5  # higher dose -> lower response

    def test_initial_guesses_full_data(self):
        """Verify initial guesses fall within optimizer bounds for full datasets."""
        curve = fit_dose_curve()
        guesses = curve.initial_guesses()
        assert len(guesses) == 4
        emin, emax, ec50, h = guesses
        assert 0.0 < emin < 0.4
        assert 0.85 < emax < 1.20
        assert ec50 > 0
        assert 0.2 <= h <= 5.0

    def test_initial_guesses_single_point(self):
        """Verify initial guesses use defaults when only one data point exists."""
        curve = DoseResponseCurve(np.array([1e-6]), np.array([0.5]))
        guesses = curve.initial_guesses()
        assert guesses[1] == pytest.approx(1.0)  # default emax
        assert guesses[0] == pytest.approx(0.1)  # default emin

    def test_initial_guesses_no_nonzero_concentrations(self):
        """Verify EC50 guess is clipped when all concentrations are zero."""
        curve = DoseResponseCurve(np.array([0.0, 0.0]), np.array([0.95, 0.90]))
        guesses = curve.initial_guesses()
        # guessed_ec50 defaults to 1e-3 but clips to max_tested*2 (=0) -> 0.0
        assert guesses[2] == pytest.approx(0.0)

    def test_curve_fit_populates_params(self):
        """Verify a successful fit stores Emin, Emax, EC50, and h in params."""
        curve = fit_dose_curve()
        assert curve.params is not None
        for key in ("Emin", "Emax", "EC50", "h"):
            assert key in curve.params

    def test_curve_fit_runtime_error_sets_params_none(self, capsys):
        """Verify convergence failure sets params to None and prints a message.

        Args:
            capsys: Pytest fixture for capturing stdout.
        """
        curve = DoseResponseCurve(make_concentrations(), make_responses(make_concentrations()))
        with patch("checkerboard_analyzer.fit_curve.curve_fit", side_effect=RuntimeError("fail")):
            curve.curve_fit()
        assert curve.params is None
        captured = capsys.readouterr()
        assert "Curve fitting failed" in captured.out

    def test_predict_before_fit_raises(self):
        """Verify predict raises ValueError when the curve has not been fit."""
        curve = DoseResponseCurve(make_concentrations(), make_responses(make_concentrations()))
        curve.params = None
        with pytest.raises(ValueError, match="not yet been fit"):
            curve.predict(1e-6)

    def test_predict_after_fit(self):
        """Verify predict returns non-negative responses for fitted curves."""
        curve = fit_dose_curve()
        pred = curve.predict(np.array([1e-7, 1e-6]))
        assert pred.shape == (2,)
        assert np.all(pred >= 0)

    def test_inverse_predict_before_fit_raises(self):
        """Verify inverse_predict raises ValueError when the curve has not been fit."""
        curve = DoseResponseCurve(make_concentrations(), make_responses(make_concentrations()))
        curve.params = None
        with pytest.raises(ValueError, match="not yet been fit"):
            curve.inverse_predict(0.5)

    def test_inverse_predict_scalar_and_array(self):
        """Verify inverse_predict returns positive doses for array inputs."""
        curve = fit_dose_curve()
        # scalar inputs are coerced to 0-d arrays; use 1-element array for robustness
        single = curve.inverse_predict(np.array([0.5]))
        assert single.shape == (1,)

        arr = curve.inverse_predict(np.array([0.3, 0.5, 0.7]))
        assert arr.shape == (3,)
        assert np.all(arr > 0)

    def test_inverse_predict_accepts_list_input(self):
        """Verify inverse_predict coerces list inputs to NumPy arrays."""
        curve = fit_dose_curve()
        doses = curve.inverse_predict([0.4, 0.6])
        assert doses.shape == (2,)

    def test_inverse_predict_zero_hill_slope(self):
        """Verify inverse_predict handles a zero Hill slope without error."""
        curve = fit_dose_curve()
        curve.params["h"] = 0.0
        result = curve.inverse_predict(np.array([0.5]))
        assert np.all(result > 0)

    def test_inverse_predict_zero_denominator_branch(self):
        """Verify inverse_predict handles responses at or near Emin."""
        curve = fit_dose_curve()
        emin = curve.params["Emin"]
        # Force denominator == 0 path by setting response exactly to emin after clip logic
        curve.params["h"] = 1.0
        responses = np.array([emin, 0.5])
        doses = curve.inverse_predict(responses)
        assert doses.shape == (2,)

    def test_get_stats_with_replicates(self):
        """Verify get_stats computes mean and standard deviation across replicates."""
        conc = np.array([1e-6, 1e-6, 1e-5, 1e-5])
        resp = np.array([0.8, 0.82, 0.3, 0.28])
        curve = DoseResponseCurve(conc, resp)
        stats = curve.get_stats()
        assert list(stats.columns) == ["concentration", "mean_response", "sd_response"]
        assert len(stats) == 2
        assert stats.loc[stats["concentration"] == 1e-6, "sd_response"].iloc[0] > 0

    def test_get_stats_single_replicate_fills_nan_sd(self):
        """Verify get_stats fills NaN standard deviation with 0 for single replicates."""
        curve = DoseResponseCurve(np.array([1e-6]), np.array([0.5]))
        stats = curve.get_stats()
        assert stats["sd_response"].iloc[0] == 0.0

    def test_plot_curve_before_fit_raises(self, tmp_path):
        """Verify plot_curve raises ValueError when the curve has not been fit.

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        curve = DoseResponseCurve(make_concentrations(), make_responses(make_concentrations()))
        curve.params = None
        with pytest.raises(ValueError, match="not yet been fit"):
            curve.plot_curve("DrugA", tmp_path)

    def test_plot_curve_creates_file(self, tmp_path):
        """Verify plot_curve writes a PNG to the dose_response_curves subdirectory.

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        curve = fit_dose_curve()
        curve.plot_curve("DrugA", tmp_path)
        out = tmp_path / "dose_response_curves" / "DrugA_dose_response.png"
        assert out.exists()

    def test_save_params_creates_excel(self, tmp_path):
        """Verify save_params writes fit parameters to an Excel file.

        Args:
            tmp_path: Pytest temporary directory fixture.
        """
        curve = fit_dose_curve()
        curve.save_params("DrugA", tmp_path)
        out = tmp_path / "parameters" / "DrugA_parameters.xlsx"
        assert out.exists()
        loaded = pd.read_excel(out, index_col=0)
        assert "DrugA" in loaded.columns or len(loaded) == 4


class TestParseData:
    """Tests for the ``parse_data`` utility function."""

    def test_parse_data_success(self, sample_excel):
        """Verify parse_data returns expected keys from a valid Excel file.

        Args:
            sample_excel: Path to a valid temporary Excel workbook fixture.
        """
        result = utils.parse_data(sample_excel.name)
        assert "drug1_single" in result
        assert "drug2_single" in result
        assert "combo_data" in result
        assert "assay_info" in result
        doses_1, resps_1 = result["drug1_single"]
        assert len(doses_1) > 0
        assert len(resps_1) == len(doses_1)
        assert result["assay_info"]["drug1_name"] == "DrugA"

    def test_parse_data_file_not_found(self, tmp_data_dir):
        """Verify parse_data raises FileNotFoundError for a missing file.

        Args:
            tmp_data_dir: Temporary data directory fixture.
        """
        with pytest.raises(FileNotFoundError, match="File not found"):
            utils.parse_data("missing.xlsx")

    def test_parse_data_missing_data_columns(self, tmp_data_dir):
        """Verify parse_data raises ValueError when required data columns are absent.

        Args:
            tmp_data_dir: Temporary data directory fixture.
        """
        path = tmp_data_dir / "bad_data.xlsx"
        df = pd.DataFrame({"wrong_col": [1]})
        info = pd.DataFrame([{"drug1_name": "A", "drug2_name": "B", "cell_line": "C", "conc_units": "μM"}])
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="data", index=False)
            info.to_excel(writer, sheet_name="assay_info", index=False)
        with pytest.raises(ValueError, match="Missing expected data columns"):
            utils.parse_data(path.name)

    def test_parse_data_missing_assay_columns(self, tmp_data_dir):
        """Verify parse_data raises ValueError when assay_info columns are absent.

        Args:
            tmp_data_dir: Temporary data directory fixture.
        """
        path = tmp_data_dir / "bad_info.xlsx"
        df = pd.DataFrame({"drug1_conc": [1], "drug2_conc": [0], "response": [0.5]})
        info = pd.DataFrame([{"drug1_name": "A"}])
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="data", index=False)
            info.to_excel(writer, sheet_name="assay_info", index=False)
        with pytest.raises(ValueError, match="Missing expected assay information columns"):
            utils.parse_data(path.name)

    def test_parse_data_drops_na_rows(self, tmp_data_dir, capsys):
        """Verify parse_data drops NA rows and prints a warning.

        Args:
            tmp_data_dir: Temporary data directory fixture.
            capsys: Pytest fixture for capturing stdout.
        """
        path = tmp_data_dir / "na_rows.xlsx"
        write_test_excel(
            path,
            extra_data_rows=[{"drug1_conc": 1e-6, "drug2_conc": 1e-6, "response": np.nan}],
        )
        result = utils.parse_data(path.name)
        assert result is not None
        captured = capsys.readouterr()
        assert "Dropped" in captured.out

    def test_parse_data_averages_combo_replicates(self, tmp_data_dir):
        """Verify parse_data averages duplicate combination concentration pairs.

        Args:
            tmp_data_dir: Temporary data directory fixture.
        """
        path = tmp_data_dir / "replicates.xlsx"
        combo = pd.DataFrame([
            {"drug1_conc": 1e-6, "drug2_conc": 1e-6, "response": 0.4},
            {"drug1_conc": 1e-6, "drug2_conc": 1e-6, "response": 0.6},
        ])
        write_test_excel(path, combo_df=combo.to_dict("records"))
        result = utils.parse_data(path.name)
        combo_avg = result["combo_data"]
        assert len(combo_avg[combo_avg["drug1_conc"] == 1e-6]) == 1
        assert combo_avg["response"].iloc[0] == pytest.approx(0.5)


class TestSynergy:
    """Tests for the ``Synergy`` class."""

    def test_init(self, synergy_data, fitted_curves):
        """Verify constructor stores parsed data and baseline curve references.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        assert syn.data is synergy_data
        assert syn.curve_drug1 is drug1
        assert syn.curve_drug2 is drug2

    def test_get_2d_matrices(self, synergy_data, fitted_curves):
        """Verify _get_2D_matrices returns aligned concentration and response grids.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        d1, d2, eobs = syn._get_2D_matrices()
        assert d1.shape == d2.shape == eobs.shape
        assert d1.shape[0] == len(synergy_data["combo_data"]["drug2_conc"].unique())

    def test_calc_bliss(self, synergy_data, fitted_curves):
        """Verify calc_bliss produces a score matrix matching the combo grid shape.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        syn.calc_bliss()
        assert syn.bliss_scores.shape == syn._get_2D_matrices()[2].shape

    def test_calc_loewe(self, synergy_data, fitted_curves):
        """Verify calc_loewe produces a score matrix matching the combo grid shape.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        syn.calc_loewe()
        assert syn.loewe_scores.shape == syn._get_2D_matrices()[2].shape

    def test_plot_synergy_heatmaps_both(self, synergy_data, fitted_curves, tmp_path):
        """Verify plot_synergy_heatmaps writes Bliss and Loewe heatmaps when model is 'both'.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
            tmp_path: Pytest temporary directory fixture.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        syn.calc_bliss()
        syn.calc_loewe()
        assay_info = synergy_data["assay_info"]
        syn.plot_synergy_heatmaps(assay_info, tmp_path, synergy_model="both")
        heatmap_dir = tmp_path / "synergy_heatmaps"
        assert heatmap_dir.exists()
        pngs = list(heatmap_dir.glob("*.png"))
        assert len(pngs) == 2

    def test_plot_synergy_heatmaps_bliss_only(self, synergy_data, fitted_curves, tmp_path):
        """Verify plot_synergy_heatmaps writes only a Bliss heatmap when model is 'bliss'.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
            tmp_path: Pytest temporary directory fixture.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        syn.calc_bliss()
        syn.plot_synergy_heatmaps(synergy_data["assay_info"], tmp_path, synergy_model="bliss")
        pngs = list((tmp_path / "synergy_heatmaps").glob("*Bliss*"))
        assert len(pngs) == 1

    def test_plot_synergy_heatmaps_loewe_only(self, synergy_data, fitted_curves, tmp_path):
        """Verify plot_synergy_heatmaps writes only a Loewe heatmap when model is 'loewe'.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
            tmp_path: Pytest temporary directory fixture.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        syn.calc_loewe()
        syn.plot_synergy_heatmaps(synergy_data["assay_info"], tmp_path, synergy_model="loewe")
        pngs = list((tmp_path / "synergy_heatmaps").glob("*Loewe*"))
        assert len(pngs) == 1

    def test_plot_synergy_heatmaps_invalid_model(self, synergy_data, fitted_curves):
        """Verify plot_synergy_heatmaps returns an error message for an invalid model.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        result = syn.plot_synergy_heatmaps(synergy_data["assay_info"], Path("/tmp"), synergy_model="invalid")
        assert "Synergy model must be" in result

    def test_plot_synergy_heatmaps_skips_none_matrix(self, synergy_data, fitted_curves, tmp_path, capsys):
        """Verify plot_synergy_heatmaps skips plots when score matrices are None.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
            tmp_path: Pytest temporary directory fixture.
            capsys: Pytest fixture for capturing stdout.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        syn.bliss_scores = None
        syn.loewe_scores = None
        syn.plot_synergy_heatmaps(synergy_data["assay_info"], tmp_path, synergy_model="both")
        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    def test_plot_synergy_heatmaps_clips_extreme_values(self, synergy_data, fitted_curves, tmp_path, capsys):
        """Verify plot_synergy_heatmaps clips scores outside [-1, 1] and prints a warning.

        Args:
            synergy_data: Parsed assay data fixture.
            fitted_curves: Tuple of fitted single-drug curve fixtures.
            tmp_path: Pytest temporary directory fixture.
            capsys: Pytest fixture for capturing stdout.
        """
        drug1, drug2 = fitted_curves
        syn = Synergy(synergy_data, drug1, drug2)
        syn.calc_bliss()
        syn.bliss_scores[0, 0] = 5.0
        syn.bliss_scores[0, 1] = -5.0
        syn.plot_synergy_heatmaps(synergy_data["assay_info"], tmp_path, synergy_model="bliss")
        captured = capsys.readouterr()
        assert "Clipping values" in captured.out

    def test_plot_synergy_heatmaps_zero_concentration_labels(self, fitted_curves, tmp_path):
        """Verify heatmap axis labels render correctly for zero concentrations.

        Args:
            fitted_curves: Tuple of fitted single-drug curve fixtures.
            tmp_path: Pytest temporary directory fixture.
        """
        drug1, drug2 = fitted_curves
        combo_df = pd.DataFrame([
            {"drug1_conc": 0.0, "drug2_conc": 1e-6, "response": 0.5},
            {"drug1_conc": 1e-6, "drug2_conc": 0.0, "response": 0.5},
            {"drug1_conc": 1e-6, "drug2_conc": 1e-6, "response": 0.3},
        ])
        data = {
            "combo_data": combo_df,
        }
        syn = Synergy(data, drug1, drug2)
        syn.calc_bliss()
        syn.plot_synergy_heatmaps(
            {"drug1_name": "A", "drug2_name": "B", "conc_units": "μM"},
            tmp_path,
            synergy_model="bliss",
        )
        assert (tmp_path / "synergy_heatmaps").exists()
