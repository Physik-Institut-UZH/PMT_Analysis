import pytest
import os
from pmt_analysis.processing.basics import *


class TestFixedWindow:
    """Tests for `pmt_analysis.processing.basics.FixedWindow` class."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_path_adc_data = os.path.join(base_path, 'data',
                                       'SPE_MA0055_MA0058_basev1.03_U0-1000V_U1-1000V_L1_Lamp1.78V_'
                                       'Loff0.5V_Lwid30ns_Lfreq300Hz_20221018_00',
                                       'wf0.npy'
                                       )
    # Load test data. Use extracted data in .npy file instead of loading the ROOT file data for more robustness.
    data = np.load(input_path_adc_data)

    def test_init(self):
        """Init of FixedWindow class with valid inputs."""
        FixedWindow((0, 100), (100, None))
        FixedWindow(np.array([0, 100]), [100, None])  # type: ignore

    def test_bounds_not_iter(self):
        """Raise `TypeError` if input to set_bounds not iterable."""
        with pytest.raises(TypeError):
            FixedWindow(0, (100, None))  # type: ignore

    def test_bounds_incorrect_length(self):
        """Raise `ValueError` if iterable passed to set_bounds not of length 2."""
        with pytest.raises(ValueError):
            FixedWindow((0, 100, 1), (100, None))  # type: ignore

    def test_bounds_conversion_float(self):
        """Convert float in set_bounds input tuple to int."""
        x = FixedWindow((0.0, '100'), (100, None))  # type: ignore
        assert type(x.bounds_baseline[0]) == int

    def test_bounds_failing_conversion(self):
        """Raise `TypeError` if failing to convert elements in set_bounds input tuple to int."""
        with pytest.raises(TypeError):
            FixedWindow(('a', 100), (100, None))  # type: ignore

    def test_bounds_output(self):
        """Check for correct output of set_bounds."""
        x = FixedWindow((0.0, '100'), (100, None))  # type: ignore
        assert x.bounds_baseline == (0, 100)
        assert x.bounds_peak == (100, None)

    def test_baseline(self):
        """Check baseline calculation fixed window."""
        x_bsl = FixedWindow((0, 100), (100, None)).get_baseline(self.data)
        assert x_bsl.shape == (2000,)
        assert x_bsl[2] == 15497.5
        assert x_bsl[42] == 15499.0
        assert x_bsl[1234] == 15500.0

    def test_baseline_std(self):
        """Check baseline standard deviation calculation fixed window."""
        x_bsl_std = FixedWindow((0, 100), (100, None)).get_baseline_std(self.data)
        assert x_bsl_std.shape == (2000,)
        assert round(x_bsl_std[1234], 4) == 4.3585

    def test_amplitude(self):
        """Check amplitude calculation fixed window."""
        x_amp = FixedWindow((0, 100), (100, None)).get_amplitude(self.data)
        assert x_amp.shape == (2000,)
        assert x_amp[1234] == 211.0

    def test_peak_position(self):
        """Check peak position calculation fixed window."""
        x_pp = FixedWindow((0, 100), (100, None)).get_peak_position(self.data)
        assert x_pp.shape == (2000,)
        assert x_pp[1234] == 146
        assert x_pp.dtype == np.int64

    def test_area(self):
        """Check areas calculation fixed window."""
        x_area = FixedWindow((0, 100), (100, None)).get_area(self.data)
        assert x_area.shape == (2000,)
        assert x_area[1234] == 1620.0

    def test_basic_properties(self):
        """Check get_basic_properties function."""
        x = FixedWindow((0, 100), (100, None))
        x = x.get_basic_properties(self.data, parameters=['baseline', 'amplitude'])
        assert x.shape == (2000, 2)
        assert type(x) == pd.DataFrame

    def test_basic_properties_input(self):
        """Check get_basic_properties function input sanity."""
        with pytest.raises(TypeError):
            x = FixedWindow((0, 100), (100, None))
            x.get_basic_properties(1, parameters=['baseline'])  # type: ignore
        with pytest.raises(ValueError):
            x = FixedWindow((0, 100), (100, None))
            x.get_basic_properties(np.arange(300), parameters=['baseline'])

    def test_basic_properties_boundaries(self):
        """Check get_basic_properties function window boundaries and data shape compatibility."""
        with pytest.raises(ValueError):
            x = FixedWindow((0, 100), (250, None))
            x.get_basic_properties(self.data, parameters=['baseline'])
        with pytest.raises(ValueError):
            x = FixedWindow((300, 400), (100, None))
            x.get_basic_properties(self.data, parameters=['baseline'])


class TestFullWindow:
    """Tests for `pmt_analysis.processing.basics.FullWindow` class."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_path_adc_data = os.path.join(base_path, 'data',
                                       'SPE_MA0055_MA0058_basev1.03_U0-1000V_U1-1000V_L1_Lamp1.78V_'
                                       'Loff0.5V_Lwid30ns_Lfreq300Hz_20221018_00',
                                       'wf0.npy'
                                       )
    # Load test data. Use extracted data in .npy file instead of loading the ROOT file data for more robustness.
    data = np.load(input_path_adc_data)

    def test_init(self):
        """Init of FixedWindow class with valid inputs."""
        x = FullWindow()
        assert x.bounds_baseline == (None, None)
        assert x.bounds_peak == (None, None)

    def test_baseline(self):
        """Check baseline calculation full window."""
        x_bsl = FullWindow(n_slices=7).get_baseline(self.data)
        assert x_bsl.shape == (2000,)
        assert x_bsl[2] == 15498.0

    def test_baseline_std(self):
        """Check baseline standard deviation calculation full window."""
        x_bsl_std = FullWindow(n_slices=7).get_baseline_std(self.data)
        assert x_bsl_std.shape == (2000,)
        assert round(x_bsl_std[1], 1) == 4.5

    def test_amplitude(self):
        """Check amplitude calculation full window."""
        x_amp = FullWindow(n_slices=7).get_amplitude(self.data)
        assert x_amp.shape == (2000,)
        assert x_amp[1] == 216.0
