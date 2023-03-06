import pytest
import os
from pmt_analysis.analysis.dark_counts import *


class TestDarkCountRate:
    """Tests for `pmt_analysis.analysis.dark_counts.DarkCountRate`."""

    def test_init(self):
        """Tests for the init of the `pmt_analysis.analysis.dark_counts.DarkCountRate` class."""
        with pytest.warns(UserWarning):
            DarkCountRate(np.array([1, 2, 3]), -1, 10)
        assert np.all(DarkCountRate(np.array([1, 2, 3]), 2.5, 10).amplitudes == np.array([1, 2, 3]))
        assert DarkCountRate(np.array([1, 2, 3]), 2.5, 10).threshold == 2.5

    def test_correction_poisson(self):
        """Tests of `pmt_analysis.analysis.dark_counts.DarkCountRate.correction_poisson`"""
        assert np.isclose(DarkCountRate.correction_poisson(0.1), 0.105, atol=0.001)
        with pytest.raises(ValueError):
            DarkCountRate.correction_poisson(10)

    def test_compute(self):
        """Tests of `pmt_analysis.analysis.dark_counts.DarkCountRate.compute`"""
        base_path = os.path.abspath(os.path.dirname(__file__))
        input_path_adc_data = os.path.join(base_path, 'data', 'dark_counts', 'amplitudes_off.npy')
        amplitudes = np.load(input_path_adc_data)
        threshold = 42
        samples_per_wf = 250
        adc_f = 500e6
        dc_rate = DarkCountRate(amplitudes, threshold, samples_per_wf, adc_f)
        assert np.all(np.isclose(np.array(dc_rate.compute()), np.array([450, 30]), atol=1))
