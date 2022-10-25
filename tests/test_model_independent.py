import pytest
import os
from pmt_analysis.analysis.model_independent import *


class TestGainModelIndependent:
    """Tests for `pmt_analysis.analysis.model_independent.GainModelIndependent` class."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_path_adc_data1 = os.path.join(base_path, 'data',
                                       'SPE_MA0055_MA0058_basev1.03_U0-1000V_U1-1000V_L1_Lamp1.78V_'
                                       'Loff0.5V_Lwid30ns_Lfreq300Hz_20221018_00',
                                       'wf0_areas.npy'
                                       )
    data1 = np.load(input_path_adc_data1)

    def test_inputs_to_numpy_types(self):
        """Tests inputs_to_numpy method for allowed input types."""
        y = np.array([1, 2, 3])
        for el in [np.array([1, 2, 3]), [1, 2, 3], {'area': [1, 2, 3]},
                   pd.Series([1, 2, 3]), pd.DataFrame({'area': [1, 2, 3]}),
                   pd.DataFrame({'area': [1, 2, 3]})['area']]:
            x = GainModelIndependent.inputs_to_numpy(el)
            assert np.all(x == y)
            assert type(x) == np.ndarray

    def test_inputs_to_numpy_raises(self):
        """Tests inputs_to_numpy method for unsupported input types."""
        with pytest.raises(TypeError):
            GainModelIndependent.inputs_to_numpy(0)  # type: ignore
        with pytest.raises(TypeError):
            GainModelIndependent.inputs_to_numpy({'area': 0})  # type: ignore
        with pytest.raises(KeyError):
            GainModelIndependent.inputs_to_numpy({'amplitude': [1, 2, 3]})  # type: ignore
        with pytest.raises(KeyError):
            GainModelIndependent.inputs_to_numpy(pd.DataFrame({'amplitude': [1, 2, 3]}))  # type: ignore

    def test_get_outlier_bounds(self):
        """Validate get_outlier_bounds method on test data."""
        assert GainModelIndependent.get_outlier_bounds(self.data1) == (-1043.0, 5857.0)

    def test_get_histogram_unity_bins(self):
        """Validate get_histogram_unity_bins method on test data."""
        assert GainModelIndependent.get_histogram_unity_bins(self.data1)[0][432] == 0.0
        assert GainModelIndependent.get_histogram_unity_bins(self.data1)[1][432] == 5

    def test_get_moments(self):
        """Validate get_moments method with dummy data."""
        x = np.array([1, 2, 3, 4, 5])
        y = GainModelIndependent.get_moments(x)
        assert type(y) == dict
        assert y['mean'] == 3.0
        assert y['variance'] == 2.0
