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
    input_path_adc_data2 = os.path.join(base_path, 'data',
                                        'model_independent',
                                        'areas_on.npy'
                                        )
    data2 = np.load(input_path_adc_data2)
    input_path_adc_data3 = os.path.join(base_path, 'data',
                                        'model_independent',
                                        'areas_off.npy'
                                        )
    data3 = np.load(input_path_adc_data3)

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

    def test_get_area_histogram(self):
        """Validate get_area_histogram method on test data."""
        assert GainModelIndependent.get_area_histogram(self.data1, 1)[0][432] == 0.0
        assert GainModelIndependent.get_area_histogram(self.data1, 1)[1][432] == 5
        with pytest.raises(TypeError):
            print(GainModelIndependent.get_area_histogram(self.data1, None)[0][432])  # type: ignore

    def test_get_moments(self):
        """Validate get_moments method with dummy data."""
        x = np.array([1, 2, 3, 4, 5])
        y = GainModelIndependent.get_moments(x)
        assert type(y) == dict
        assert y['mean'] == 3.0
        assert y['variance'] == 2.0

    def test_sav_gol_smoothing(self):
        """Validate sav_gol_smoothing method with dummy data."""
        x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3,
                      2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9])
        x = round(GainModelIndependent.sav_gol_smoothing(x)[10], 2)
        assert x == 6.23

    def test_compute(self):
        """Validate compute method with dummy data."""
        x = GainModelIndependent(self.data2, self.data3)
        y = x.compute(self.data2, self.data3)
        for el in ['occupancy', 'occupancy_err', 'thr_occ_det_integral_fraction',
                   'tot_entries_b', 'gain', 'gain_err', 'mean_psi', 'iterations',
                   'spe_resolution', 'spe_resolution_err']:
            assert el in y
        assert round(y['occupancy'], 1) == 1.7
        assert round(y['mean_psi'], -1) == 970
        assert round(y['spe_resolution'], 1) == 0.4
