import pytest
import os
from pmt_analysis.utils.input import ADCRawData
from pmt_analysis.processing.afterpulses import *


class TestAfterPulse:
    """Tests for `pmt_analysis.processing.afterpulses.Afterpulse`."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_path_adc_data = os.path.join(base_path, 'data',
                                       'SPE_MA0055_MA0058_basev1.03_U0-1000V_U1-1000V_L1_Lamp1.78V_'
                                       'Loff0.5V_Lwid30ns_Lfreq300Hz_20221018_00')
    data = ADCRawData(input_path_adc_data).get_branch_data(0)
    ap = AfterPulses(data, adc_f=500e6, verbose=True, pre_filter_threshold=3, pre_filter_threshold_type='std',
                     occupancy=1.7, occupancy_unc=0.2, amp_thr_ap=42)

    def test_init(self):
        """Tests for the init of the `pmt_analysis.processing.afterpulses.Afterpulse` class."""
        ap_tmp = AfterPulses(self.data, adc_f=500e6, verbose=True, pre_filter_threshold=3,
                             pre_filter_threshold_type='std', occupancy=1.7, occupancy_unc=0.2,
                             amp_thr_ap=42)
        assert ap_tmp.input_data.shape == (1540, 250)
        AfterPulses(self.data, adc_f=500e6, verbose=False, pre_filter_threshold=42, pre_filter_threshold_type='abs',
                    occupancy=None, occupancy_unc=None, amp_thr_ap=None)
        with pytest.raises(ValueError):
            AfterPulses(self.data, adc_f=500e6, pre_filter_threshold_type='invalid')

    def test_pre_filter(self):
        """Tests for the init of the `pmt_analysis.processing.afterpulses.Afterpulse.pre_filter` method."""
        self.ap.pre_filter(amplitude_threshold_abs=None, amplitude_threshold_std=3)
        self.ap.pre_filter(amplitude_threshold_abs=-1, amplitude_threshold_std=None)
        with pytest.raises(ValueError):
            self.ap.pre_filter(amplitude_threshold_abs=42, amplitude_threshold_std=3)
        with pytest.raises(ValueError):
            self.ap.pre_filter(amplitude_threshold_abs=None, amplitude_threshold_std=None)

    def test_find_ap(self):
        """Tests for the init of the `pmt_analysis.processing.afterpulses.Afterpulse.find_ap` method."""
        self.ap.find_ap(height=42, distance=6, prominence_std=8)
        assert list(self.ap.__dict__.keys()) == ['adc_f', 'occupancy', 'occupancy_unc', 'amp_thr_ap',
                                                 'input_data', 'input_data_std', 'verbose', 'n_samples',
                                                 'df', 'ap_rate_dict']
        assert self.ap.n_samples == 2000
        assert self.ap.df.shape == (5, 5)
        assert self.ap.df.at[0, 'p0_position'] == 145
        assert self.ap.df.at[1, 'p1_position'] == 233

    def test_constrain_main_peak(self):
        """Tests for the init of the `pmt_analysis.processing.afterpulses.Afterpulse.constrain_main_peak` method."""
        self.ap.constrain_main_peak(trim=False)
        assert np.all(self.ap.df.valid_main_pulse)
        self.ap.constrain_main_peak(trim=True)
        assert self.ap.df.shape == (5, 5)

    def test_get_ap_properties(self):
        """Tests for the init of the `pmt_analysis.processing.afterpulses.Afterpulse.get_ap_properties` method."""
        self.ap.get_ap_properties()
        assert self.ap.df.shape == (5, 15)
        assert self.ap.df.at[0, 't_diff_ns'] == 36
        assert self.ap.df.at[1, 'p1_amplitude'] == 114
        assert not self.ap.df.at[2, 'separable']
        assert self.ap.df.at[3, 'p1_area'] == 478

    def test_multi_ap(self):
        """Tests for the init of the `pmt_analysis.processing.afterpulses.Afterpulse.multi_ap` method.
        The provided test data is not affected by this method and hence does not test its full functionality.
        """
        self.ap.multi_ap()
        assert self.ap.df.shape == (5, 15)

    def test_ap_rate(self):
        """Tests for the init of the `pmt_analysis.processing.afterpulses.Afterpulse.ap_rate` method."""
        self.ap.ap_rate()
        assert list(self.ap.ap_rate_dict.keys()) == ['n_ap', 'n_ap_separable', 'ap_fraction', 'ap_fraction_unc',
                                                     'ap_fraction_separable', 'ap_fraction_separable_unc', 'ap_rate',
                                                     'ap_rate_unc', 'ap_rate_separable', 'ap_rate_separable_unc',
                                                     'amp_thr_ap', 'ap_rate_separable_above_thr',
                                                     'ap_rate_separable_unc_above_thr']
        assert round(self.ap.ap_rate_dict['ap_rate'], 4) == 0.0015
        assert round(self.ap.ap_rate_dict['ap_rate_unc'], 4) == 0.0007
        assert round(self.ap.ap_rate_dict['ap_rate_separable'], 4) == 0.0012

    def test_compute(self):
        """Tests for the init of the `pmt_analysis.processing.afterpulses.Afterpulse.compute` method."""
        ap = AfterPulses(self.data, adc_f=500e6, verbose=True, pre_filter_threshold=3, pre_filter_threshold_type='std',
                         occupancy=1.7, occupancy_unc=0.2, amp_thr_ap=42)
        ap.compute(height=42, distance=6, prominence_std=8, constrain_main_peak=True)
        assert round(ap.ap_rate_dict['ap_rate_separable'], 4) == 0.0012
