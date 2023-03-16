import pytest
from pmt_analysis.utils.input import *


class TestADCRawData:
    """Tests for `pmt_analysis.utils.input.ADCRawData` class."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_path_adc_data = os.path.join(base_path, 'data',
                                       'SPE_MA0055_MA0058_basev1.03_U0-1000V_U1-1000V_L1_Lamp1.78V_' 
                                       'Loff0.5V_Lwid30ns_Lfreq300Hz_20221018_00')

    def test_incorrect_raw_input_path(self):
        """Raise `ValueError` if no files found to be loaded."""
        with pytest.raises(ValueError):
            ADCRawData(self.input_path_adc_data[:-1])

    def test_incorrect_raw_input_filepattern(self):
        """Raise `TypeError` if no `.root` files among the selected files."""
        with pytest.raises(TypeError):
            ADCRawData(self.input_path_adc_data, raw_input_filepattern='*.xml')

    def test_incorrect_adc_type(self):
        """Raise `ValueError` if no valid option for `adc_type` selected."""
        with pytest.raises(ValueError):
            ADCRawData(self.input_path_adc_data, adc_type='v1234')

    def test_adc_f(self):
        """Assert that correct ADC sampling frequency deduced."""
        adc_test_data = ADCRawData(self.input_path_adc_data)
        adc_test_data_adc_f = adc_test_data.adc_f
        assert adc_test_data_adc_f == 500e6

    def test_trees(self):
        """Assert that correct tree name extracted from test data."""
        adc_test_data = ADCRawData(self.input_path_adc_data)
        adc_test_data_tree = adc_test_data.get_trees()
        assert adc_test_data_tree == 't1'

    def test_branches(self):
        """Assert that correct branch names extracted from test data."""
        adc_test_data = ADCRawData(self.input_path_adc_data)
        adc_test_data_branches = adc_test_data.get_branches()
        expectation = np.array(['wf0', 'wf1', 'Time'])
        assert len(adc_test_data_branches) == len(expectation), f"Unexpected number of branches in test ROOT file."
        assert np.all(adc_test_data_branches == expectation), f"Unexpected branch in test ROOT file."

    def test_shape_wf0(self):
        """Assert that loaded channel 0 test data of correct shape and loading with int branch working."""
        adc_test_data = ADCRawData(self.input_path_adc_data)
        adc_test_data_wf0 = adc_test_data.get_branch_data(0)
        assert adc_test_data_wf0.shape == (2000, 250)

    def test_shape_time(self):
        """Assert that loaded time stamp test data of correct shape."""
        adc_test_data = ADCRawData(self.input_path_adc_data)
        adc_test_data_time = adc_test_data.get_branch_data('Time')
        assert adc_test_data_time.shape == (2000,)

    def test_incorrect_branch(self):
        """Raise `ValueError` if unavailable branch name selected."""
        with pytest.raises(ValueError):
            adc_test_data = ADCRawData(self.input_path_adc_data)
            adc_test_data.get_branch_data('wf2')

    def test_incorrect_branch_type(self):
        """Raise `TypeError` if incorrect branch name type is passed, here a list of strings."""
        with pytest.raises(TypeError):
            adc_test_data = ADCRawData(self.input_path_adc_data)
            adc_test_data.get_branch_data(['wf0', 'wf1'])  # type: ignore


class TestScalerRawData:
    """Tests for `pmt_analysis.utils.input.ScalerRawData` class."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_path_scaler_data = os.path.join(base_path, 'data', 'scaler')

    def test_input_formats(self):
        """Test different allowed and unsupported input formats."""
        assert len(ScalerRawData(files=self.input_path_scaler_data).files) == 2
        assert len(ScalerRawData(files=os.path.join(self.input_path_scaler_data,
                                                    'Scaler_test_data_1_221206.dat')).files) == 1
        assert 'Scaler_test_data_1_221206.dat' in ScalerRawData(files=os.path.join(self.input_path_scaler_data,
                                                                'Scaler_test_data_1_221206.dat')).files[0]
        assert len(ScalerRawData(files=[os.path.join(self.input_path_scaler_data, 'Scaler_test_data_1_221206.dat'),
                                        os.path.join(self.input_path_scaler_data, 'Scaler_test_data_2_221206.dat'),
                                        os.path.join(self.input_path_scaler_data, 'does_not_exist.root')
                                        ]).files) == 2
        with pytest.warns(UserWarning):
            ScalerRawData(files=[os.path.join(self.input_path_scaler_data, 'Scaler_test_data_1_221206.dat'),
                                 os.path.join(self.input_path_scaler_data, 'Scaler_test_data_2_221206.dat'),
                                 os.path.join(self.input_path_scaler_data, 'does_not_exist.root')
                                 ])
        with pytest.raises(ValueError):
            ScalerRawData(files=os.path.join(self.base_path, 'data'))
        with pytest.raises(ValueError):
            ScalerRawData(files=os.path.join(self.base_path, 'does_not_exist'))
        with pytest.raises(TypeError):
            ScalerRawData(files=123)  # type: ignore
        with pytest.raises(ValueError):
            ScalerRawData(files=[os.path.join(self.input_path_scaler_data, 'does_not_exist.root')])

    def test_t_int(self):
        """Test correct extraction of data acquisition interval."""
        assert ScalerRawData(files=self.input_path_scaler_data).t_int == 1

    def test_get_data(self):
        """Test of `pmt_analysis.utils.input.ScalerRawData.get_data` method and its outputs."""
        data = ScalerRawData(files=self.input_path_scaler_data).get_data()
        assert data.shape == (1998, 5)
        assert data.loc[3, 'ch0_cnts'] == 30
