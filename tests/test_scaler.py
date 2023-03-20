import os
import pytest
from pmt_analysis.analysis.scaler import *
from pmt_analysis.utils.input import ScalerRawData


class TestScaler:
    """Tests for `pmt_analysis.analysis.scaler.Scaler` class."""
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_path_scaler_data = os.path.join(base_path, 'data', 'scaler')
    scaler_raw_data = ScalerRawData(files=input_path_scaler_data)
    t_int = scaler_raw_data.t_int
    data = scaler_raw_data.get_data()

    def test_init(self):
        """Test init."""
        scaler = Scaler(self.data, t_int=self.t_int, partition_t=None)
        assert scaler.partition_t[0] == 1670329432
        assert scaler.partition_t.shape[0] == 1
        scaler = Scaler(self.data, t_int=self.t_int, partition_t=300)
        assert scaler.partition_t[2] == 1670330032
        assert scaler.partition_t.shape[0] == 6
        scaler = Scaler(self.data, t_int=self.t_int, partition_t=[np.nan, 1670330332, 1670330632])
        assert scaler.partition_t[0] == 1670329432
        assert scaler.partition_t.shape[0] == 3

    def test_raises_and_warnings(self):
        """Test raises and warnings in init."""
        with pytest.raises(TypeError):
            Scaler(self.data, t_int=self.t_int, partition_t='unsupported')  # type: ignore
        with pytest.warns(UserWarning):
            Scaler(self.data, t_int=self.t_int, partition_t=10000)

    def test_get_values(self):
        """Test `pmt_analysis.analysis.scaler.Scaler.get_values` method."""
        scaler = Scaler(self.data, t_int=self.t_int, partition_t=None)
        values_dict = scaler.get_values(channel=0, verbose=True, give_rate=True, margin_start=10, margin_end=10)
        assert values_dict['w_window'][0] == 7
        assert values_dict['mode'][0] == 18.5
        assert values_dict['median'][0] == 22.0
        with pytest.warns(UserWarning):
            scaler.get_values(channel=0, verbose=True, margin_start=10, margin_end=-10)
        with pytest.warns(UserWarning):
            scaler.get_values(channel=0, verbose=True, margin_start=-10, margin_end=10)
        with pytest.raises(ValueError):
            scaler.get_values(channel=0, verbose=True, margin_start=10000)
        scaler = Scaler(self.data, t_int=self.t_int, partition_t=700)
        values_dict = scaler.get_values(channel=1, verbose=True, give_rate=False)
        assert values_dict['w_window'][1] == 10
        assert values_dict['mode'][1] == 19.5
        assert values_dict['median'][1] == 28.0
        with pytest.raises(ValueError):
            scaler.get_values(channel=0, verbose=True, margin_start=500, margin_end=1500)
