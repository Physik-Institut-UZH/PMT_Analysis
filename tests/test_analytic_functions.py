import pytest
from pmt_analysis.utils.analytic_functions import *


class TestAnalyticFunctions:
    """Tests for analytic functions defined in `pmt_analysis.utils.analytic_functions`."""

    def test_func_gain_vs_voltage(self):
        """Tests of `func_gain_vs_voltage`"""
        with pytest.raises(ValueError):
            func_gain_vs_voltage(np.array([-1000]), 1, 1)
        with pytest.raises(ValueError):
            func_gain_vs_voltage(np.array([1000]), 1, -1)
        with pytest.warns(UserWarning):
            func_gain_vs_voltage(np.array([1000]), 1, 1, 5)
        assert np.all(np.isclose(func_gain_vs_voltage(np.array([800, 900, 1000]), 0.247, 0.640, 10),
                                 np.array([694756, 1476415, 2897720]),
                                 atol=1))
