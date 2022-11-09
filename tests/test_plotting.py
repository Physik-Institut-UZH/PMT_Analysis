import pytest
from pmt_analysis.plotting.model_independent import *


class TestPlottingGainModelIndependent:
    """Tests for `pmt_analysis.plotting.model_independent.PlottingGainModelIndependent` class."""

    def test_init(self):
        """Test initialization with invalid inputs."""
        with pytest.raises(NameError):
            PlottingGainModelIndependent(dict(), save_plots=True, save_dir=None)
        with pytest.raises(NameError):
            PlottingGainModelIndependent(dict(), save_plots=True, save_name_suffix=None)
