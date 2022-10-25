import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import count
from typing import Union


class GainModelIndependent:
    """Class for model independent occupancy and gain calculation.

    Calculation of the gain, i.e. the current amplification factor of the PMT, following the statistical
    method proposed in `Saldanha, R., et al. 'Model independent approach to the single photoelectron
    calibration of photomultiplier tubes.' Nuclear Instruments and Methods in Physics Research Section A:
    Accelerators, Spectrometers, Detectors and Associated Equipment 863 (2017): 35-46`.

    Uses the peak area spectrum from pulsed low-intensity LED light illumination and LED triggered
    data taking. The fixed window analysis takes two data sets, one with LED 'off'
    (practically pulsed at low enough voltage amplitude to ensure no light emission from the LED)
    and one with LED 'on', usually at a target occupancy (i.e. mean number photo electrons released
    per LED trigger) of ca. 2 for uncertainty minimization.

    Attributes:
        areas_led_on: Array with 'LED on' data pulse areas.
        areas_led_off: Array with 'LED off' data pulse areas.
        verbose: Verbosity of the output.
        outliers_thresholds: Range of area values used for the time dependent gain calculation.
    """

    def __init__(self, areas_led_on: np.ndarray, areas_led_off: np.ndarray, verbose: bool = False,
                 trim_outliers_bool: bool = True):
        """Init of the GainModelIndependent class.

        Cleans and prepares the pulse area data inputs.

        Args:
            areas_led_on: Array with 'LED on' data pulse areas.
            areas_led_off: Array with 'LED off' data pulse areas.
            verbose: Verbosity of the output.
            trim_outliers_bool: Remove outliers from input data using the `get_outlier_bounds` method.
        """
        self.verbose = verbose
        # Bring pulse area data inputs to numpy.ndarray format
        self.areas_led_on = self.inputs_to_numpy(areas_led_on)
        self.areas_led_off = self.inputs_to_numpy(areas_led_off)
        # Remove outlier values
        self.outliers_thresholds = (None, None)
        if trim_outliers_bool:
            if verbose:
                print('Calculating range of area values to be used for the time dependent gain calculation.')
            outlier_bound_lower, outlier_bound_upper = self.get_outlier_bounds(self.areas_led_on, self.verbose)
            self.outliers_thresholds = (outlier_bound_lower, outlier_bound_upper)
            if verbose:
                print('Trimming area outliers with limits [{}, {}].'.format(outlier_bound_lower, outlier_bound_upper))
            self.areas_led_off = self.areas_led_off[(self.areas_led_off >= outlier_bound_lower)
                                                    & (self.areas_led_off <= outlier_bound_upper)]
            self.areas_led_on = self.areas_led_on[(self.areas_led_on >= outlier_bound_lower)
                                                  & (self.areas_led_on <= outlier_bound_upper)]

    @staticmethod
    def inputs_to_numpy(input_data: Union[np.ndarray, list, dict, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Bring pulse area data inputs to numpy.ndarray format.

        Args:
            input_data: Input pulse area data.
                Allowed types: `numpy.ndarray, list, dict, pandas.DataFrame, pandas.Series`
        Returns:
            input_data: Input data converted to numpy.ndarray.
        """
        if type(input_data) == np.ndarray:
            pass
        elif type(input_data) == list:
            input_data = np.asarray(input_data)
        elif type(input_data) == dict:
            if 'area' not in input_data.keys():
                raise KeyError('No key name `area` found in provided dictionary `input_data`.')
            elif type(input_data['area']) == np.ndarray:
                input_data = input_data['area']
            elif type(input_data['area']) == list:
                input_data = np.asarray(input_data['area'])
            else:
                raise TypeError('Unsupported type for `input_data`.')
        elif type(input_data) == pd.DataFrame:
            if 'area' not in input_data.columns:
                raise KeyError('No column name `area` found in provided data frame `input_data`.')
            else:
                input_data = input_data['area'].to_numpy()
        elif type(input_data) == pd.Series:
            input_data = input_data.to_numpy()
        else:
            raise TypeError('Unsupported type for `input_data`.')
        return input_data

    @staticmethod
    def get_outlier_bounds(input_data: np.ndarray, verbose: bool = False) -> tuple:
        """Calculate boundaries for outlier rejection.

        Outliers may bias the model independent gain calculation due to their large lever and
        should hence be removed.
        Determined are the values at which the entries in a window of size based on square-root
        binning falls below a certain threshold.

        Args:
            input_data: Input pulse area data.
            verbose: Verbosity of the output.

        Returns:
            Tuple with upper and lower bound to be used for outlier rejection.
        """
        window_width = np.ceil((np.percentile(input_data, 99.99) - np.percentile(input_data, 0.01))
                               / np.sqrt(input_data.shape[0]))
        window_width = 5*window_width
        window_counts_threshold = 1
        for i in tqdm(count(0), disable=not bool(verbose)):
            if len(input_data[(input_data >= i) & (input_data < i + window_width)]) <= window_counts_threshold:
                break
        outlier_bound_upper = i + window_width
        for i in tqdm(count(0), disable=not bool(verbose)):
            if len(input_data[(input_data <= -i) & (input_data > -(i + window_width))]) <= window_counts_threshold:
                break
        outlier_bound_lower = -(i + window_width)
        return outlier_bound_lower, outlier_bound_upper

    @staticmethod
    def get_histogram_unity_bins(input_data: np.ndarray) -> tuple:
        """Function to generate histogram with bin width 1.

        Args:
            input_data: Input pulse area data.

        Returns:
            tuple(bins_centers, cnts): Tuple with bins centers and corresponding counts values.
        """
        bins_edges = np.arange(np.min(input_data) - 0.5, np.max(input_data) + 1.5, 1)
        bins_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
        counts, _ = np.histogram(input_data, bins=bins_edges)
        return bins_centers, counts

    @staticmethod
    def get_moments(input_data: np.ndarray) -> dict:
        """Calculate moments relevant for model independent gains calculation.

        Args:
            input_data: Input pulse area data.

        Returns:
            Dictionary with moments relevant for model independent gains calculation.
        """
        dict_moments = dict()
        dict_moments['mean'] = np.mean(input_data)
        dict_moments['variance'] = np.var(input_data)
        return dict_moments
