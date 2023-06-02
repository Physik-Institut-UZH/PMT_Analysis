import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from tqdm import tqdm
from itertools import count
import warnings
from typing import Tuple, Union, Optional


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
                print('Calculating range of area values to be used for the model independent gain calculation.')
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
    def get_area_histogram(input_data: np.ndarray, bin_width: int = 1,
                           limits: Optional[Tuple[int, int]] = None) -> tuple:
        """Function to generate area histogram with given bin width.

        Args:
            input_data: Input pulse area data.
            bin_width: Bin width.
            limits: Range of histogram bins.

        Returns:
            tuple(bins_centers, cnts): Tuple with bins centers and corresponding counts values.
        """
        try:
            bin_width = int(bin_width)
        except Exception:
            raise TypeError('bin_width must be of type int.')
        if limits is not None:
            if len(limits) != 2:
                raise ValueError('Limits must be a tuple of length 2.')
            try:
                limits_lower = int(limits[0])
                limits_upper = int(limits[1])
            except Exception:
                raise TypeError('Entries of limits must be of type int.')
            bins_edges = np.arange(limits_lower - 0.5, limits_upper + 1.5, bin_width)
        else:
            bins_edges = np.arange(np.min(input_data) - 0.5, np.max(input_data) + 1.5, bin_width)
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

    @staticmethod
    def sav_gol_smoothing(input_data: np.ndarray) -> np.ndarray:
        """Smooth `input_data` with Savitzky–Golay filter.

        Args:
            input_data: Input threshold dependent occupancy values.

        Returns:
            output_data: Savitzky–Golay filter smoothed input data.
        """
        # Set length of the filter window
        num = len(input_data) - 1
        if num % 2 == 0:
            num = num - 1
        # Set order of the polynomial used to fit the samples
        order = 10
        order = min(order, num - 1)
        # Apply filter
        output_data = savgol_filter(input_data, num, order)
        return output_data

    def get_occupancy_model_independent(self, areas_led_on: np.ndarray, areas_led_off: np.ndarray) -> dict:
        """Calculate the occupancy for the model independent gain calculation.

        For a range of thresholds within the 0PE peak areas, calculate the number of entries
        below threshold for 'LED off' (`integral_b`) and 'LED on' (`integral_s`) data areas.
        The occupancy can be estimated from the number of 'LED on' sample triggers with
        zero LED-induced photoelectrons, as can be assumed for a sufficiently low threshold
        for `integral_s`, and the total number of sample triggers, as can be by proportion estimated
        through `integral_b`. As the number of photoelectrons produced follows a Poisson distribution,
        we can use the expression `-ln(integral_s/integral_b)` as an estimator for the occupancy for the
        selected threshold. This threshold is supposed to be sufficiently low such that `integral_s` is
        not contaminated by a significant 1PE contribution. We therefore try to find the lowest
        threshold for which sufficient data points are available below threshold to achieve a
        relative occupancy error of below 1%. Ideally this is located in a local and global
        maximum plateau around / slightly below zero areas.

        Args:
            areas_led_on: Array with 'LED on' data pulse areas.
            areas_led_off: Array with 'LED off' data pulse areas.

        Returns:
            occupancy_estimator: Dictionary with the following keys:
                {'occupancy': final occupancy estimate;
                'occupancy_err': uncertainty final occupancy estimate;
                'threshold_occupancy_determination': threshold in 0PE peak area calculations
                    for final occupancy estimate;
                'thr_occ_det_integral_fraction': fraction of entries in 'LED off' data below threshold;
                'tot_entries_b': total number of waveforms in 'LED off' data,
                'iterations': dictionary with the tested thresholds and corresponding
                    occupancy, occupancy uncertainty, and smoothed occupancy values}
        """
        # Total number of waveforms in 'LED off' data
        tot_entries_b = areas_led_off.shape[0]

        # Define list of thresholds to probe in iterative occupancy determination
        if self.outliers_thresholds[0] is None:
            thr_it_occ_det = np.abs(self.get_outlier_bounds(areas_led_on)[0])
        else:
            thr_it_occ_det = np.abs(self.outliers_thresholds[0])
        lower_thr_it_occ_det = - thr_it_occ_det / 2
        upper_thr_it_occ_det = thr_it_occ_det / 2
        list_thr_it_occ_det = np.arange(lower_thr_it_occ_det, upper_thr_it_occ_det + 1)

        # Initialize lists for resulting occupancies as a function of threshold.
        # Use lists with append as it showed to be faster than np.arrays in this application.
        occupancy_list = list()
        occupancy_err_list = list()
        thresholds_list = list()
        f_list = list()

        # Loop over thresholds and calculate corresponding occupancy and error
        for threshold in list_thr_it_occ_det:
            # Calculate number of entries below peak area threshold for 'LED off' (integral_b) and
            # 'LED on' (integral_s) data. Correction factor on integral_b (and later also f) should
            # be close to one and only correct for differences due to outlier removal.
            # If significantly different cardinalities of LED on and off data sets are used,
            # the error calculation below may become incorrect.
            integral_b = np.sum(areas_led_off < threshold) * len(areas_led_on) / len(areas_led_off)
            integral_s = np.sum(areas_led_on < threshold)

            # Perform occupancy calculations only for positive number of entries below threshold
            if integral_b > 0 and integral_s > 0:
                # Fraction of entries in 'LED off' data below threshold
                f = integral_b / (tot_entries_b * len(areas_led_on) / len(areas_led_off))
                # The occupancy can be estimated from the number of 'LED on' sample triggers with zero LED-induced
                # photoelectrons, as can be assumed for a sufficiently low threshold for integral_s, and the
                # total number of sample triggers, as can be by proportion estimated through integral_b.
                # As the number of photoelectrons produced follows a Poisson distribution, we can use the following
                # expression as an estimator for the occupancy for the selected threshold (which, if sufficiently low
                # to not be contaminated by a significant 1PE contribution,
                # should not change the obtained occupancy value).
                l_val = -np.log(integral_s / integral_b)
                l_err = np.sqrt((np.exp(l_val) + 1. - 2. * f) / integral_b)

                # Only consider occupancy values with relative uncertainties below 5%.
                if l_err / l_val <= 0.05:
                    occupancy_list.append(l_val)
                    occupancy_err_list.append(l_err)
                    thresholds_list.append(threshold)
                    f_list.append(f)

        if len(occupancy_list) == 0:
            raise ValueError('No occupancy values determined in threshold iterations.')

        # Convert to numpy arrays
        occupancy_list = np.asarray(occupancy_list)
        occupancy_err_list = np.asarray(occupancy_err_list)
        thresholds_list = np.asarray(thresholds_list)
        f_list = np.asarray(f_list)

        # Smooth threshold-dependent occupancies with Savitzky–Golay filter
        occupancy_list_smooth = self.sav_gol_smoothing(occupancy_list)

        # Sort indices in smoothed occupancy list by their element value in descending order
        occupancy_list_smooth_argsort = occupancy_list_smooth.argsort()[::-1]

        # Find threshold and corresponding occupancy for highest smoothed occupancy value
        # for which the relative occupancy error is below 1%.
        # Ideally this is located in a local and global maximum plateau around / slightly below zero areas
        # where 1PE contributions are negligible but on the other hand sufficient data points are available
        # below the threshold to bring the relative uncertainty down to the required level.
        # If no threshold value fulfils this criterion, return NaN occupancy.
        occupancy_estimate = np.nan
        occupancy_estimate_err = np.nan
        threshold_occ_det = np.nan
        thr_occ_det_integral_fraction = np.nan
        for idx in occupancy_list_smooth_argsort:
            if occupancy_err_list[idx] / occupancy_list[idx] < 0.01:
                occupancy_estimate = occupancy_list[idx]
                occupancy_estimate_err = occupancy_err_list[idx]
                threshold_occ_det = thresholds_list[idx]
                thr_occ_det_integral_fraction = f_list[idx]
                break

        if np.isnan(occupancy_estimate):
            warnings.warn('Failed to estimate occupancy with the required precision. Returned NaN.')
        elif occupancy_estimate <= 0:
            warnings.warn('Warning: Estimated occupancy ({:.3f} ± {:.3f}) seems to be '
                          'less than zero.'.format(occupancy_estimate, occupancy_estimate_err))
        elif self.verbose:
            print('Threshold for occupancy estimation: {}'.format(threshold_occ_det))
            print('Estimated occupancy: {:.3f} ± {:.3f}'.format(occupancy_estimate, occupancy_estimate_err))

        occupancy_estimator = {'occupancy': occupancy_estimate,
                               'occupancy_err': occupancy_estimate_err,
                               'threshold_occupancy_determination': threshold_occ_det,
                               'thr_occ_det_integral_fraction': thr_occ_det_integral_fraction,
                               'tot_entries_b': tot_entries_b,
                               'iterations': {'threshold': thresholds_list,
                                              'occupancy': occupancy_list,
                                              'occupancy_err': occupancy_err_list,
                                              'occupancy_smoothed': occupancy_list_smooth},
                               }

        return occupancy_estimator

    def get_gain_model_independent(self, areas_led_on: np.ndarray, areas_led_off: np.ndarray,
                                   occupancy_estimator: dict, calc_spe_resolution: bool = True) -> dict:
        """Calculate model independent gain value.

        Args:
            areas_led_on: Array with 'LED on' data pulse areas.
            areas_led_off: Array with 'LED off' data pulse areas.
            occupancy_estimator: Output dictionary from `get_occupancy_model_independent` method.
                Must contain at least the following keys: `occupancy`, `occupancy_err`,
                `thr_occ_det_integral_fraction`, `tot_entries_b`.
            calc_spe_resolution: If true, calculate the SPE resolution from mean and variance
                of the single photoelectron response.

        Returns:
            gain_estimator: Dictionary with the following keys:
                {'moments_s': dict with first two moments of 'LED on' area distribution,
                'moments_b': dict with first two moments of 'LED off' area distribution,
                'mean_psi': mean of the single photoelectron response (unconverted gain),
                'var_psi': variance of the single photoelectron response,
                'mean_psi_stat_err': statistical error of mean_psi,
                'mean_psi_sys_err': systematic error of mean_psi,
                'mean_psi_err': total error of mean_psi,
                'spe_resolution': SPE resolution (if `calc_spe_resolution` argument true),
                'spe_resolution_stat_err': statistical error of spe_resolution (if `calc_spe_resolution` argument true),
                'spe_resolution_sys_err': statistical error of spe_resolution (if `calc_spe_resolution` argument true),
                'spe_resolution_err': total error of spe_resolution (if `calc_spe_resolution` argument true)}
        """
        # Get moments of area distributions
        moments_s = self.get_moments(areas_led_on)
        mean_s = moments_s['mean']
        var_s = moments_s['variance']
        moments_b = self.get_moments(areas_led_off)
        mean_b = moments_b['mean']
        var_b = moments_b['variance']

        # Get occupancy related values
        occupancy = occupancy_estimator['occupancy']
        occupancy_err = occupancy_estimator['occupancy_err']
        f_b = occupancy_estimator['thr_occ_det_integral_fraction']
        tot_b = occupancy_estimator['tot_entries_b']

        # Calculate first two central moments of the single photoelectron response
        # and the uncertainties of the mean
        mean_psi = (mean_s - mean_b) / occupancy
        var_psi = (var_s - var_b) / occupancy - mean_psi**2
        mean_psi_stat_err = np.sqrt((occupancy * (mean_psi**2 + var_psi) + 2 * var_b) / (tot_b * occupancy**2) + (
                mean_psi * mean_psi * (np.exp(occupancy) + 1 - 2 * f_b)) / (f_b * tot_b * occupancy**2))
        mean_psi_sys_err = (mean_s - mean_b) * occupancy_err / (occupancy**2)
        mean_psi_err = np.sqrt(mean_psi_stat_err**2 + mean_psi_sys_err**2)

        gain_estimator = {'moments_s': moments_s, 'moments_b': moments_b,
                          'mean_psi': mean_psi, 'variance_psi': var_psi,
                          'mean_psi_stat_err': mean_psi_stat_err, 'mean_psi_sys_err': mean_psi_sys_err,
                          'mean_psi_err': mean_psi_err}

        # Calculate SPE resolution
        if calc_spe_resolution:
            spe_resolution = np.sqrt(var_psi) / mean_psi
            var_psi_stat_err = np.sqrt(((mean_psi ** 2 - var_psi) ** 2 * (np.exp(occupancy) + 1 - 2 * f_b)) / (
                        f_b * tot_b * occupancy ** 2))
            spe_resolution_stat_err = spe_resolution / 2 * var_psi_stat_err / var_psi
            var_psi_sys_err = np.sqrt(
                ((var_s - var_b) * occupancy_err / (occupancy ** 2)) ** 2 + (2 * mean_psi * mean_psi_sys_err) ** 2)
            spe_resolution_sys_err = spe_resolution / 2 * var_psi_sys_err / var_psi
            spe_resolution_err = np.sqrt(spe_resolution_stat_err ** 2 + spe_resolution_sys_err ** 2)
            gain_estimator.update({'spe_resolution': spe_resolution,
                                   'spe_resolution_stat_err': spe_resolution_stat_err,
                                   'spe_resolution_sys_err': spe_resolution_sys_err,
                                   'spe_resolution_err': spe_resolution_err})

        return gain_estimator

    def compute(self, areas_led_on: np.ndarray, areas_led_off: np.ndarray, adc_to_e: float = np.nan) -> dict:
        """Perform full model independent gain and occupancy calculation.

        Args:
            areas_led_on: Array with 'LED on' data pulse areas.
            areas_led_off: Array with 'LED off' data pulse areas.
            adc_to_e: Conversion factor pulse area in ADC units to charge in units of elementary charge.

        Returns:
            output_dict: Dictionary with the following keys:
                {'occupancy': final occupancy estimate;
                'occupancy_err': uncertainty final occupancy estimate;
                'threshold_occupancy_determination': threshold in 0PE peak area calculations
                    for final occupancy estimate;
                'thr_occ_det_integral_fraction': fraction of entries in 'LED off' data below threshold;
                'tot_entries_b': total number of waveforms in 'LED off' data,
                'iterations': dictionary with the tested thresholds and corresponding
                    occupancy, occupancy uncertainty, and smoothed occupancy values,
                'moments_s': dict with first two moments of 'LED on' area distribution,
                'moments_b': dict with first two moments of 'LED off' area distribution,
                'mean_psi': mean of the single photoelectron response (unconverted gain),
                'var_psi': variance of the single photoelectron response,
                'mean_psi_stat_err': statistical error of mean_psi,
                'mean_psi_sys_err': systematic error of mean_psi,
                'mean_psi_err': total error of mean_psi,
                'gain': gain values (in units of read out electrons per induced photoelectron),
                'gain_stat_err': statistical error gain,
                'gain_sys_err': systematic error gain,
                'gain_err': total error gain,
                'outlier_thresholds': range of area values to be used (excluding outliers),
                'histograms:' dictionary with the bin centers and counts for 'LED on' and 'LED off'
                    area histograms with default bin width of 10
                }
        """
        # Calculate occupancy and gain
        occupancy_estimator = self.get_occupancy_model_independent(areas_led_on, areas_led_off)
        gain_estimator = self.get_gain_model_independent(areas_led_on, areas_led_off, occupancy_estimator)

        # Convert to gain values (in units of read out electrons per induced photoelectron)
        if np.isnan(adc_to_e):
            warnings.warn('Unable to convert gain values as no valid input for adc_to_e provided.')
        gain_converted = gain_estimator['mean_psi'] * adc_to_e
        gain_stat_err_converted = gain_estimator['mean_psi_stat_err'] * adc_to_e
        gain_sys_err_converted = gain_estimator['mean_psi_sys_err'] * adc_to_e
        gain_err_converted = gain_estimator['mean_psi_err'] * adc_to_e
        if self.verbose:
            print('Estimated gain [10^6]: {:.3f} ± {:.3f} (stat) ± {:.3f} (syst)'.format(
                gain_converted*1e-6, gain_stat_err_converted*1e-6, gain_sys_err_converted*1e-6))

        gain_estimator_converted = {'gain': gain_converted, 'gain_stat_err': gain_stat_err_converted,
                                    'gain_sys_err': gain_sys_err_converted, 'gain_err': gain_err_converted}

        # Generate histograms for later reference
        hist_bin_centers, hist_counts_led_on = self.get_area_histogram(areas_led_on, bin_width=10,
                                                                       limits=self.outliers_thresholds)
        hist_bin_centers, hist_counts_led_off = self.get_area_histogram(areas_led_off, bin_width=10,
                                                                        limits=self.outliers_thresholds)
        histograms = {'histograms': {'bin_centers': hist_bin_centers,
                                     'counts_led_on': hist_counts_led_on,
                                     'counts_led_off': hist_counts_led_off}}

        # Save everything in a dictionary
        output_dict = dict(**occupancy_estimator, **gain_estimator, **gain_estimator_converted,
                           **{'outlier_thresholds': self.outliers_thresholds}, **histograms)
        return output_dict
