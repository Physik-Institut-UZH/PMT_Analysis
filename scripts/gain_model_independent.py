"""Script for full processing, calculations, and plotting for model independent gains and occupancy."""

# Imports
from pmt_analysis.utils.input import ADCRawData
from pmt_analysis.processing.basics import FixedWindow
from pmt_analysis.analysis.model_independent import GainModelIndependent
from pmt_analysis.plotting.model_independent import PlottingGainModelIndependent
import argparse

# Argument parser
parser = argparse.ArgumentParser(
    description=('Script for full processing, calculations, and plotting for '
                 'model independent gains and occupancy.')
)
parser.add_argument('-pon', '--input_path_on',
                    help='Path with the LED on data ROOT files to be used.',
                    type=str,
                    required=True)
parser.add_argument('-poff', '--input_path_off',
                    help='Path with the LED off data ROOT files to be used.',
                    type=str,
                    required=True)
parser.add_argument('-bbl', '--bsl_bound_lower',
                    help='Lower (included)index of window for baseline calculations.',
                    type=int,
                    default=0)
parser.add_argument('-bbu', '--bsl_bound_upper',
                    help='Upper (excluded) index of window for baseline calculations.',
                    type=int,
                    required=True)
parser.add_argument('-bpl', '--pks_bound_lower',
                    help='Lower (included) index of window for peak calculations.',
                    type=int,
                    required=True)
parser.add_argument('-bpu', '--pks_bound_upper',
                    help='Upper (excluded) index of window for peak calculations.',
                    type=int,
                    default=None)
parser.add_argument('-c', '--channel',
                    help='ADC channel number.',
                    type=int,
                    required=True)
parser.add_argument('-v', '--verbose',
                    help='Set verbose output.',
                    default=True,
                    type=bool)
parser.add_argument('-tr', '--trim_outliers_bool',
                    help='Remove outliers from input data using the `get_outlier_bounds` method.',
                    default=True,
                    type=bool)
args = parser.parse_args()


def compute() -> dict:
    """Perform all steps for model independent gain and occupancy calculation and plotting.

    Returns:
        estimates_dict: Dictionary with results from model independent gain and occupancy computation
            as obtained from `pmt_analysis.analysis.model_independent.GainModelIndependent.compute`.
    """
    # Load data
    data_on = ADCRawData(args.input_path_on, verbose=args.verbose).get_branch_data(args.channel)
    data_off = ADCRawData(args.input_path_off, verbose=args.verbose).get_branch_data(args.channel)
    # Set baseline and peak bounds
    bsl_bounds = (args.bsl_bound_lower, args.bsl_bound_upper)
    pks_bounds = (args.pks_bound_lower, args.pks_bound_upper)
    # Calculate baseline subtracted areas
    areas_on = FixedWindow(bsl_bounds, pks_bounds).get_area(data_on)
    areas_off = FixedWindow(bsl_bounds, pks_bounds).get_area(data_off)
    # Generate model independent gain model
    gain_model_independent = GainModelIndependent(areas_on, areas_off, verbose=args.verbose,
                                                  trim_outliers_bool=args.trim_outliers_bool)
    adc_to_e = ADCRawData(args.input_path_on).adc_area_to_e
    estimates_dict = gain_model_independent.compute(areas_on, areas_off, adc_to_e)
    # Generate plots
    plotting_gain_model_independent = PlottingGainModelIndependent(estimates_dict)
    plotting_gain_model_independent.plot_essentials()
    # Return estimates dictionary
    return estimates_dict


# MAIN
if __name__ == '__main__':
    estimates = compute()
