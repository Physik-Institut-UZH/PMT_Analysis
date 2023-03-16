"""Script for afterpulse processing and analysis."""

# Imports
from pmt_analysis.utils.input import ADCRawData
from pmt_analysis.processing.afterpulses import AfterPulses
from pmt_analysis.plotting.afterpulses import PlottingAfterpulses
import argparse

# Argument parser
parser = argparse.ArgumentParser(
    description='Script for afterpulse processing and analysis.'
)
parser.add_argument('-p', '--input_path',
                    help='Path with the data ROOT files to be used.',
                    type=str,
                    required=True)
parser.add_argument('-c', '--channel',
                    help='ADC channel number.',
                    type=int,
                    required=True)
parser.add_argument('-z', '--height',
                    help='Required height of peaks.',
                    type=float,
                    required=True)
parser.add_argument('-v', '--verbose',
                    help='Set verbose output.',
                    default=True,
                    type=bool)
parser.add_argument('-f', '--pre_filter_threshold',
                    help='Amplitude threshold to exclude waveforms that do not contain any entries above threshold.',
                    default=3,
                    type=float)
parser.add_argument('-tp', '--pre_filter_threshold_type',
                    help='`abs` for absolute threshold or `std` for thresholds of multiples of the baseline std.',
                    default='std',
                    type=str)
parser.add_argument('-o', '--occupancy',
                    help='Occupancy of the main pulse in PE per (LED) trigger.',
                    default=None,
                    type=float)
parser.add_argument('-g', '--gain',
                    help='PMT gain (in units of read out electrons per induced photoelectron).',
                    default=None,
                    type=float)
parser.add_argument('-u', '--occupancy_unc',
                    help='Occupancy uncertainty of the main pulse in PE per (LED) trigger.',
                    default=None,
                    type=float)
parser.add_argument('-ta', '--area_thr_ap',
                    help='Lower area threshold for afterpulse candidates.',
                    default=None,
                    type=float)
parser.add_argument('-tt', '--t_thr_ap',
                    help='Lower time threshold for afterpulse candidates.',
                    default=None,
                    type=float)
parser.add_argument('-d', '--distance',
                    help='Required minimal horizontal distance in samples between neighbouring peaks.',
                    default=6,
                    type=float)
parser.add_argument('-pr', '--prominence_std',
                    help='Required prominence of peaks in units of baseline standard deviations.',
                    default=8,
                    type=float)
parser.add_argument('-m', '--constrain_main_peak',
                    help='Remove events where the first found pulse is not a viable candidate for the main pulse.',
                    default=True,
                    type=bool)
args = parser.parse_args()


def compute() -> tuple:
    """Perform all steps for afterpulse processing, analysis, and plotting.

    Returns:
        df: Pandas dataframe with after pulse candidates and their properties.
        ap_rate_dict: Dictionary with resulting afterpulse rate value and uncertainty.
    """
    # Load data
    raw_data = ADCRawData(args.input_path, verbose=args.verbose)
    data = raw_data.get_branch_data(args.channel)
    # Set ADC properties
    adc_area_to_e = raw_data.adc_area_to_e
    adc_f = raw_data.adc_f
    if args.verbose:
        print('Inferred ADC settings: adc_area_to_e {}, adc_f {}'.format(adc_area_to_e, adc_f))
    # Define `AfterPluses` class object
    ap = AfterPulses(input_data=data,
                     adc_f=adc_f,
                     verbose=args.verbose,
                     pre_filter_threshold=args.pre_filter_threshold,
                     pre_filter_threshold_type=args.pre_filter_threshold_type,
                     occupancy=args.occupancy,
                     occupancy_unc=args.occupancy_unc,
                     area_thr_ap=args.area_thr_ap,
                     t_thr_ap=args.t_thr_ap
                     )
    # Afterpulse processing and analysis
    ap.compute(height=args.height,
               distance=args.distance,
               prominence_std=args.prominence_std,
               constrain_main_peak=args.constrain_main_peak
               )
    # Display results
    if args.verbose:
        print('\nFound afterpulse candidates:\n')
        print(ap.df)
        print('\nDetermined afterpulse rates:\n')
        print('{:<32} {:<12}'.format('Parameter', 'Value'))
        print('{:<32} {:<12}'.format('---------', '-----'))
        for key, val in ap.ap_rate_dict.items():
            if val is None:
                val = '-'
            print("{:<32} {:<12}".format(key, val))
    # Plotting
    if args.verbose:
        print('Generating essential plots.')
    plotting_ap = PlottingAfterpulses(df=ap.df,
                                      adc_f=adc_f,
                                      ap_rate_dict=ap.ap_rate_dict,
                                      adc_area_to_e=adc_area_to_e,
                                      gain=args.gain
                                      )
    plotting_ap.plot_essentials()

    return ap.df, ap.ap_rate_dict


# MAIN
if __name__ == '__main__':
    estimates = compute()
    if args.verbose:
        print('Done.')
