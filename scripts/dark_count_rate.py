"""Script for dark count rate calculation from raw data."""

# Imports
from pmt_analysis.utils.input import ADCRawData
from pmt_analysis.processing.basics import FullWindow
from pmt_analysis.analysis.dark_counts import DarkCountRate
import argparse

# Argument parser
parser = argparse.ArgumentParser(
    description='Script for dark count rate calculation from raw data.'
)
parser.add_argument('-p', '--input_path',
                    help='Path with the data ROOT files to be used.',
                    type=str,
                    required=True)
parser.add_argument('-c', '--channel',
                    help='ADC channel number.',
                    type=int,
                    required=True)
parser.add_argument('-t', '--threshold',
                    help='Amplitude threshold to consider pulse as dark count.',
                    type=float,
                    required=True)
parser.add_argument('-v', '--verbose',
                    help='Set verbose output.',
                    default=True,
                    type=bool)
args = parser.parse_args()


def compute() -> tuple:
    """Perform all steps for dark count rate calculation from raw data.

    Returns:
        dc_rate: Dark count rate in units of dark counts / second.
        dc_rate_unc: Dark count rate uncertainty in units of dark counts / second.
    """
    # Load data
    data = ADCRawData(args.input_path, verbose=args.verbose).get_branch_data(args.channel)
    samples_per_wf = data.shape[1]
    # Calculate baseline subtracted amplitudes
    if args.verbose:
        print('Calculating baseline subtracted amplitudes in full waveform window...')
    amplitudes = FullWindow().get_amplitude(data)
    # Calculate dark count rate
    if args.verbose:
        print('Calculating dark count rate...')
    dc_rate = DarkCountRate(amplitudes, args.threshold, samples_per_wf)
    dc_rate, dc_rate_unc = dc_rate.compute()
    if args.verbose:
        print('Dark count rate: {:.2f} ± {:.2f} Hz'.format(dc_rate, dc_rate_unc))

    return dc_rate, dc_rate_unc


# MAIN
if __name__ == '__main__':
    estimates = compute()
