"""Script for scaler data loading and plotting."""

# Imports
from pmt_analysis.utils.input import ScalerRawData
from pmt_analysis.plotting.scaler import PlottingScaler
import argparse

# Argument parser
parser = argparse.ArgumentParser(
    description='Script for dark count rate calculation from raw data.'
)
parser.add_argument('-p', '--input_path',
                    help='Path with the scaler data files to be used. '
                         'To select specific files give a full list of the data path and file names.',
                    type=str,
                    required=True)
parser.add_argument('-c', '--channel',
                    help='Scaler channel number.',
                    type=int,
                    required=True)
parser.add_argument('-s', '--t_step_s',
                    help='Plotting time bin width in seconds.',
                    default=300,
                    type=int)
parser.add_argument('-f', '--time_format',
                    help='Time axis format: Options: '
                         '`t_s_rel` time difference in seconds since start of data period;'
                         '`t_h_rel` time difference in hours since start of data period;'
                         '`t_d_rel` time difference in days since start of data period;'
                         '`t_datetime_utc` UTC datetime;'
                         '`t_datetime_zh` datetime for Zurich time zone, i.e. CE(S)T;',
                    default='t_datetime_utc',
                    type=str)
parser.add_argument('-r', '--give_rate',
                    help='If true, use count rates, otherwise use absolute count values.',
                    default=True,
                    type=bool)
parser.add_argument('-v', '--verbose',
                    help='Set verbose output.',
                    default=True,
                    type=bool)
args = parser.parse_args()


def compute():
    """Perform all steps for scaler data loading and plotting."""
    # Load data
    scaler_raw_data = ScalerRawData(files=args.input_path, trim_empty=True, verbose=args.verbose)
    t_int = scaler_raw_data.t_int
    data = scaler_raw_data.get_data()
    # Plot 2d histogram of time-dependent counts / count rates.
    plotting_scaler = PlottingScaler(data=data, t_int=t_int, save_plots=False, show_plots=True)
    plotting_scaler.plot_rate_evolution_hist2d(channel=args.channel, t_step_s=args.t_step_s, give_rate=args.give_rate)


# MAIN
if __name__ == '__main__':
    compute()
    if args.verbose:
        print('Done.')
