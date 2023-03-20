"""Script for scaler data loading and plotting."""

# Imports
from pmt_analysis.utils.input import ScalerRawData
from pmt_analysis.analysis.scaler import Scaler
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
                    help='If true, use count rates in Hz, otherwise use absolute count values.',
                    default=False,
                    type=bool)
parser.add_argument('-v', '--verbose',
                    help='Set verbose output.',
                    default=True,
                    type=bool)
parser.add_argument('-pts', '--partition_ts',
                    help='List of UNIX timestamps partitioning the data.',
                    default=None,
                    type=list)
parser.add_argument('-ptw', '--partition_tw',
                    help='Integer number of seconds indicating the temporal width of the consecutive partitions to '
                         'split the data into.',
                    default=None,
                    type=int)
parser.add_argument('-pv', '--partition_v',
                    help='Label values for partition_t timestamps.',
                    default=None,
                    type=list)
parser.add_argument('-pu', '--partition_v_unit',
                    help='Unit of partition_v values.',
                    default=None,
                    type=list)
parser.add_argument('-m', '--m_value',
                    help='Key of values from `values_dict` to use as characteristic values in partitions. '
                         'Options: `median`, `mean`, `mode`.',
                    default='median',
                    type=str)
args = parser.parse_args()


def compute() -> dict:
    """Perform all steps for scaler data loading and plotting.

    Returns:
        values_dict: Dictionary with following keys:
            `t_start` UNIX timestamp start of partition,
            `t_end` UNIX timestamp end of partition,
            `bins_centers` bin centers histogram for mode determination,
            `cnts` counts histogram for mode determination,
            `w_window` rolling average window width for mode determination,
            `cnts_smoothed` rolling average histogram for mode determination,
            `mode` mode count / count rate value from smoothed histogram,
            `median` median count / count rate value,
            `mean` mean count / count rate value,
            `perc_25` first quartile count / count rate value,
            `perc_75` third quartile count / count rate value,
            `std` standard deviation count / count rate value,
            `std_mean` standard error of the mean count / count rate value
    """
    # Load data
    scaler_raw_data = ScalerRawData(files=args.input_path, trim_empty=True, verbose=args.verbose)
    t_int = scaler_raw_data.t_int
    data = scaler_raw_data.get_data()
    # Perform analysis
    if args.partition_tw is not None:
        if args.partition_ts is not None:
            raise ValueError('partition_ts and partition_tw cannot be used at the same time.')
        else:
            partition_t = args.partition_tw
    elif args.partition_ts is not None:
        partition_t = args.partition_ts
    else:
        partition_t = None

    scaler = Scaler(data, t_int=t_int, partition_t=partition_t)
    partition_t = scaler.partition_t
    values_dict = scaler.get_values(channel=args.channel, verbose=args.verbose)
    # Plot 2d histogram of time-dependent counts / count rates.
    plotting_scaler = PlottingScaler(data=data, t_int=t_int, save_plots=False, show_plots=True,
                                     partition_t=partition_t, partition_v=args.partition_v,
                                     partition_v_unit=args.partition_v_unit, give_rate=args.give_rate)
    plotting_scaler.plot_rate_evolution_hist2d(channel=args.channel, t_step_s=args.t_step_s,
                                               values_dict=values_dict, m_value=args.m_value)
    # Plot count (rate) distributions in partitions.
    plotting_scaler.plot_dists_in_partitions(values_dict)

    return values_dict


# MAIN
if __name__ == '__main__':
    estimates = compute()
    if args.verbose:
        print('Done.')
