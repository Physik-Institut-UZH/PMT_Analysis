import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Optional


class PlottingScaler:
    """Class for plotting of CAEN V260 scaler data and related dark count results.

    Attributes:
        data: Pandas data frame with scaler data as returned by
            `pmt_analysis.utils.input.ScalerRawData.get_data` method.
        t_int: Data acquisition interval in seconds. Attribute of
            `pmt_analysis.utils.input.ScalerRawData` object.
        save_plots: Bool defining if plots are saved (as png and pdf).
        show_plots: Bool defining if plots are displayed.
        save_dir: Target directory for saving the plots.
    """

    def __init__(self, data: pd.DataFrame, t_int: int, save_plots: bool = False, show_plots: bool = True,
                 save_dir: Optional[str] = None):
        """

        Args:
            data: Pandas data frame with scaler data as returned by
                `pmt_analysis.utils.input.ScalerRawData.get_data` method.
            t_int: Data acquisition interval in seconds. Attribute of
                `pmt_analysis.utils.input.ScalerRawData` object.
            save_plots: Bool defining if plots are saved (as png and pdf).
            show_plots: Bool defining if plots are displayed.
            save_dir: Target directory for saving the plots.
        """
        self.data = data
        self.t_int = t_int
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.save_dir = save_dir
        if save_plots & (save_dir is None):
            raise NameError('save_dir must be defined if save_plots is True.')

    def plot_rate_evolution_hist2d(self, channel: int, t_step_s: int, time_format: str = 't_datetime_utc',
                                   give_rate: bool = False):
        """Plot 2d histogram of time-dependent counts / count rates.

        Args:
            channel: Scaler channel number.
            t_step_s: Time bin width in seconds.
            time_format: Time axis format: Options:
                `t_s_rel` time difference in seconds since start of data period;
                `t_h_rel` time difference in hours since start of data period;
                `t_d_rel` time difference in days since start of data period;
                `t_datetime_utc` UTC datetime;
                `t_datetime_zh` datetime for Zurich time zone, i.e. CE(S)T;
            give_rate: If true, use count rates on vertical axis, otherwise use absolute count values.
        """
        # Convert time stamps to selected format
        if time_format == 't_s_rel':
            t_values = self.data['timestamp'] - self.data['timestamp'].min()
            t_step = t_step_s
        elif time_format == 't_h_rel':
            t_values = (self.data['timestamp'] - self.data['timestamp'].min()) / 3600
            t_step = t_step_s / 3600
        elif time_format == 't_d_rel':
            t_values = (self.data['timestamp'] - self.data['timestamp'].min()) / (24 * 3600)
            t_step = t_step_s / (24 * 3600)
        elif time_format == 't_datetime_utc':
            t_values = pd.to_datetime(self.data['timestamp'], unit='s', utc=True)
            t_values = t_values.map(lambda x: x.tz_localize(None))  # remove timezone holding local time representations
            t_step = pd.Timedelta(t_step_s, 's')
        elif time_format == 't_datetime_zh':
            t_values = pd.to_datetime(self.data['timestamp'], unit='s', utc=True)
            t_values = t_values.map(lambda x: x.tz_convert('Europe/Zurich'))
            t_values = t_values.map(lambda x: x.tz_localize(None))  # remove timezone holding local time representations
            t_step = pd.Timedelta(t_step_s, 's')
        else:
            raise ValueError('Value {} for time_format unsupported.'
                             'Supported values: [t_s_rel, t_h_rel, t_d_rel, t_datetime_utc, '
                             't_datetime_zh]'.format(time_format))

        # Define time axis bins
        bins_x = np.arange(t_values.min(),
                           t_values.max() + t_step,
                           t_step)
        if time_format in ['t_datetime_utc', 't_datetime_zh']:
            bins_x = pd.to_datetime(bins_x)

        # Define count / count rate axis bins
        if give_rate:
            bins_y_05p = np.percentile(self.data['ch{}_freq'.format(channel)], 5)
            bins_y_95p = np.percentile(self.data['ch{}_freq'.format(channel)], 95)
        else:
            bins_y_05p = np.percentile(self.data['ch{}_cnts'.format(channel)], 5)
            bins_y_95p = np.percentile(self.data['ch{}_cnts'.format(channel)], 95)
        bins_y_min = bins_y_05p - (bins_y_95p - bins_y_05p)
        bins_y_max = bins_y_95p + (bins_y_95p - bins_y_05p)
        if give_rate:
            bins_y = np.arange(bins_y_min, bins_y_max, 1)
        else:
            bins_y = np.arange(bins_y_min, bins_y_max, self.t_int)

        # Generate plot
        fig, ax = plt.subplots()
        if give_rate:
            plt.hist2d(t_values, self.data['ch{}_freq'.format(channel)],
                       bins=[bins_x, bins_y], cmap='Blues')
        else:
            plt.hist2d(t_values, self.data['ch{}_cnts'.format(channel)],
                       bins=[bins_x, bins_y], cmap='Blues')
        if time_format == 't_s_rel':
            plt.xlabel('Time [s]')
        elif time_format == 't_h_rel':
            plt.xlabel('Time [h]')
        elif time_format == 't_d_rel':
            plt.xlabel('Time [d]')
        elif time_format == 't_datetime_utc':
            plt.xlabel('Date [UTC]')
            fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        elif time_format == 't_datetime_zh':
            plt.xlabel('Date')
            fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        if give_rate:
            plt.ylabel('Count Rate [s$^{-1}$]')
        else:
            if self.t_int == 1:
                plt.ylabel('Counts Per 1 Second')
            else:
                plt.ylabel('Counts Per {} Seconds'.format(self.t_int))
        cbar = plt.colorbar()
        cbar.set_label('Entries')
        plt.tight_layout()
        filename = 'scaler_channel_{}_'.format(channel)
        if give_rate:
            filename += 'rate_vs_'
        else:
            filename += 'counts_vs_'
        filename += time_format
        filename += '_{}-{}'.format(self.data['timestamp'].min(), self.data['timestamp'].max())
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()
