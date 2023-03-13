import os
import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


class PlottingAfterpulses:
    """
        Class for plots related to the afterpulse processing.

        Attributes:
            df: Pandas dataframe with after pulse candidates and their properties.
            adc_f: ADC sampling frequency in samples per second as provided by same attribute of
                `pmt_analysis.utils.input.ADCRawData`.
            adc_area_to_e: Conversion factor pulse area in ADC units to charge in units of elementary charge.
                Attribute of `pmt_analysis.utils.input.ADCRawData` objects.
            gain: PMT gain (in units of read out electrons per induced photoelectron).
                Obtainable with `pmt_analysis.analysis.model_independent.GainModelIndependent` class.
            save_plots: Bool defining if plots are saved (as png and pdf).
            show_plots: Bool defining if plots are displayed.
            save_dir: Target directory for saving the plots.
            save_name_suffix: Measurement specific figure name suffix.
        """

    def __init__(self, df: pd.DataFrame, adc_f: float,
                 save_plots: bool = False, show_plots: bool = True,
                 save_dir: Optional[str] = None, save_name_suffix: Optional[str] = None,
                 adc_area_to_e: Optional[float] = None, gain: Optional[float] = None):
        """Init of the PlottingAfterpulses class.

        Generation of standard plots with respect to the afterpulse processing.

        Args:
            df: Pandas dataframe with after pulse candidates and their properties.
            adc_f: ADC sampling frequency in samples per second as provided by same attribute of
                `pmt_analysis.utils.input.ADCRawData`.
            save_plots: Bool defining if plots are saved (as png and pdf).
            show_plots: Bool defining if plots are displayed.
            save_dir: Target directory for saving the plots.
            save_name_suffix: Measurement specific figure name suffix.
            adc_area_to_e: Conversion factor pulse area in ADC units to charge in units of elementary charge.
                Attribute of `pmt_analysis.utils.input.ADCRawData` objects.
            gain: PMT gain (in units of read out electrons per induced photoelectron).
                Obtainable with `pmt_analysis.analysis.model_independent.GainModelIndependent` class.
        """
        self.df = df
        self.adc_f = adc_f
        self.adc_area_to_e = adc_area_to_e
        self.gain = gain
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.save_dir = save_dir
        self.save_name_suffix = save_name_suffix
        if save_plots & (save_dir is None):
            raise NameError('save_dir must be defined if save_plots is True.')
        if save_plots & (save_name_suffix is None):
            raise NameError('save_name_suffix must be defined if save_plots is True.')

    def plot_wf(self, i: int = 0):
        """Plot i-th afterpulse candidate waveform.

        Args:
            i: Integer-location based index of waveform in `df` to be plotted.
        """
        if (i >= self.df.shape[0]) or (i < 0):
            raise IndexError('Integer-location based index i must be between 0 and {}'.format(self.df.shape[0]-1))
        separability = 'Separable' if self.df.iloc[i]['separable'] else 'Non-Separable'
        x_dummy = np.arange(len(self.df.iloc[i]['input_data_converted'])) / self.adc_f * 1e9
        plt.step(x_dummy, self.df.iloc[i]['input_data_converted'], where='mid')
        plt.axvline(x=self.df.iloc[i]['p0_position'] / self.adc_f * 1e9, c='gray', linestyle='dashed', zorder=-1,
                    label='Peak Positions\n'+r'$\Delta$t = {} ns'.format(self.df.iloc[i]['t_diff_ns']))
        plt.axvline(x=self.df.iloc[i]['p1_position'] / self.adc_f * 1e9, c='gray', linestyle='dashed', zorder=-1)
        plt.axvspan(self.df.iloc[i]['p0_lower_bound'] / self.adc_f * 1e9 - 0.5,
                    self.df.iloc[i]['p0_upper_bound'] / self.adc_f * 1e9,
                    color='C1', lw=0, alpha=0.4, zorder=-2,
                    label='Main Pulse')
        plt.axvspan(self.df.iloc[i]['p1_lower_bound'] / self.adc_f * 1e9,
                    self.df.iloc[i]['p1_upper_bound'] / self.adc_f * 1e9 + 0.5,
                    color='C3', lw=0, alpha=0.4, zorder=-2,
                    label='Afterpulse\n({})'.format(separability))
        plt.xlim(x_dummy[0] - 0.5, x_dummy[-1] + 0.5)
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [ADC Bins]')
        legend = plt.legend(loc=1, bbox_to_anchor=(0.99, 0.99))
        legend.get_frame().set_linewidth(matplotlib.rcParams['axes.linewidth'])
        filename = 'ap_candidate_wf_example_{}_{}'.format(i, self.save_name_suffix)
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_first_n_wfs(self, n: int = 3):
        """Plot first n afterpulse candidate waveforms.

        Args:
            n: Number of afterpulse candidate waveforms to be plotted.
        """
        for i in range(n):
            self.plot_wf(i)

    def plot_hist_tdiff(self):
        """Plot time differences after pulse - main pulse for both all and separable afterpulse candidates."""
        x_dummy = np.arange(0, int(self.df['t_diff_ns'].max()), step=int(1e9 / self.adc_f))
        n_all, bins_edges, _ = plt.hist(self.df['t_diff_ns'], bins=x_dummy - 0.5,
                                        histtype='step', color='C0',
                                        label='All Afterpulse Candidates')
        bins_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
        plt.fill_between(bins_centers, n_all - np.sqrt(n_all), n_all + np.sqrt(n_all),
                         color='C0', alpha=0.5, zorder=-1)
        n_sep, _, _ = plt.hist(self.df[self.df['separable']]['t_diff_ns'], bins=x_dummy - 0.5,
                               histtype='step', color='C1',
                               label='Separable Afterpulse Candidates')
        plt.fill_between(bins_centers, n_sep - np.sqrt(n_sep), n_sep + np.sqrt(n_sep),
                         color='C1', alpha=0.5, zorder=-1)
        plt.xlabel('Time Difference [ns]')
        plt.ylabel('Entries')
        plt.yscale('log')
        plt.xlim(right=x_dummy[-1])
        legend = plt.legend(loc=1, bbox_to_anchor=(0.99, 0.99))
        legend.get_frame().set_linewidth(matplotlib.rcParams['axes.linewidth'])
        filename = 'ap_tdiff_{}'.format(self.save_name_suffix)
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_ap_area_vs_tdiff(self, separable_only=True, ymax=15):
        """Plot afterpulse area vs time difference histogram.

        Args:
            separable_only: Constrain to afterpulse candidates separable from main pulse.
            ymax: Upper vertical axis plot limit.
        """
        well_defined = True
        if self.adc_area_to_e is None:
            warnings.warn('Attribute adc_area_to_e set to None, will default to value 3047.6 for ADC V1730D.')
            adc_area_to_e = 3047.6
            well_defined = False
        else:
            adc_area_to_e = self.adc_area_to_e
        if self.gain is None:
            warnings.warn('Attribute gain set to None, will default to value 3e6.')
            gain = 3e6
            well_defined = False
        else:
            gain = self.gain

        self.df['p1_area_conv'] = self.df['p1_area'] * adc_area_to_e / gain
        self.df['t_diff_us'] = self.df['t_diff_ns'] / 1e3
        binsx = np.arange(0, round(self.df['t_diff_us'].max(), 2), 0.01)
        binsy = np.arange(0, ymax, 0.1)
        if separable_only:
            plt.hist2d(self.df[self.df['separable']]['t_diff_us'], self.df[self.df['separable']]['p1_area_conv'],
                       bins=(binsx, binsy), norm=matplotlib.colors.LogNorm())
        else:
            plt.hist2d(self.df['t_diff_us'], self.df['p1_area_conv'],
                       bins=(binsx, binsy), norm=matplotlib.colors.LogNorm())
        plt.xlabel(r'Time Difference [$\mu$s]')
        if well_defined:
            plt.ylabel('Afterpulse Area [PE]')
        else:
            plt.ylabel('Afterpulse Area [A.U.]')
        plt.colorbar(label="Entries")
        if separable_only:
            filename = 'ap_area_vs_tdiff_separable_{}'.format(self.save_name_suffix)
        else:
            filename = 'ap_area_vs_tdiff_{}'.format(self.save_name_suffix)
        if self.save_plots:
            plt.savefig(os.path.join(self.save_dir, filename + '.png'))
            plt.savefig(os.path.join(self.save_dir, filename + '.pdf'))
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        self.df.drop(columns=['p1_area_conv', 't_diff_us'], inplace=True)
