#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import fractions
import logging
import os
import pathlib
import traceback
import typing

import cupy as cp
import cupyx
import cupyx.scipy.signal as cpss
import digital_rf as drf
import holoscan
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("agg")

DRF_RECORDING_DIR = os.getenv("DRF_RECORDING_DIR", "/data/ringbuffer")


@dataclasses.dataclass
class SpectrogramParams:
    """Spectrogram parameters"""

    window: str = "hann"
    """Window function to apply before taking FFT"""
    nperseg: int = 1024
    """Length of each segment of samples on which to calculate a spectrum"""
    noverlap: typing.Optional[int] = None
    """Number of samples to overlap between segments. If None, `noverlap = nperseg // 2`"""
    nfft: typing.Optional[int] = None
    """Length of FFT used per segment. If None, `nfft = nperseg`"""
    detrend: typing.Union[
        typing.Literal["linear"], typing.Literal["constant"], typing.Literal[False]
    ] = False
    """Specifies how to detrend each segment. ["constant", "linear", or False]"""
    reduce_op: typing.Union[
        typing.Literal["max"], typing.Literal["median"], typing.Literal["mean"]
    ] = "max"
    """Operation to use to reduce segment spectra to one result per chunk. ["max", "median", or "mean"]"""
    num_spectra_per_chunk: int = 1
    """Number of spectra samples to calculate per chunk of data. Must evenly divide `chunk_size`."""
    num_chunks_per_output: int = 300
    """Number of chunks to combine in a single output, either a data sample or plot"""
    figsize: tuple[float, float] = (6.4, 4.8)
    """Figure size in inches given as a tuple of (width, height)"""
    dpi: int = 150
    """Figure dots per inch"""
    col_wrap: int = 1
    """Number of columns of spectrograms to use in the figure, wrapping to new rows"""
    cmap: str = "viridis"
    """Colormap"""
    snr_db_min: float = -10
    """Spectrogram color scale minimum, given as SNR in decibels"""
    snr_db_max: float = 40
    """Spectrogram color scale maximum, given as SNR in decibels"""
    plot_outdir: os.PathLike = f"{DRF_RECORDING_DIR}/spectrograms"
    """Directory for writing spectrogram plots"""


class Spectrogram(holoscan.core.Operator):
    chunk_size: int
    num_subchannels: int
    data_outdir = os.PathLike
    window: str
    nperseg: int
    noverlap: typing.Optional[int]
    nfft: typing.Optional[int]
    detrend: typing.Union[
        typing.Literal["linear"], typing.Literal["constant"], typing.Literal[False]
    ]
    reduce_op: typing.Union[
        typing.Literal["max"], typing.Literal["median"], typing.Literal["mean"]
    ]
    num_spectra_per_chunk: int
    num_chunks_per_output: int
    figsize: tuple[float, float]
    dpi: int
    col_wrap: int
    cmap: str
    snr_db_min: float
    snr_db_max: float
    plot_outdir: os.PathLike

    def __init__(
        self,
        fragment,
        *args,
        chunk_size,
        num_subchannels,
        data_outdir,
        window="hann",
        nperseg=1024,
        noverlap=None,
        nfft=None,
        detrend=False,
        reduce_op="max",
        num_spectra_per_chunk=1,
        num_chunks_per_output=300,
        figsize=(6.4, 4.8),
        dpi=150,
        col_wrap=1,
        cmap="viridis",
        snr_db_min=-10,
        snr_db_max=40,
        plot_outdir=f"{DRF_RECORDING_DIR}/spectrograms",
        **kwargs,
    ):
        """Operator that computes spectrograms from RF data.

        **==Named Inputs==**

            rf_in : RFArray
                RFArray, including metadata.

        Parameters
        ----------
        fragment : Fragment
            The fragment that the operator belongs to
        chunk_size: int
            Number of samples in an RFArray chunk of data
        num_subchannels: int
            Number of subchannels contained in the RFArray data
        data_outdir: os.PathLike
            Directory for writing spectrogram data
        window: str
            Window function to apply before taking FFT
        nperseg: int
            Length of each segment of samples on which to calculate a spectrum
        noverlap: int or None
            Number of samples to overlap between segments. If None, `noverlap = nperseg // 2`
        nfft: int or None
            Length of FFT used per segment. If None, `nfft = nperseg`
        detrend: "constant", "linear", or False
            Specifies how to detrend each segment.
        reduce_op: "max", "median", or "mean"
            Operation to use to reduce multiple segment spectra to one.
        num_spectra_per_chunk: int
            Number of spectra samples to calculate per chunk of data.
            Must evenly divide `chunk_size`.
        num_chunks_per_output: int
            Number of chunks to combine in a single output, either a data sample or plot
        figsize: tuple[float, float]
            Figure size in inches given as a tuple of (width, height)
        dpi: int
            Figure dots per inch
        col_wrap: int
            Number of columns of spectrograms to use in the figure, wrapping to new rows
        cmap: str
            Colormap
        snr_db_min: float
            Spectrogram color scale minimum, given as SNR in decibels
        snr_db_max: float
            Spectrogram color scale maximum, given as SNR in decibels
        plot_outdir: os.PathLike
            Directory for writing spectrogram plots
        """
        self.chunk_size = chunk_size
        self.num_subchannels = num_subchannels
        self.data_outdir = pathlib.Path(data_outdir).resolve()
        self.window = window
        self.nperseg = nperseg
        if noverlap is None:
            noverlap = nperseg // 2
        self.noverlap = noverlap
        if nfft is None:
            nfft = nperseg
        self.nfft = nfft
        self.detrend = detrend
        if reduce_op == "max":
            self.reduce_op = cp.amax
        elif reduce_op == "median":
            self.reduce_op = cp.median
        else:
            self.reduce_op = cp.mean
        if (self.chunk_size % num_spectra_per_chunk) != 0:
            msg = (
                f"Number of spectra per chunk ({num_spectra_per_chunk}) must evenly"
                f" divide the chunk size ({chunk_size})."
            )
            raise ValueError(msg)
        self.num_spectra_per_chunk = num_spectra_per_chunk
        self.num_chunks_per_output = num_chunks_per_output
        self.num_spectra_per_output = (
            self.num_spectra_per_chunk * self.num_chunks_per_output
        )
        self.figsize = figsize
        self.dpi = dpi
        self.col_wrap = col_wrap
        self.cmap = cmap
        self.snr_db_min = snr_db_min
        self.snr_db_max = snr_db_max
        self.plot_outdir = pathlib.Path(plot_outdir).resolve()

        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("holoscan.sdr_mep_recorder.spectrogram")

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("rf_in")

    def create_spec_figure(self):
        ncols = min(self.col_wrap, self.num_subchannels)
        nrows = int(np.ceil(self.num_subchannels / self.col_wrap))
        fig, axs = plt.subplots(
            nrows,
            ncols,
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="compressed",
            figsize=self.figsize,
            dpi=self.dpi,
        )
        fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72)
        self.norm = mpl.colors.Normalize(vmin=self.snr_db_min, vmax=self.snr_db_max)
        xlocator = mpl.dates.AutoDateLocator(minticks=3, maxticks=7)
        xformatter = mpl.dates.ConciseDateFormatter(xlocator)
        axs_1d = []
        imgs = []
        ref_lvl_texts = []
        for sch in range(self.num_subchannels):
            row_idx = sch // self.col_wrap
            col_idx = sch % self.col_wrap
            ax = axs[row_idx, col_idx]
            img = ax.imshow(
                self.spec_host_data[:, sch, :],
                cmap=self.cmap,
                norm=self.norm,
                aspect="auto",
                interpolation="none",
                origin="lower",
            )
            cb = fig.colorbar(img, ax=ax, fraction=0.05, pad=0.01)
            cb.set_label("Power relative to reference [dB]")
            ax.set_ylabel("Frequency [MHz]")
            if self.num_subchannels > 1:
                title = ax.set_title(f"Subchannel {sch}", fontsize="small")
            else:
                title = ax.set_title(" ", fontsize="small")
            ref_lvl_text = ax.text(
                1.0,
                title.get_position()[1],
                "Ref: 1.23e-9 [$V_{ADC}^2$]",
                fontsize="small",
                fontstyle="italic",
                va=title.get_verticalalignment(),
                ha="right",
                transform=title.get_transform(),
            )
            ax.xaxis.set_major_locator(xlocator)
            ax.xaxis.set_major_formatter(xformatter)
            imgs.append(img)
            axs_1d.append(ax)
            ref_lvl_texts.append(ref_lvl_text)
        axs_1d[-1].set_xlabel("Time (UTC)")
        self.suptitle = fig.suptitle("Spectrogram", fontsize="medium")
        fig.autofmt_xdate(rotation=0, ha="center")

        self.fig = fig
        self.axs = axs_1d
        self.imgs = imgs
        self.ref_lvl_texts = ref_lvl_texts

    def initialize(self):
        self.logger.debug("Initializing spectrogram operator")
        self.data_outdir.mkdir(parents=True, exist_ok=True)
        self.spec_host_data = cupyx.zeros_pinned(
            (self.nfft, self.num_subchannels, self.num_spectra_per_output),
            dtype=np.float32,
            order="F",
        )
        self.fill_data = np.full(
            (self.nfft, self.num_subchannels, self.num_spectra_per_output),
            np.nan,
            dtype=np.float32,
            order="F",
        )
        self.spec_host_data[...] = self.fill_data
        self.last_written_sample_idx = -1
        self.last_seen_sample_idx = -1
        self.create_spec_figure()
        self.prior_metadata = None
        self.freq_idx = None
        self.dmd_writer = None
        self.chunk_rate_frac = None

        # warm up CUDA calculation and extract an FFT plan
        self.calc_spectrogram_chunk(
            cp.ones((self.chunk_size, self.num_subchannels), dtype="complex64"),
            0,
        )
        plan_cache = cp.fft.config.get_plan_cache()
        for key, node in plan_cache:
            self.cufft_plan = node.plan

    def set_metadata(self, rf_metadata):
        self.prior_metadata = rf_metadata
        self.freq_idx = np.fft.fftshift(
            np.fft.fftfreq(
                self.nfft,
                rf_metadata.sample_rate_denominator / rf_metadata.sample_rate_numerator,
            )
        )
        self.spectra_rate_frac = fractions.Fraction(
            self.prior_metadata.sample_rate_numerator * self.num_spectra_per_chunk,
            self.prior_metadata.sample_rate_denominator * self.chunk_size,
        )

    def get_chunk_idx(self, sample_idx):
        # prior_metadata.sample_idx marks the start of an output cycle
        return (
            (sample_idx - self.prior_metadata.sample_idx) // self.chunk_size
        ) % self.num_chunks_per_output

    def write_output(self):
        sample_idx = self.last_seen_sample_idx
        if sample_idx <= self.last_written_sample_idx:
            # skip writing because we already wrote this data
            return
        if self.last_written_sample_idx != -1 and (
            sample_idx - self.last_written_sample_idx
        ) > (self.num_chunks_per_output * self.chunk_size):
            # shouldn't be here, trying to write data that spans more than one output batch
            msg = (
                f"Call to write_output() with {sample_idx=} when "
                f"last_written_sample_idx={self.last_written_sample_idx} is more than "
                f"size of output batch ({self.num_chunks_per_output * self.chunk_size})"
            )
            self.logger.warning(msg)
        chunk_idx = self.get_chunk_idx(sample_idx)

        spec_sample_idx = sample_idx - chunk_idx * self.chunk_size
        sr_frac = fractions.Fraction(
            self.prior_metadata.sample_rate_numerator,
            self.prior_metadata.sample_rate_denominator,
        )
        spec_start_dt = drf.util.sample_to_datetime(
            spec_sample_idx,
            sr_frac,
        )
        spectra_arange = np.arange(0, (chunk_idx + 1) * self.num_spectra_per_chunk)
        sample_idx_arr = (
            spec_sample_idx
            + self.chunk_size // self.num_spectra_per_chunk * spectra_arange
        )
        time_idx = np.datetime64(spec_start_dt.replace(tzinfo=None)) + (
            np.timedelta64(int(1000000000 / self.spectra_rate_frac), "ns")
            * spectra_arange
        )
        output_spec_data = self.spec_host_data[
            ...,
            0 : (chunk_idx + 1) * self.num_spectra_per_chunk,
        ]

        self.logger.info(f"Outputting spectrogram for time {spec_start_dt}")

        num_retries = 3
        for retry in range(0, num_retries):
            try:
                self.dmd_writer.write(
                    [spec_sample_idx],
                    [
                        {
                            "spectrogram": output_spec_data.transpose((1, 0, 2)),
                            "freq_idx": self.freq_idx + self.prior_metadata.center_freq,
                            "sample_idx": sample_idx_arr,
                            "center_freq": self.prior_metadata.center_freq,
                        }
                    ],
                )
            except IOError:
                if retry == (num_retries - 1):
                    self.logger.warning(traceback.format_exc())
            else:
                break

        timestr = spec_start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        freqstr = f"{self.prior_metadata.center_freq / 1e6:n}MHz"
        datestr = spec_start_dt.strftime("%Y-%m-%d")

        reference_pwr = np.nanpercentile(
            output_spec_data, 15, axis=(0, 2), keepdims=True
        )
        spec_power_db = 10 * np.log10(output_spec_data / reference_pwr)
        delta_t = time_idx[1] - time_idx[0]
        delta_f = self.freq_idx[1] - self.freq_idx[0]
        extent = (
            time_idx[0],
            time_idx[-1] + delta_t,
            (self.prior_metadata.center_freq + self.freq_idx[0] - delta_f / 2) / 1e6,
            (self.prior_metadata.center_freq + self.freq_idx[-1] + delta_f / 2) / 1e6,
        )
        for sch in range(self.num_subchannels):
            self.imgs[sch].set(
                data=spec_power_db[:, sch, :],
                extent=extent,
            )
            self.ref_lvl_texts[sch].set_text(
                f"Ref: {float(reference_pwr[0, sch, 0]):.3n} [$V_{{ADC}}^2$]"
            )
        self.suptitle.set_text(
            f"{self.data_outdir.parent.name}/{self.data_outdir.name} @ {freqstr}"
        )
        self.fig.canvas.draw()

        fname = f"spec_{timestr}_{freqstr}.png"
        outpath = self.plot_outdir / freqstr / datestr / fname
        outpath.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(outpath)
        latest_spec_path = outpath.parent.parent / "spec_latest.png"
        latest_spec_path.unlink(missing_ok=True)
        os.link(outpath, latest_spec_path)

        # reset spectrogram data for next plot
        self.spec_host_data[...] = self.fill_data
        self.last_written_sample_idx = sample_idx

    def compute(
        self,
        op_input: holoscan.core.InputContext,
        op_output: holoscan.core.OutputContext,
        context: holoscan.core.ExecutionContext,
    ):
        rf_arr = op_input.receive("rf_in")
        stream_ptr = op_input.receive_cuda_stream("rf_in", allocate=True)
        rf_metadata = rf_arr.metadata

        if (rf_metadata.sample_idx - self.last_seen_sample_idx) > (
            self.num_chunks_per_output * self.chunk_size
        ):
            # triggers on first compute call, but write_output returns immediately
            # new data is not in same output batch as unwritten, so write that first
            self.write_output()

        if self.prior_metadata is None:
            self.set_metadata(rf_metadata)
            self.dmd_writer = drf.DigitalMetadataWriter(
                metadata_dir=str(self.data_outdir),
                subdir_cadence_secs=3600,
                file_cadence_secs=1,
                sample_rate_numerator=self.prior_metadata.sample_rate_numerator,
                sample_rate_denominator=self.prior_metadata.sample_rate_denominator,
                file_name="spectrogram",
            )
        if (
            (
                self.prior_metadata.sample_rate_numerator
                != rf_metadata.sample_rate_numerator
            )
            or (
                self.prior_metadata.sample_rate_denominator
                != rf_metadata.sample_rate_denominator
            )
            or (self.prior_metadata.center_freq != rf_metadata.center_freq)
        ):
            # metadata changed, write out existing data and start anew
            self.write_output()
            self.set_metadata(rf_metadata)

        self.last_seen_sample_idx = rf_metadata.sample_idx
        chunk_idx = self.get_chunk_idx(self.last_seen_sample_idx)

        msg = (
            f"Processing spectrogram for chunk with sample_idx {rf_metadata.sample_idx}"
            f" into chunk_idx={chunk_idx}"
        )
        self.logger.debug(msg)

        with cp.cuda.ExternalStream(stream_ptr) as stream:
            rf_data = cp.from_dlpack(rf_arr.data)
            with self.cufft_plan:
                self.calc_spectrogram_chunk(rf_data, chunk_idx)
            stream.synchronize()
        # chunk_spec = self.spec_host_data[
        # ...,
        # chunk_idx * self.num_spectra_per_chunk : (chunk_idx + 1)
        # * self.num_spectra_per_chunk,
        # ]
        if chunk_idx == (self.num_chunks_per_output - 1):
            self.write_output()

    def calc_spectrogram_chunk(self, rf_data, chunk_idx):
        for chunk_spectrum_idx, spectrum_chunk in enumerate(
            cp.split(rf_data, self.num_spectra_per_chunk, axis=0)
        ):
            _freqs, sidxs, Zxx = cpss.stft(
                spectrum_chunk,
                fs=1,
                window=self.window,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                detrend=self.detrend,
                return_onesided=False,
                boundary=None,
                padded=True,
                axis=0,
                scaling="spectrum",
            )
            # reduce over time axis
            spec = cp.fft.fftshift(
                self.reduce_op(Zxx.real**2 + Zxx.imag**2, axis=-1), axes=0
            )

            cp.asnumpy(
                spec,
                out=self.spec_host_data[
                    ..., chunk_idx * self.num_spectra_per_chunk + chunk_spectrum_idx
                ],
                blocking=False,
            )

    def stop(self):
        msg = (
            "Stopping spectrogram operator with "
            f"last_seen_sample_idx={self.last_seen_sample_idx}."
        )
        self.logger.info(msg)
        self.write_output()
