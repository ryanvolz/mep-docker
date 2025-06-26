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
import logging
import os
import pathlib
import signal
import sys
import tempfile
import typing

import cupy as cp
import cupyx
import cupyx.scipy.signal as cpss
import digital_rf as drf
import holoscan
import jsonargparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from holohub import basic_network, rf_array
from holohub.rf_array.digital_metadata import DigitalMetadataSink
from holohub.rf_array.params import (
    DigitalRFSinkParams,
    NetConnectorBasicParams,
    ResamplePolyParams,
    RotatorScheduledParams,
    SubchannelSelectParams,
    add_chunk_kwargs,
)
from jsonargparse.typing import NonNegativeInt, PositiveInt

mpl.use("agg")

logger = logging.getLogger("sdr_mep_recorder.py")

jsonargparse.set_parsing_settings(docstring_parse_attribute_docstrings=True)

DRF_RECORDING_DIR = os.getenv("DRF_RECORDING_DIR", "/data/ringbuffer")


@dataclasses.dataclass
class SchedulerParams:
    """Event-based scheduler parameters"""

    worker_thread_number: PositiveInt = 8
    """Number of worker threads"""
    stop_on_deadlock: bool = True
    """Whether the application will terminate if a deadlock occurs"""
    stop_on_deadlock_timeout: int = 500
    """Time (in ms) to wait before determining that a deadlock has occurred"""


@dataclasses.dataclass
class PipelineParams:
    """Pipeline configuration parameters"""

    selector: bool = False
    "Enable / disable subchannel selector"
    converter: bool = True
    "Enable / disable complex int to float converter"
    rotator: bool = False
    "Enable / disable frequency rotator"
    resampler0: bool = True
    "Enable / disable the first stage resampler"
    resampler1: bool = False
    "Enable / disable the second stage resampler"
    resampler2: bool = True
    "Enable / disable the third stage resampler"


@dataclasses.dataclass
class BasicNetworkOperatorParams:
    """Basic network operator parameters"""

    ip_addr: str = "0.0.0.0"
    """IP address to bind to"""
    dst_port: NonNegativeInt = 60133
    "UDP or TCP port to listen on"
    l4_proto: str = "udp"
    "Layer 4 protocol (udp or tcp)"
    batch_size: PositiveInt = 6250
    "Number of packets in batch"
    max_payload_size: PositiveInt = 8256
    "Maximum payload size expected from sender"


@dataclasses.dataclass
class SpectrogramParams:
    """Spectrogram parameters"""

    cmap: str = "viridis"
    """Colormap"""


@dataclasses.dataclass
class AdvancedNetworkOperatorParams:
    """Advanced network operator parameters"""

    cfg: typing.Optional[dict] = None


def build_config_parser():
    parser = jsonargparse.ArgumentParser(
        prog="sdr_mep_recorder",
        description="Process and record RF data for the SpectrumX Mobile Experiment Platform (MEP)",
        default_env=True,
    )
    parser.add_argument("--config", action="config")
    parser.add_argument("--scheduler", type=SchedulerParams)
    parser.add_argument("--pipeline", type=PipelineParams)
    parser.add_argument("--basic_network", type=BasicNetworkOperatorParams)
    parser.add_argument("--advanced_network", type=AdvancedNetworkOperatorParams)
    parser.add_argument(
        "--packet",
        type=NetConnectorBasicParams,
        default=NetConnectorBasicParams(spoof_header=True),
    )
    parser.add_argument("--selector", type=SubchannelSelectParams)
    parser.add_argument("--rotator", type=RotatorScheduledParams)
    parser.add_argument(
        "--resampler0",
        type=ResamplePolyParams,
        default=ResamplePolyParams(
            up=1,
            down=8,
            outrate_cutoff=1.0,
            # transition_width: 2 * (cutoff - 1 / remaining_dec)
            #                   2 * (1.0 - 1 / 8) = 1.75
            outrate_transition_width=1.75,
            attenuation_db=105,
        ),
    )
    parser.add_argument(
        "--resampler1",
        type=ResamplePolyParams,
        default=ResamplePolyParams(
            up=5,
            down=16,
            outrate_cutoff=1.0,
            outrate_transition_width=0.2,
            attenuation_db=99.65,
        ),
    )
    parser.add_argument(
        "--resampler2",
        type=ResamplePolyParams,
        default=ResamplePolyParams(
            up=1,
            down=8,
            outrate_cutoff=1.0,
            outrate_transition_width=0.2,
            attenuation_db=99.475,
        ),
    )
    parser.add_argument("--drf_sink", type=DigitalRFSinkParams)
    parser.add_argument(
        "--metadata", type=typing.Optional[dict[str, typing.Any]], default=None
    )
    parser.add_argument("--spectrogram", type=SpectrogramParams)

    return parser


class Spectrogram(holoscan.core.Operator):
    window: str
    reduce_op: str
    chunk_size: int
    num_subchannels: int

    def __init__(
        self,
        fragment,
        chunk_size,
        num_subchannels,
        *args,
        window="hann",
        nperseg=1024,
        noverlap=None,
        nfft=None,
        detrend=False,
        reduce_op="max",
        num_chunks_per_plot=300,
        figsize=(6.4, 4.8),
        dpi=150,
        col_wrap=3,
        cmap="viridis",
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
            The fragment that the operator belongs to.
        """
        self.chunk_size = chunk_size
        self.num_subchannels = num_subchannels
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
        self.num_chunks_per_plot = num_chunks_per_plot
        self.figsize = figsize
        self.dpi = dpi
        self.col_wrap = col_wrap
        self.cmap = cmap
        self.plot_outdir = pathlib.Path(plot_outdir).resolve()

        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("Spectrogram")

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("rf_in").connector(
            holoscan.core.IOSpec.ConnectorType.DOUBLE_BUFFER,
            capacity=100,
        )

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
        self.spec_host_data = cupyx.zeros_pinned(
            (self.nfft, self.num_subchannels, self.num_chunks_per_plot),
            dtype=np.float32,
            order="F",
        )
        self.fill_data = np.full(
            (self.nfft, self.num_subchannels, self.num_chunks_per_plot),
            np.nan,
            dtype=np.float32,
            order="F",
        )
        self.spec_host_data[...] = self.fill_data
        self.norm = mpl.colors.Normalize(vmin=None, vmax=None)
        axs_1d = []
        imgs = []
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
            imgs.append(img)
            axs_1d.append(ax)

        self.fig = fig
        self.axs = axs_1d
        self.imgs = img

    def initialize(self):
        self.logger.debug("Initializing spectrogram operator")
        self.create_spec_figure()
        self.prior_metadata = None
        self.freq_idx = None

    def compute(
        self,
        op_input: holoscan.core.InputContext,
        op_output: holoscan.core.OutputContext,
        context: holoscan.core.ExecutionContext,
    ):
        rf_array = op_input.receive("rf_in")
        rf_data = cp.from_dlpack(rf_array.data)
        rf_metadata = rf_array.metadata

        if self.prior_metadata is None:
            self.prior_metadata = rf_metadata
            self.freq_idx = np.fft.fftshift(
                np.fft.fftfreq(
                    self.nfft,
                    rf_metadata.sample_rate_denominator
                    / rf_metadata.sample_rate_numerator,
                )
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
            self.logger.info("rf_metadata does not match prior")

        chunk_plot_idx = (
            rf_metadata.sample_idx // self.chunk_size
        ) % self.num_chunks_per_plot

        msg = (
            f"Processing spectrogram for chunk with sample_idx {rf_metadata.sample_idx}"
            f" into chunk_plot_idx={chunk_plot_idx}"
        )
        self.logger.debug(msg)

        with cp.cuda.ExternalStream(rf_array.stream):
            freqs, ts, Zxx = cpss.stft(
                rf_data,
                fs=rf_metadata.sample_rate_numerator
                / rf_metadata.sample_rate_denominator,
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
                spec, out=self.spec_host_data[..., chunk_plot_idx], blocking=False
            )

        if chunk_plot_idx == (self.num_chunks_per_plot - 1):
            msg = (
                "Saving spectrogram figure after chunk with sample_idx"
                f" {rf_metadata.sample_idx}"
            )
            self.logger.debug(msg)

            plot_start_dt = drf.util.sample_to_datetime(
                rf_metadata.sample_idx
                - self.chunk_size * (self.num_chunks_per_plot - 1),
                np.longdouble(rf_metadata.sample_rate_numerator)
                / rf_metadata.sample_rate_denominator,
            )

            spec_power_db = 10 * np.log10(self.spec_host_data)
            # update self.norm.vmin, self.norm.vmax?
            for sch in range(self.num_subchannels):
                self.imgs[sch].set(
                    data=spec_power_db[:, sch, :],
                    extent=(0, 1, -1, 1),
                )
            self.fig.canvas.draw()

            fname = f"spec_{plot_start_dt.isoformat()}.png"
            subdir = plot_start_dt.strftime("%Y-%m-%d")
            outpath = self.plot_outdir / subdir / fname
            outpath.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(outpath)
            latest_spec_path = outpath.parent / "spec_latest.png"
            latest_spec_path.unlink(missing_ok=True)
            os.link(outpath, latest_spec_path)

            # reset spectrogram data for next plot
            self.spec_host_data[...] = self.fill_data


class App(holoscan.core.Application):
    def compose(self):
        basic_net_rx = basic_network.BasicNetworkOpRx(
            self, name="basic_network_rx", **self.kwargs("basic_network")
        )

        net_connector_rx = rf_array.NetConnectorBasic(
            self, name="net_connector_rx", **self.kwargs("packet")
        )
        self.add_flow(basic_net_rx, net_connector_rx, {("burst_out", "burst_in")})

        last_chunk_shape = (
            self.kwargs("packet")["num_samples"],
            self.kwargs("packet")["num_subchannels"],
        )
        last_op = net_connector_rx

        if self.kwargs("pipeline")["selector"]:
            selector = rf_array.SubchannelSelect_sc16(
                self, name="selector", **self.kwargs("selector")
            )
            self.add_flow(last_op, selector)
            last_op = selector
            last_chunk_shape = (
                last_chunk_shape[0],
                len(self.kwargs("selector")["subchannel_idx"]),
            )

        if self.kwargs("pipeline")["converter"]:
            converter = rf_array.TypeConversionComplexIntToFloat(
                self,
                name="converter",
            )
            self.add_flow(last_op, converter)
            last_op = converter

            if self.kwargs("pipeline")["rotator"]:
                rotator = rf_array.RotatorScheduled(
                    self, name="rotator", **self.kwargs("rotator")
                )
                self.add_flow(last_op, rotator)
                last_op = rotator

            if self.kwargs("pipeline")["resampler0"]:
                resample_kwargs = add_chunk_kwargs(
                    last_chunk_shape, **self.kwargs("resampler0")
                )
                resampler0 = rf_array.ResamplePoly(
                    self, name="resampler0", **resample_kwargs
                )
                self.add_flow(last_op, resampler0)
                last_op = resampler0
                last_chunk_shape = (
                    last_chunk_shape[0]
                    * resample_kwargs["up"]
                    // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resampler1"]:
                resample_kwargs = add_chunk_kwargs(
                    last_chunk_shape, **self.kwargs("resampler1")
                )
                resampler1 = rf_array.ResamplePoly(
                    self, name="resampler1", **resample_kwargs
                )
                self.add_flow(last_op, resampler1)
                last_op = resampler1
                last_chunk_shape = (
                    last_chunk_shape[0]
                    * resample_kwargs["up"]
                    // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            if self.kwargs("pipeline")["resampler2"]:
                resample_kwargs = add_chunk_kwargs(
                    last_chunk_shape, **self.kwargs("resampler2")
                )
                resampler2 = rf_array.ResamplePoly(
                    self, name="resampler2", **resample_kwargs
                )
                self.add_flow(last_op, resampler2)
                last_op = resampler2
                last_chunk_shape = (
                    last_chunk_shape[0]
                    * resample_kwargs["up"]
                    // resample_kwargs["down"],
                    last_chunk_shape[1],
                )

            drf_sink = rf_array.DigitalRFSink_fc32(
                self,
                name="drf_sink",
                **add_chunk_kwargs(last_chunk_shape, **self.kwargs("drf_sink")),
            )
            self.add_flow(last_op, drf_sink)

        else:
            drf_sink = rf_array.DigitalRFSink_sc16(
                self,
                name="drf_sink",
                **add_chunk_kwargs(last_chunk_shape, **self.kwargs("drf_sink")),
            )
            self.add_flow(last_op, drf_sink)

        dmd_sink = DigitalMetadataSink(
            self,
            name="dmd_sink",
            metadata_dir=f"{self.kwargs('drf_sink')['channel_dir']}/metadata",
            subdir_cadence_secs=self.kwargs("drf_sink")["subdir_cadence_secs"],
            file_cadence_secs=self.kwargs("drf_sink")["file_cadence_millisecs"] // 1000,
            uuid=self.kwargs("drf_sink")["uuid"],
            filename_prefix="metadata",
            metadata=self.kwargs("metadata"),
        )
        self.add_flow(last_op, dmd_sink)

        spectrogram = Spectrogram(
            self,
            name="spectrogram",
            **add_chunk_kwargs(last_chunk_shape, **self.kwargs("spectrogram")),
        )
        self.add_flow(last_op, spectrogram)


def main():
    parser = build_config_parser()
    cfg = parser.parse_args()

    env_log_level = os.environ.get("HOLOSCAN_LOG_LEVEL", "WARN").upper()
    log_level_map = {
        "OFF": "NOTSET",
        "CRITICAL": "CRITICAL",
        "ERROR": "ERROR",
        "WARN": "WARNING",
        "INFO": "INFO",
        "DEBUG": "DEBUG",
        "TRACE": "DEBUG",
    }
    log_level = log_level_map[env_log_level]
    logging.basicConfig(level=log_level, force=True)

    # We have a parsed configuration (using jsonargparse), but the holoscan app wants
    # to read all of its configuration parameters from a YAML file, so we write out
    # the configuration to a file in the temporary directory and feed it that
    config_path = pathlib.Path(tempfile.gettempdir()) / "sdr_mep_recorder_config.yaml"
    logger.debug(f"Writing temporary config file to {config_path}")
    parser.save(cfg, config_path, format="yaml", overwrite=True)

    app = App([sys.executable, sys.argv[0]])
    app.config(str(config_path))

    scheduler = holoscan.schedulers.EventBasedScheduler(
        app,
        name="event-based-scheduler",
        **app.kwargs("scheduler"),
    )
    app.scheduler(scheduler)

    def sigterm_handler(signal, frame):
        logger.info("Received SIGTERM, cleaning up")
        sys.stdout.flush()
        sys.exit(128 + signal)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        app.run()
    except KeyboardInterrupt:
        # catch keyboard interrupt and simply exit
        pass
    finally:
        logger.info("Done")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
