#!/usr/bin/env python3
# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})
sns.set_palette(sns.color_palette("Paired"))


class ReportCreatorVerifier:
    """
    Mirror of `ReportCreator`, but grouping/hue by `verifier` instead of `network`.

    This class expects a list of DataFrames (one per verifier run) as used in
    `generate_verifier_comparison_plots`, but can also accept a single DataFrame.
    """

    def __init__(self, dfs: list[pd.DataFrame] | pd.DataFrame, custom_colors: dict[str, str] | None = None):
        if isinstance(dfs, pd.DataFrame):
            self.df = dfs
        else:
            # Concatenate multiple verifier result DataFrames
            self.df = pd.concat(dfs, ignore_index=True)
        self.custom_colors = custom_colors

    def create_hist_figure(self) -> plt.Figure:
        kwargs = {}
        if self.custom_colors is not None:
            kwargs["palette"] = self.custom_colors
        hist_plot = sns.histplot(
            data=self.df,
            x="epsilon_value",
            hue="verifier",
            multiple="stack",
            **kwargs,
        )
        hist_plot.set_xlabel("Epsilon value")
        figure = hist_plot.get_figure()
        plt.close()
        return figure

    def create_box_figure(self) -> plt.Figure:
        kwargs = {}
        if self.custom_colors is not None:
            kwargs["palette"] = self.custom_colors
        box_plot = sns.boxplot(
            data=self.df,
            x="verifier",
            y="epsilon_value",
            **kwargs,
        )
        box_plot.set_ylabel("Epsilon value")
        box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=90)
        figure = box_plot.get_figure()
        plt.close()
        return figure

    def create_kde_figure(self) -> plt.Figure:
        kwargs = {}
        if self.custom_colors is not None:
            kwargs["palette"] = self.custom_colors
        kde_plot = sns.kdeplot(
            data=self.df,
            x="epsilon_value",
            hue="verifier",
            multiple="stack",
            **kwargs,
        )
        kde_plot.set_xlabel("Epsilon value")
        figure = kde_plot.get_figure()
        plt.close()
        return figure

    def create_ecdf_figure(self) -> plt.Figure:
        kwargs = {}
        if self.custom_colors is not None:
            kwargs["palette"] = self.custom_colors
        ecdf_plot = sns.ecdfplot(
            data=self.df,
            x="epsilon_value",
            hue="verifier",
            **kwargs,
        )
        ecdf_plot.set_xlabel("Epsilon value")
        ecdf_plot.set_ylabel("Fraction epsilon values found")
        figure = ecdf_plot.get_figure()
        plt.close()
        return figure

    def create_anneplot(self) -> plt.Axes:
        """Analog of `ReportCreator.create_anneplot`, but grouped by verifier."""
        df = self.df
        verifiers = df.verifier.unique()
        for verifier in verifiers:
            verifier_df = df[df.verifier == verifier].sort_values(by="epsilon_value")
            cdf_x = np.linspace(0, 1, len(verifier_df))
            color = None
            if self.custom_colors is not None and verifier in self.custom_colors:
                color = self.custom_colors[verifier]
            plt.plot(verifier_df.epsilon_value, cdf_x, label=verifier, color=color)
            plt.fill_betweenx(cdf_x, verifier_df.epsilon_value, verifier_df.smallest_sat_value, alpha=0.3, color=color)
            plt.xlim(0, 0.35)
            plt.xlabel("Epsilon value")
            plt.ylabel("Fraction epsilon values found")
            plt.legend()

        return plt.gca()
