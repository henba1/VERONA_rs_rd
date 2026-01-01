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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})
sns.set_palette(sns.color_palette("Paired"))


class ReportCreator:
    def __init__(self, df: pd.DataFrame, custom_colors: dict[str, str] | None = None):
        self.df = df
        self.custom_colors = custom_colors

    def create_hist_figure(self) -> plt.Figure:
        kwargs = {}
        if self.custom_colors is not None:
            kwargs["palette"] = self.custom_colors
        hist_plot = sns.histplot(data=self.df, x="epsilon_value", hue="network", multiple="stack", **kwargs)
        figure = hist_plot.get_figure()

        plt.close()

        return figure

    def create_box_figure(self) -> plt.Figure:
        kwargs = {}
        if self.custom_colors is not None:
            kwargs["palette"] = self.custom_colors
        box_plot = sns.boxplot(data=self.df, x="network", y="epsilon_value", **kwargs)
        box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=90)

        figure = box_plot.get_figure()

        plt.close()

        return figure

    def create_kde_figure(self) -> plt.Figure:
        kwargs = {}
        if self.custom_colors is not None:
            kwargs["palette"] = self.custom_colors
        kde_plot = sns.kdeplot(data=self.df, x="epsilon_value", hue="network", multiple="stack", **kwargs)

        figure = kde_plot.get_figure()

        plt.close()

        return figure

    def create_ecdf_figure(self) -> plt.Figure:
        kwargs = {}
        if self.custom_colors is not None:
            kwargs["palette"] = self.custom_colors
        ecdf_plot = sns.ecdfplot(data=self.df, x="epsilon_value", hue="network", **kwargs)

        figure = ecdf_plot.get_figure()

        plt.close()

        return figure

    def create_anneplot(self):
        df = self.df
        networks = df.network.unique()
        for _idx, network in enumerate(networks):
            network_df = df[df.network == network].sort_values(by="epsilon_value")
            cdf_x = np.linspace(0, 1, len(network_df))
            color = None
            if self.custom_colors is not None and network in self.custom_colors:
                color = self.custom_colors[network]
            plt.plot(network_df.epsilon_value, cdf_x, label=network, color=color)
            plt.fill_betweenx(cdf_x, network_df.epsilon_value, network_df.smallest_sat_value, alpha=0.3, color=color)
            plt.xlim(0, 4)
            plt.xlabel("Epsilon values")
            plt.ylabel("Fraction critical epsilon values found")
            plt.legend()

        return plt.gca()
