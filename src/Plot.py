#!/usr/bin/env python
# coding=utf-8
#? -------------------------------------------------------------------------------
#?
#?                 ______  ____  _______  _____
#?                / __ \ \/ /  |/  / __ \/ ___/
#?               / /_/ /\  / /|_/ / / / /\__ \
#?              / ____/ / / /  / / /_/ /___/ /
#?             /_/     /_/_/  /_/\____//____/
#?
#? Name:        plot.py
#? Purpose:     Plot and compare results from different MOSFET models using Plotly
#?
#? Author:      Mohamed Gueni (mohamedgueni@outlook.com)
#?
#? Created:     21/05/2025
#? Licence:     Refer to the LICENSE file
#? -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
import re
#? -------------------------------------------------------------------------------
class MOSFETModelComparer:
    def __init__(self, csv_paths, output_html=None):
        if len(csv_paths) < 1:
            raise ValueError("At least one CSV file is required.")
        self.csv_paths      = csv_paths
        self.output_html    = output_html or r'D:\WORKSPACE\PyModules\10_pymos\data\comparison_plot.html'
        self.models         = self._load_models()

    def _load_models(self):
        models = []
        required_cols = {"time", "T", "VGS", "VDS", "ID"}
        for path in self.csv_paths:
            df = pd.read_csv(path)
            if not required_cols.issubset(df.columns):
                raise ValueError(f"CSV '{path}' missing required columns: {required_cols}")
            name = self._extract_model_name(path)
            models.append({"name": name, "df": df})
        return models

    def _extract_model_name(self, path):
        # Extract file name without extension and clean it
        return re.sub(r'[_\-]', ' ', os.path.splitext(os.path.basename(path))[0]).strip()

    def _create_dropdown_menu(self):
        # Create dropdown menu options with multi-select capability
        options = [
            {'label': 'All Models', 'value': 'all'},
            {'label': 'None', 'value': 'none'}
        ]
        options.extend([{'label': model['name'], 'value': model['name']} for model in self.models])
        
        # Create custom buttons for selecting combinations
        if len(self.models) > 1:
            for i in range(len(self.models)):
                for j in range(i+1, len(self.models)):
                    name1 = self.models[i]['name']
                    name2 = self.models[j]['name']
                    options.append({
                        'label': f"{name1} & {name2}",
                        'value': f"{name1},{name2}"
                    })
        return options

    def _create_figure_with_dropdown(self, title, x_title, y_title, traces):
        fig = go.Figure()
        
        # Add all traces initially (will be controlled by dropdown)
        for trace in traces:
            fig.add_trace(trace)
        
        # Create dropdown buttons
        dropdown_buttons = []
        options = self._create_dropdown_menu()
        
        # Create visibility lists for each option
        visibility_lists = []
        for option in options:
            if option['value'] == 'all':
                # Show all traces
                visibility = [True] * len(traces)
            elif option['value'] == 'none':
                # Hide all traces
                visibility = [False] * len(traces)
            else:
                # Show traces for selected models (can be multiple)
                selected_models = option['value'].split(',')
                visibility = [
                    any(trace.name.startswith(model) for model in selected_models)
                    for trace in traces
                ]
            visibility_lists.append(visibility)
        
        # Create buttons for dropdown
        for i, option in enumerate(options):
            dropdown_buttons.append(
                dict(
                    label=option['label'],
                    method='update',
                    args=[{'visible': visibility_lists[i]},
                          {'title': f"{title} - {option['label']}"}]
                )
            )
        
        # Update layout with dropdown
        fig.update_layout(
            title=f"{title} - All Models",
            xaxis_title=x_title,
            yaxis_title=y_title,
            updatemenus=[{
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 1.0,
                'xanchor': 'right',
                'y': 1.15,
                'yanchor': 'top'
            }]
        )
        
        return fig

    def plot(self):
        # Prepare all traces for each figure
        id_temp_traces = []
        id_vgs_traces = []
        id_vds_traces = []
        
        for model in self.models:
            df = model["df"]
            
            # Id vs Temperature traces
            idx = (df["VGS"] == 20) & (df["VDS"] == 10)
            id_temp_traces.append(
                go.Scatter(x=df["T"][idx], y=df["ID"][idx], name=f"{model['name']} Id")
            )
            
            # Id vs Vgs traces
            for vds in np.unique(df["VDS"]):
                idx = (df["VDS"] == vds) & (df["T"] == 400)
                if np.any(idx):
                    id_vgs_traces.append(
                        go.Scatter(x=df["VGS"][idx], y=df["ID"][idx],
                        name=f"{model['name']} Vds={vds:.1f}")
                    )
            
            # Id vs Vds traces
            for vgs in np.unique(df["VGS"]):
                idx = (df["VGS"] == vgs) & (df["T"] == 400)
                if np.any(idx):
                    id_vds_traces.append(
                        go.Scatter(x=df["VDS"][idx], y=df["ID"][idx],
                        name=f"{model['name']} Vgs={vgs:.1f}")
                    )
        
        # Create figures with dropdown menus
        fig_id_temp = self._create_figure_with_dropdown(
            "Id vs Temperature", "Temperature [K]", "Id [A]", id_temp_traces)
        fig_id_vgs = self._create_figure_with_dropdown(
            "Id vs Vgs", "Vgs [V]", "Id [A]", id_vgs_traces)
        fig_id_vds = self._create_figure_with_dropdown(
            "Id vs Vds", "Vds [V]", "Id [A]", id_vds_traces)
        
        figures = [fig_id_temp, fig_id_vgs, fig_id_vds]

        # --- Save HTML ---
        os.makedirs(os.path.dirname(self.output_html), exist_ok=True)
        html_parts = [pio.to_html(fig, full_html=False, include_plotlyjs='cdn') for fig in figures]
        full_html = """
        <html>
        <head>
            <title>MOSFET Comparison</title>
            <style>
                .plot-container {
                    margin: 20px auto;
                    max-width: 1200px;
                }
                .instructions {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    max-width: 1200px;
                    margin: 0 auto 20px auto;
                }
            </style>
        </head>
        <body>
            <div class="instructions">
                <h2>MOSFET Model Comparison</h2>
                <p>Use the dropdown menus in each plot to select which models to display.</p>
                <p>Options include viewing all models together, none, individual models, or combinations of models.</p>
            </div>
        """ + "\n<div class='plot-container'>\n".join(html_parts) + "\n</body></html>"

        with open(self.output_html, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(f"HTML with all plots saved to: {self.output_html}")

#? -------------------------------------------------------------------------------