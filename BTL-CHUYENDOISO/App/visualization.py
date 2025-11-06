import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def create_roadmap_fig(roadmap: list) -> go.Figure:
    fig = go.Figure()

    if len(roadmap) == 0:
        return fig  # Empty fig if no roadmap

    step_gap = 1.4
    box_height = 0.6
    box_width = 2.2
    x_center = 0

    for i, step in enumerate(roadmap):
        y_center = -i * step_gap

        fig.add_shape(
            type="rect",
            x0=x_center - box_width / 2,
            x1=x_center + box_width / 2,
            y0=y_center - box_height / 2,
            y1=y_center + box_height / 2,
            line=dict(color="#B71C1C", width=1.8),
            fillcolor="#FFCDD2",
            layer="below"
        )

        fig.add_annotation(
            x=x_center,
            y=y_center,
            text=step,
            showarrow=False,
            font=dict(size=12, color='black'),
            xanchor='center',
            yanchor='middle'
        )

        if i < len(roadmap) - 1:
            fig.add_annotation(
                x=x_center,
                y=y_center - box_height / 2,
                ax=x_center,
                ay=y_center - step_gap + box_height / 2,
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='black'
            )

    y_min = -((len(roadmap) - 1) * step_gap + 1)
    y_max = 1
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-2.5, 2.5]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[y_min, y_max]),
        height=max(250, len(roadmap) * 110),
        margin=dict(l=30, r=30, t=30, b=30),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def create_radar_fig(row: pd.Series, score_columns: list) -> px.line_polar:
    radar_data = pd.DataFrame(dict(
        r=[row[col] for col in score_columns],
        theta=score_columns
    ))
    fig_radar = px.line_polar(radar_data, r='r', theta='theta', line_close=True,
                              color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_radar.update_traces(fill='toself')
    return fig_radar

def create_bar_fig(row: pd.Series, avg_scores: pd.Series, score_columns: list) -> px.bar:
    compare_df = pd.DataFrame({
        'Môn học': score_columns,
        'Điểm cá nhân': [row[col] for col in score_columns],
        'Trung bình lớp': avg_scores.values
    })
    fig_bar = px.bar(compare_df, x='Môn học', y=['Điểm cá nhân', 'Trung bình lớp'], barmode='group',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    return fig_bar

def create_boxplot_fig(df: pd.DataFrame, score_columns: list) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df[score_columns], ax=ax, palette='Pastel1')
    ax.set_title('Boxplots of Scores')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    return fig

def create_pca_fig(features: np.ndarray, clusters: pd.Series) -> plt.Figure:
    pca_vis = PCA(n_components=2)
    reduced_vis = pca_vis.fit_transform(features)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=reduced_vis[:, 0], y=reduced_vis[:, 1], hue=clusters, palette='Pastel1', ax=ax)
    ax.set_title('Clusters Visualization (PCA)')
    return fig