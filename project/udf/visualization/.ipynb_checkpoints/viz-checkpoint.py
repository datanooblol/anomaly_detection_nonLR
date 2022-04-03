import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def test_viz():
    print("test viz")
    
    
def viz_na(data, w=16,h=8):
    fig = plt.figure(figsize=(w,h))
    g = sns.heatmap(data.isna(), cbar=False)
    return g

def viz_sensor(data, sensor="00"):
    sensor_name = f"sensor_{sensor}"
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.loc[:,'timestamp'],
        y=data.loc[:,sensor_name],
        mode="lines",
        name=sensor_name,
        line=dict(color="royalblue")
    ))
    fig.update_layout(title=sensor_name)
    return fig