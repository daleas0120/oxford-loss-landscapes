import streamlit as st
import numpy as np
import plotly.graph_objs as go

st.set_page_config(layout="wide")

landscape = np.genfromtxt('examples/example_loss_landscape_bandgap_Fe_ood.csv', delimiter=',', usecols=None)

x = np.linspace(-1, 1, landscape.shape[0])
y = np.linspace(-1, 1, landscape.shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

# Sliders for axis ranges
st.sidebar.header("Axis Range Controls")
x_min = st.sidebar.slider("X min", float(x.min()), float(x.max()), float(x.min()))
x_max = st.sidebar.slider("X max", float(x.min()), float(x.max()), float(x.max()))
y_min = st.sidebar.slider("Y min", float(y.min()), float(y.max()), float(y.min()))
y_max = st.sidebar.slider("Y max", float(y.min()), float(y.max()), float(y.max()))
z_min = st.sidebar.slider("Z min", float(np.min(landscape)), float(np.max(landscape)), float(np.min(landscape)))
z_max = st.sidebar.slider("Z max", float(np.min(landscape)), float(np.max(landscape)), float(np.max(landscape)))

fig = go.Figure(data=[go.Surface(z=landscape, x=X, y=Y, colorscale='Viridis')])
fig.update_layout(
    scene=dict(
        xaxis=dict(title='Direction 1', range=[x_min, x_max]), 
        yaxis=dict(title='Direction 2', range=[y_min, y_max]),
        zaxis=dict(title='Loss', range=[z_min, z_max]),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(
            eye=dict(x=1, y=1, z=0.5)  # Change these values to set the initial view
        )
     ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600,
)

st.plotly_chart(fig, use_container_width=True) 

