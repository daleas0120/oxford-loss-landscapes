import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import State
# Load data
landscape = np.genfromtxt('examples/example_loss_landscape_bandgap_Fe_ood.csv', delimiter=',', usecols=None)
x = np.linspace(-1, 1, landscape.shape[0])
y = np.linspace(-1, 1, landscape.shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')

fig1 = go.Figure(data=[go.Surface(z=landscape, x=X, y=Y, colorscale='Viridis')])
fig2 = go.Figure(data=[go.Surface(z=landscape, x=X, y=Y, colorscale='Viridis')])

scene = dict(
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=1.25)
    ),
    aspectmode='manual',
    aspectratio=dict(x=1, y=1, z=1)
)

fig1.update_layout(scene=scene)
fig2.update_layout(scene=scene)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='camera-store', data={'x': 2, 'y': 2, 'z': 0.5}),
    html.H2("3D Loss Landscape Dashboard (Dash)"),
    html.Div([
        html.Div([
            html.Label("X min"),
            dcc.Slider(id='x-min', min=float(x.min()), max=float(x.max()), value=float(x.min()), step=0.01),
            html.Label("X max"),
            dcc.Slider(id='x-max', min=float(x.min()), max=float(x.max()), value=float(x.max()), step=0.01),
            html.Label("Y min"),
            dcc.Slider(id='y-min', min=float(y.min()), max=float(y.max()), value=float(y.min()), step=0.01),
            html.Label("Y max"),
            dcc.Slider(id='y-max', min=float(y.min()), max=float(y.max()), value=float(y.max()), step=0.01),
            html.Label("Z min"),
            dcc.Slider(id='z-min', min=float(np.min(landscape)), max=float(np.max(landscape)), value=float(np.min(landscape)), step=0.01),
            html.Label("Z max"),
            dcc.Slider(id='z-max', min=float(np.min(landscape)), max=float(np.max(landscape)), value=float(np.max(landscape)), step=0.01),
            html.Br(),
            html.Label("Notes / Textboard"),
            dcc.Textarea(id='textboard', value='', style={'width': '100%', 'height': 100}),
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        html.Div([
            html.Div([
                dcc.Graph(id='surface-plot1', style={'height': '600px', 'width': '100%'}, figure=fig1,
                          config={"displayModeBar": True}),
                html.Div(id='camera-eye-display', style={'marginTop': 10, 'fontWeight': 'bold'})
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                dcc.Graph(id='surface-plot2', style={'height': '600px', 'width': '100%'}, figure=fig2,
                          config={"displayModeBar": True})
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ]) 
])


# Main callback for figures and camera display
@app.callback(
    [Output('surface-plot1', 'figure'),
     Output('surface-plot2', 'figure')],
    [Input('x-min', 'value'),
     Input('x-max', 'value'),
     Input('y-min', 'value'),
     Input('y-max', 'value'),
     Input('z-min', 'value'),
     Input('z-max', 'value'),
     Input('surface-plot1', 'relayoutData')]
)
def update_figures(x_min, x_max, y_min, y_max, z_min, z_max, camera_eye):
    if camera_eye is None:
        return dash.no_update, dash.no_update

    if 'scene.camera' in camera_eye or 'x-min' in camera_eye or 'x-max' in camera_eye or 'y-min' in camera_eye or 'y-max' in camera_eye or 'z-min' in camera_eye or 'z-max' in camera_eye:
        
        print(callback_context.triggered)

        layout = dict(
            scene=dict(
             xaxis=dict(title='Direction 1', range=[x_min, x_max]),
             yaxis=dict(title='Direction 2', range=[y_min, y_max]),
             zaxis=dict(title='Loss', range=[z_min, z_max]),
         ),
         margin=dict(l=0, r=0, b=0, t=0),
         height=600,
        )
        fig1.update_layout(**layout)
        fig1.layout.scene.camera = camera_eye['scene.camera']
        fig2.layout.scene.camera = camera_eye['scene.camera']
        fig2.update_layout(uirevision='static', dragmode=False)
        #camera_text = f"Current camera eye: x = {camera_eye['x']}, y = {camera_eye['y']}, z = {camera_eye['z']}"
        return fig1, fig2

    return dash.no_update, dash.no_update



if __name__ == "__main__":
    app.run_server(port=8060, debug=True)
