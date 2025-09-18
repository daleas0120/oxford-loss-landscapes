import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import State
import os
import datetime
import tomllib

# Define functions


def get_cube_edges_and_points(slider_values):
    """
    Returns the 8 cube corner points and the 12 edge index pairs for a cube
    bounded by the given min/max values.
    """
    # Define the 8 corners of the cube
    cube_points = [
        [slider_values[0], slider_values[2], slider_values[4]],
        [slider_values[1], slider_values[2], slider_values[4]],
        [slider_values[1], slider_values[3], slider_values[4]],
        [slider_values[0], slider_values[3], slider_values[4]],
        [slider_values[0], slider_values[2], slider_values[5]],
        [slider_values[1], slider_values[2], slider_values[5]],
        [slider_values[1], slider_values[3], slider_values[5]],
        [slider_values[0], slider_values[3], slider_values[5]],
    ]

    # Define the 12 edges as pairs of point indices
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ]
    return cube_points, edges


def convert_slider_to_data_ranges(
    x_min, x_max, y_min, y_max, z_min, z_max,
    x_data, y_data, landscape_data
):
    """
    Converts slider values (0 to 1) to actual data ranges.
    """
    X_range = np.max(x_data) - np.min(x_data)
    Y_range = np.max(y_data) - np.min(y_data)
    landscape_range = np.max(landscape_data) - np.min(landscape_data)
    slider_values = [x_data[0] + (x_min*X_range), 
                     x_data[0] + (x_max*X_range), 
                     y_data[0] + (y_min*Y_range),
                     y_data[0] + (y_max*Y_range),
                     np.min(landscape_data) + (z_min*landscape_range),
                     np.min(landscape_data) + (z_max*landscape_range)]
    return slider_values


def get_landscape_summary(slider_value_data, slider_step_data, landscape_data):
    """
    Returns text summary of loss landscape from current selection.
    """
    landscape = np.array(landscape_data)
    row_min = int((slider_value_data[0] - 0) / slider_step_data['x_step'])
    row_max = int((slider_value_data[1] - 0) / slider_step_data['x_step'])
    col_min = int((slider_value_data[2] - 0) / slider_step_data['y_step'])
    col_max = int((slider_value_data[3] - 0) / slider_step_data['y_step'])
    min_val = np.min(landscape[row_min:row_max, col_min:col_max])
    max_val = np.max(landscape[row_min:row_max, col_min:col_max])
    
    lol_summary_txt = (
        f"Minimum Loss: {min_val:.4f}\n"
        f"Maximum Loss: {max_val:.4f}"
    )
    return lol_summary_txt


# Load data
results_dir = os.path.join(os.getcwd(), "results")
npy_files = [f for f in os.listdir(results_dir) if f.endswith('.npy')]
npy_files_sorted = sorted(
    npy_files,
    key=lambda fname: (
        datetime.datetime.strptime(
            fname.split('_LOLxAI.npy')[0],
            "%Y%m%d_%H%M%S"
        )
        if '_LOLxAI.npy' in fname else datetime.datetime.min
    ),
    reverse=True 
)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='x_data'),
    dcc.Store(id='y_data'),
    dcc.Store(id='landscape_data'),
    dcc.Store(id='config'),
    dcc.Store(id='slider-min-store'),
    dcc.Store(id='slider-step-store'),
    html.H2("Loss Landscape Viewer"),
    html.Div([
        html.Div([
            html.Label("Select landscape file"),
            dcc.Dropdown(
                id='landscape-dropdown',
                options=[{'label': fname, 'value': fname} for fname in npy_files_sorted],
                value=npy_files_sorted[0],
                clearable=False,
                style={'marginBottom': '20px'}
            ),
            html.Label(" Direction 1 min"),
            dcc.Slider(
                id='slider-x-min', min=0, max=1, value=0, step=0.01, marks=None
            ),
            html.Label("Direction 1 max"),
            dcc.Slider(
                id='slider-x-max', min=0, max=1, value=1, step=0.01, marks=None
            ),
            html.Label("Direction 2 min"),
            dcc.Slider(
                id='slider-y-min', min=0, max=1, value=0, step=0.01, marks=None
            ),
            html.Label("Direction 2 max"),
            dcc.Slider(
                id='slider-y-max', min=0, max=1, value=1, step=0.01, marks=None
            ),
            html.Label("Loss min"),
            dcc.Slider(
                id='slider-z-min', min=0, max=1, value=0, step=0.01, marks=None
            ),
            html.Label("Loss max"),
            dcc.Slider(
                id='slider-z-max', min=0, max=1, value=1, step=0.01, marks=None
            ),

            ],
            style={
                'width': '25%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px'
            }
        ),
        html.Div([
            html.Div([
                dcc.Graph(
                    id='surface-plot2', style={'height': '400px', 'width': '100%'}, 
                    config={"displayModeBar": True}
                ),
                html.Br(),
                html.Div([
                    dcc.Markdown(
                        id='landscape_summary', children="holder"
                    )
                ], style={
                    'border': '2px solid #888', 'padding': '16px', 'margin': '16px 0', 
                    'borderRadius': '8px', 'backgroundColor': '#f9f9f9', 
                    'fontWeight': 'bold', 'textAlign': 'center'
                })  
            ], style={'width': '39%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                dcc.Graph(
                    id='surface-plot1', style={'height': '600px', 'width': '100%'}, 
                    config={"displayModeBar": True}
                ),
                html.Div(
                    id='camera-eye-display', style={'marginTop': 10, 'fontWeight': 'bold'}
                )
            ], style={'width': '59%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ]) 
])


# Callback for loading landscape data
@app.callback(
    [Output('x_data', 'data'),
     Output('y_data', 'data'),
     Output('landscape_data', 'data'),
     Output('slider-min-store', 'data'),
     Output('slider-step-store', 'data'),
     Output('config', 'data')],
    [Input('landscape-dropdown', 'value')]
)
def load_landscape_data(landscape_file):
    landscape = np.load(os.path.join(results_dir, landscape_file))
    base = os.path.splitext(landscape_file)[0]
    toml_file = f"{base}.toml"
    config = tomllib.load(open(os.path.join(results_dir, toml_file), "rb"))
    x = np.linspace(0, float(config['distance']), landscape.shape[0])
    y = np.linspace(0, float(config['distance']), landscape.shape[1])
    slider_min = {
        'x_min': float(np.min(x)),
        'y_min': float(np.min(y)), 
        'z_min': float(np.min(landscape))
    }
    slider_step = {'x_step': 1/x.shape[0], 'y_step': 1/y.shape[0]}
    
    return x, y, landscape, slider_min, slider_step, config
    

# Callback for updating slider steps
@app.callback(
    [Output('slider-x-min', 'step'),
     Output('slider-x-max', 'step'),
     Output('slider-y-min', 'step'),
     Output('slider-y-max', 'step')],
    [Input('slider-step-store', 'data')]
)   
def update_slider_steps(slider_step_data):
    if callback_context.triggered[0]['prop_id'] == '.':
        # Initial app load, do nothing
        return dash.no_update
    return (
        slider_step_data['x_step'],
        slider_step_data['x_step'],
        slider_step_data['y_step'],
        slider_step_data['y_step']
    )


# Callback for figures and camera display
@app.callback(
    [Output('surface-plot1', 'figure'),
     Output('surface-plot2', 'figure')],
    [Input('slider-x-min', 'value'),
     Input('slider-x-max', 'value'),
     Input('slider-y-min', 'value'),
     Input('slider-y-max', 'value'),
     Input('slider-z-min', 'value'),
     Input('slider-z-max', 'value'),
     Input('surface-plot1', 'relayoutData'),
     Input('x_data', 'data'),
     Input('y_data', 'data'),
     Input('landscape_data', 'data'),
     State('surface-plot1', 'figure'),
     State('surface-plot2', 'figure')]
)
def update_figures(
    x_min, x_max, y_min, y_max, z_min, z_max,
    camera_eye, x_data, y_data, landscape_data,
    fig1_prev, fig2_prev
):
    if (
        callback_context.triggered[0]['prop_id'] == 'x_data.data'
        or callback_context.triggered[0]['prop_id'] == 'y_data.data'
        or callback_context.triggered[0]['prop_id'] == 'landscape_data.data'
    ):
        X, Y = np.meshgrid(x_data, y_data, indexing='ij')
        fig1 = go.Figure(
            data=[go.Surface(z=landscape_data, x=X, y=Y, colorscale='Viridis')]
        )
        fig2 = go.Figure(
            data=[go.Surface(z=landscape_data, x=X, y=Y, colorscale='Viridis', showscale=False, opacity=0.9)]
        )
        layout = dict(
            scene=dict(
                xaxis=dict(title='Direction 1'),
                yaxis=dict(title='Direction 2'),
                zaxis=dict(title='Loss'),
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.55, y=1.55, z=1.25)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            ),
            margin=dict(l=5, r=5, b=10, t=20),
        )

        fig1.update_layout(**layout)
        fig2.update_layout(**layout)
        fig2.update_layout(uirevision='static', dragmode=False)

        # Create figure 2 focus cube
        # Get cube points and edges
        slider_values = convert_slider_to_data_ranges(
            x_min, x_max, y_min, y_max, z_min, z_max,
            x_data, y_data, landscape_data
        )
        cube_points, edges = get_cube_edges_and_points(slider_values)
        # Add a line for each edge
        for i, j in edges:
            fig2.add_trace(go.Scatter3d(
                x=[cube_points[i][0], cube_points[j][0]],
                y=[cube_points[i][1], cube_points[j][1]],
                z=[cube_points[i][2], cube_points[j][2]],
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False,
                name='Cube Edge'
            ))         
        return fig1, fig2
    
    if (
        'surface-plot1.relayoutData' in callback_context.triggered[0]['prop_id']
        and 'scene.camera' in camera_eye
    ):
        # Update layout with new camera
        fig1 = go.Figure(fig1_prev)
        fig2 = go.Figure(fig2_prev)
        fig1.layout.scene.camera = camera_eye['scene.camera']
        # Keep same distance from origin for figure 2
        a = camera_eye['scene.camera']['eye']['x']
        b = camera_eye['scene.camera']['eye']['y']
        c = camera_eye['scene.camera']['eye']['z']
        scale = 2.8 / (a**2 + b**2 + c**2) ** 0.5
        new_eye = {
            'x': a * scale,
            'y': b * scale,
            'z': c * scale
        }
        fig2.layout.scene.camera.eye = new_eye
        return fig1, fig2
   
    elif any(
        f"{axis}.value" in callback_context.triggered[0]['prop_id']
        for axis in ['x-min', 'x-max', 'y-min', 'y-max', 'z-min', 'z-max']
    ):    
        # Retrieve previous figures
        fig1 = go.Figure(fig1_prev)
        fig2 = go.Figure(fig2_prev)
        
        # Convert slider values to actual data ranges
        slider_values = convert_slider_to_data_ranges(
            x_min, x_max, y_min, y_max, z_min, z_max,
            x_data, y_data, landscape_data
        )
        # Update figure 1 with new axis ranges
        layout = dict(
            scene=dict(
                xaxis=dict(title='Direction 1', range=[slider_values[0], slider_values[1]]),
                yaxis=dict(title='Direction 2', range=[slider_values[2], slider_values[3]]),
                zaxis=dict(title='Loss', range=[slider_values[4], slider_values[5]]),
            ),
        )
        fig1.update_layout(**layout)

        # Update figure 2 focus cube
        # Get cube points and edges
        cube_points, edges = get_cube_edges_and_points(slider_values)
        # Add a line for each edge
        for idx, (i, j) in enumerate(edges):
            fig2.data[idx + 1].x = [cube_points[i][0], cube_points[j][0]]
            fig2.data[idx + 1].y = [cube_points[i][1], cube_points[j][1]]
            fig2.data[idx + 1].z = [cube_points[i][2], cube_points[j][2]]
        return fig1, fig2 
     
    return dash.no_update, dash.no_update


# Callback for updating landscape summary
@app.callback(
    [Output('landscape_summary', 'children')],
    [Input('slider-x-min', 'value'),
     Input('slider-x-max', 'value'),
     Input('slider-y-min', 'value'),
     Input('slider-y-max', 'value'),
     Input('slider-z-min', 'value'),
     Input('slider-z-max', 'value'),
     Input('slider-min-store', 'data'),
     Input('slider-step-store', 'data'),
     State('x_data', 'data'),
     State('y_data', 'data'),
     State('landscape_data', 'data')]
)
def update_summary(xmin_val, xmax_val, ymin_val, ymax_val, zmin_val, zmax_val, slider_min_data, slider_step_data, x_data_val, y_data_val, landscape_data):
    if callback_context.triggered[0]['prop_id'] == '.':
        # Initial app load, do nothing
        return dash.no_update
    
    slider_values = [xmin_val, xmax_val, ymin_val, ymax_val, zmin_val, zmax_val]
    summary_text = get_landscape_summary(slider_values, slider_step_data, landscape_data)
    
    return [f"```\n{summary_text}\n```"]


if __name__ == "__main__":
    app.run_server(port=8097, debug=True)
