import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import State

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

def get_landscape_summary(slider_value_data, slider_min_data, slider_step_data):
    """
    Returns text summary of loss landscape from current selection.
    """
    row_min = int((slider_value_data[0] - slider_min_data['x_min']) / slider_step_data['x_step'])
    row_max = int((slider_value_data[1] - slider_min_data['x_min']) / slider_step_data['x_step'])
    col_min = int((slider_value_data[2] - slider_min_data['y_min']) / slider_step_data['y_step'])
    col_max = int((slider_value_data[3] - slider_min_data['y_min']) / slider_step_data['y_step'])
    min_val = np.min(landscape[row_min:row_max, col_min:col_max])
    max_val = np.max(landscape[row_min:row_max, col_min:col_max])

    lol_summary_txt= f"Minimum Loss: {min_val:.4f}\nMaximum Loss: {max_val:.4f}"
    return lol_summary_txt

# Load data
landscape = np.genfromtxt('/Users/cdharding/Downloads/gpt2_loss_landscape.csv', delimiter=',', usecols=None)
x = np.linspace(1, landscape.shape[0], landscape.shape[0])
y = np.linspace(1, landscape.shape[1], landscape.shape[1])
X, Y = np.meshgrid(x, y, indexing='ij')


fig1 = go.Figure(data=[go.Surface(z=landscape, x=X, y=Y, colorscale='Viridis')])
fig2 = go.Figure(data=[go.Surface(z=landscape, x=X, y=Y, colorscale='Viridis', showscale=False, opacity=0.9)])

# Get cube points and edges
slider_values_init = [float(x.min()), float(x.max()), float(y.min()), float(y.max()), float(np.min(landscape)), float(np.max(landscape))]
cube_points, edges = get_cube_edges_and_points(slider_values_init)
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




layout = dict(
    scene = dict(
        camera = dict(
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


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='slider-min-store', data={'x_min': float(np.min(x)), 'y_min': float(np.min(y)), 'z_min': float(np.min(landscape))}),
    dcc.Store(id='slider-step-store', data={'x_step': x[1] - x[0], 'y_step': y[1] - y[0]}),
    html.H2("3D Loss Landscape Dashboard (Dash)"),
    html.Div([
        html.Div([
            html.Label(" Direction 1 min"),
            dcc.Slider(id='slider-x-min', min=float(x.min()), max=float(x.max()), value=float(x.min()), step=x[1] - x[0], marks=None),
            html.Label("Direction 1 max"),
            dcc.Slider(id='slider-x-max', min=float(x.min()), max=float(x.max()), value=float(x.max()), step=x[1] - x[0], marks=None),
            html.Label("Direction 2 min"),
            dcc.Slider(id='slider-y-min', min=float(y.min()), max=float(y.max()), value=float(y.min()), step=y[1] - y[0], marks=None),
            html.Label("Direction 2 max"),
            dcc.Slider(id='slider-y-max', min=float(y.min()), max=float(y.max()), value=float(y.max()), step=y[1] - y[0], marks=None),
            html.Label("Loss min"),
            dcc.Slider(id='slider-z-min', min=float(np.min(landscape)), max=float(np.max(landscape)), value=float(np.min(landscape)), step=0.01, marks=None),
            html.Label("Loss max"),
            dcc.Slider(id='slider-z-max', min=float(np.min(landscape)), max=float(np.max(landscape)), value=float(np.max(landscape)), step=0.01, marks=None),

        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        html.Div([
            html.Div([
                dcc.Graph(id='surface-plot1', style={'height': '600px', 'width': '100%'}, figure=fig1,
                          config={"displayModeBar": True}),
                html.Div(id='camera-eye-display', style={'marginTop': 10, 'fontWeight': 'bold'})
            ], style={'width': '59%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                dcc.Graph(id='surface-plot2', style={'height': '400px', 'width': '100%'}, figure=fig2,
                          config={"displayModeBar": True}),
                html.Br(),
                html.Div([
                    dcc.Markdown(id='landscape_summary',children="**Bold text** and _italic text_")
                ],style={'border': '2px solid #888','padding': '16px','margin': '16px 0','borderRadius': '8px',
                    'backgroundColor': '#f9f9f9','fontWeight': 'bold','textAlign': 'center'
                })  
            ], style={'width': '39%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ]) 
])


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
     Input('surface-plot1', 'relayoutData')]
)
def update_figures(x_min, x_max, y_min, y_max, z_min, z_max, camera_eye):
    if camera_eye is None:
        return dash.no_update, dash.no_update
    
    if 'surface-plot1.relayoutData' in callback_context.triggered[0]['prop_id'] and 'scene.camera' in camera_eye:
        # Update layout with new camera
        fig1.layout.scene.camera = camera_eye['scene.camera']
        fig2.layout.scene.camera = camera_eye['scene.camera']
        return fig1, fig2
   
    elif any(
    f"{axis}.value" in callback_context.triggered[0]['prop_id']
    for axis in ['x-min', 'x-max', 'y-min', 'y-max', 'z-min', 'z-max']
    ):    
        # Update figure 1 with new axis ranges
        layout = dict(
            scene=dict(
             xaxis=dict(title='Direction 1', range=[x_min, x_max]),
             yaxis=dict(title='Direction 2', range=[y_min, y_max]),
             zaxis=dict(title='Loss', range=[z_min, z_max]),
         ),
        )
        fig1.update_layout(**layout)

        # Update figure 2 focues cube
        # Get cube points and edges
        slider_values = [x_min, x_max, y_min, y_max, z_min, z_max]
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
     Input('slider-step-store', 'data')]
)

def update_summary(xmin_val, xmax_val, ymin_val, ymax_val, zmin_val, zmax_val, slider_min_data, slider_step_data):

    slider_value_data = [xmin_val, xmax_val, ymin_val, ymax_val, zmin_val, zmax_val]
    summary_text = get_landscape_summary(slider_value_data, slider_min_data, slider_step_data)
    
    return [f"```\n{summary_text}\n```"]

if __name__ == "__main__":
    app.run_server(port=8093, debug=True)
