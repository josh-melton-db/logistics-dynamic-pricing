import dash
from dash import dcc, html, Input, Output, State, callback_context, MATCH, ALL
import dash_bootstrap_components as dbc
from datetime import datetime, date
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import Format
import json
import os
from dotenv import load_dotenv
import threading
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import regex as re

# Load environment variables from .env file if it exists
load_dotenv()

model_serving_endpoint_name = "dynamic_pricing"
catalog = "manufacturing"
schema = "dynamic_pricing"
table = "features"
SQL_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID") 
FEATURES_TABLE = f"{catalog}.{schema}.{table}"
PRODUCT_TABLE = f"{catalog}.{schema}.product"
GEO_TABLE = f"{catalog}.{schema}.geography"

# Initialize Databricks client
w = WorkspaceClient()

# Default data based on your sample records
PRODUCTS = [
    {"label": "Product 913", "value": "913"},
    {"label": "Product 1962", "value": "1962"},
    {"label": "Product 1805", "value": "1805"},
    {"label": "Product 2175", "value": "2175"}
]

GEOGRAPHIES = [
    {"label": "Geography 1", "value": "1"}
]

# Default dates from your sample data
DEFAULT_DATES = ["20220718", "20220509", "20210526", "20210811"]

# Global variables to store loaded data
loaded_products = []
loaded_geographies = []
data_loading_complete = False

def load_data_from_databricks():
    """Load product and geography options from Databricks table"""
    global loaded_products, loaded_geographies, data_loading_complete
    
    try:
        print("Loading data from Databricks...")
        
        # Query distinct products with descriptions
        product_query = f"""
        SELECT DISTINCT f.productKey, p.productDescription 
        FROM {FEATURES_TABLE} f 
        JOIN {PRODUCT_TABLE} p ON f.productKey = p.productKey 
        ORDER BY f.productKey
        """
        print(f"Executing query: {product_query}")
        
        product_result = w.statement_execution.execute_statement(
            warehouse_id=SQL_WAREHOUSE_ID,
            statement=product_query,
            wait_timeout="30s"
        )
        
        # Extract products with descriptions from result
        if product_result.result and product_result.result.data_array:
            loaded_products = [
                {"label": f"{row[1]} (Product {row[0]})", "value": str(row[0])} 
                for row in product_result.result.data_array
            ]
            print(f"Loaded {len(loaded_products)} products")
        
        # Query distinct geographies with descriptions
        geography_query = f"""
        SELECT DISTINCT f.geographyKey, g.geographyDescription 
        FROM {FEATURES_TABLE} f 
        JOIN {GEO_TABLE} g ON f.geographyKey = g.geographyKey 
        ORDER BY f.geographyKey
        """
        print(f"Executing query: {geography_query}")
        
        geography_result = w.statement_execution.execute_statement(
            warehouse_id=SQL_WAREHOUSE_ID,
            statement=geography_query,
            wait_timeout="30s"
        )
        
        # Extract geographies with descriptions from result
        if geography_result.result and geography_result.result.data_array:
            loaded_geographies = [
                {"label": f"{row[1]} (Geography {row[0]})", "value": str(row[0])} 
                for row in geography_result.result.data_array
            ]
            print(f"Loaded {len(loaded_geographies)} geographies")
        else:
            print(str(geography_result))

        data_loading_complete = True
        print("Data loading complete!")
        
    except Exception as e:
        print(f"Error loading data from Databricks: {str(e)}")
        print("Using default data instead")
        # Keep using default data if there's an error

# Start loading data in background thread
threading.Thread(target=load_data_from_databricks, daemon=True).start()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Dynamic Pricing"

# Custom CSS for oat medium theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #EEEDE9 !important;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            .container-fluid {
                background-color: #EEEDE9;
            }
            /* Navbar specific styling to prevent override */
            .navbar {
                background-color: #343a40 !important;
            }
            .navbar .container-fluid {
                background-color: transparent !important;
            }
            .navbar-brand, .navbar-nav .nav-link {
                color: white !important;
            }
            .navbar-brand:hover, .navbar-nav .nav-link:hover {
                color: #f8f9fa !important;
            }
            /* Active navbar link styling */
            .navbar-nav .nav-link.active {
                border-bottom: 3px solid #ff3621 !important;
                font-weight: 600 !important;
            }
            .card {
                background-color: #FAFAF9 !important;
                border: 1px solid #D4D2CE !important;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
            }
            .card-header {
                background-color: #F5F4F1 !important;
                border-bottom: 1px solid #D4D2CE !important;
            }
            .btn-primary {
                background-color: #1f272d !important;
                border-color: #1f272d !important;
            }
            .btn-primary:hover {
                background-color: #0F1417 !important;
                border-color: #0F1417 !important;
            }
            .alert-success {
                background-color: #F0F8E8 !important;
                border-color: #1f272d !important;
                color: #4A5D23 !important;
            }
            .text-success {
                color: #4A5D23 !important;
            }
            .text-primary {
                color: #1f272d !important;
            }
            h1, h4 {
                color: #3D3D3D !important;
            }
            .form-control, .form-select {
                background-color: #FAFAF9 !important;
                border-color: #D4D2CE !important;
            }
            .form-control:focus, .form-select:focus {
                border-color: #1f272d !important;
                box-shadow: 0 0 0 0.2rem rgba(31, 39, 45, 0.25) !important;
            }
            .card.border-success {
                border-color: #1f272d !important;
            }
            .card.text-success {
                background-color: #F0F8E8 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# --- Genie Tab Component ---
def genie_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Genie Chatbot"),
                    dbc.CardBody([
                        # Wrap chat history and input in Loading
                        dcc.Loading(
                            id="loading",
                            type="default",
                            children=[
                                html.Div(id="chat-container", style={
                                    "height": "500px",
                                    "overflowY": "auto",
                                    "background": "#f8f9fa",
                                    "padding": "10px",
                                    "borderRadius": "5px",
                                    "marginBottom": "10px"
                                })
                            ]
                        ),
                        dbc.InputGroup([
                            dbc.Input(id="chat-input", placeholder="Type your message...", type="text"),
                            dbc.Button("Send", id="send-button", color="primary", n_clicks=0)
                        ]),
                        dcc.Store(id="chat-store", data=[])
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True)

# --- Dashboard Tab Component ---
def dashboard_tab():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("AI/BI Dashboard"),
                    dbc.CardBody([
                        dcc.Markdown(
                            '''<iframe
                            src="https://adb-984752964297111.11.azuredatabricks.net/embed/dashboardsv3/01f0634458a019a788d7807d78756658?o=984752964297111&f_dc962053%7Ecd66220d=Active%2520Shampoo"
                            width="100%"
                            height="600"
                            frameborder="0">
                            </iframe>''',
                            dangerously_allow_html=True
                        )
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True)

# --- Main App Layout with Header ---
app.layout = html.Div([
    # Navbar header - outside main container to be full width
    dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="/assets/logo.svg", height="45px"), className="me-3"),
                    dbc.Col([
                        html.H4("Dynamic Pricing Platform", className="text-white mb-0"),
                        html.Small("Intelligent Revenue Optimization", className="text-light")
                    ], className="d-flex flex-column justify-content-center")
                ],
                align="center",
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink(
                        "Dynamic Pricing",
                        href="#",
                        id="pricing-nav-link",
                        active=True,
                        className="ms-4 fs-6 fw-semibold text-light"
                    )),
                    dbc.NavItem(dbc.NavLink(
                        "Genie",
                        href="#",
                        id="genie-nav-link",
                        active=False,
                        className="ms-4 fs-6 fw-semibold text-light"
                    )),
                    dbc.NavItem(dbc.NavLink(
                        "Dashboard",
                        href="#",
                        id="dashboard-nav-link",
                        active=False,
                        className="ms-4 fs-6 fw-semibold text-light"
                    )),
                ]),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ], fluid=True),
        color="dark",
        dark=True,
        className="px-0 py-3",
        style={"marginBottom": "2rem"}
    ),
    
    # Main content container
    dbc.Container([
        html.Div(id="page-content"),
        # Loading, store, and interval components (keep outside page content)
        dcc.Loading(id="loading", type="default", children=html.Div(id="loading-output")),
        dcc.Store(id="prediction-history-store", data=[]),
        dcc.Store(id="current-page-store", data="tab-pricing"),  # Track current page
        dcc.Interval(id="data-loader-interval", interval=2000, n_intervals=0, max_intervals=30)
    ], fluid=True)
])

# --- Navbar Navigation Callbacks ---
@app.callback(
    [Output("current-page-store", "data"),
     Output("pricing-nav-link", "active"),
     Output("genie-nav-link", "active"),
     Output("dashboard-nav-link", "active")],
    [Input("pricing-nav-link", "n_clicks"),
     Input("genie-nav-link", "n_clicks"),
     Input("dashboard-nav-link", "n_clicks")],
    prevent_initial_call=True
)
def update_navigation(pricing_clicks, genie_clicks, dashboard_clicks):
    """Handle navbar navigation clicks"""
    ctx = callback_context
    if not ctx.triggered:
        return "tab-pricing", True, False, False
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "pricing-nav-link":
        return "tab-pricing", True, False, False
    elif button_id == "genie-nav-link":
        return "tab-genie", False, True, False
    elif button_id == "dashboard-nav-link":
        return "tab-dashboard", False, False, True
    
    return "tab-pricing", True, False, False

# --- Navbar Toggler Callback ---
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")]
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# --- Page Content Callback ---
@app.callback(
    Output("page-content", "children"),
    Input("current-page-store", "data")
)
def render_page_content(current_page):
    if current_page == "tab-pricing":
        # Use loaded data if available, otherwise use defaults
        global loaded_products, loaded_geographies
        product_options = loaded_products if loaded_products else PRODUCTS
        geography_options = loaded_geographies if loaded_geographies else GEOGRAPHIES
        
        # --- Existing Dynamic Pricing Layout ---
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Model Input Parameters", className="card-title"),
                        # Product Selection
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Product:", html_for="product-dropdown"),
                                dcc.Dropdown(
                                    id="product-dropdown",
                                    options=product_options,
                                    value="913",
                                    placeholder="Select a product...",
                                    className="mb-3"
                                )
                            ])
                        ]),
                        # Geography Selection
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Geography:", html_for="geography-dropdown"),
                                dcc.Dropdown(
                                    id="geography-dropdown",
                                    options=geography_options,
                                    value="1",
                                    placeholder="Select a geography...",
                                    className="mb-3"
                                )
                            ])
                        ]),
                        # Date Selection (Optional)
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Date (Optional):", html_for="date-picker"),
                                dcc.DatePickerSingle(
                                    id="date-picker",
                                    date=datetime.strptime("20220718", "%Y%m%d").date(),
                                    display_format="YYYY-MM-DD",
                                    className="mb-3"
                                )
                            ])
                        ]),
                        # Price Percent Change
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Price Percent Change (%):", html_for="price-percent-input"),
                                dbc.Input(
                                    id="price-percent-input",
                                    type="number",
                                    placeholder="Enter percentage change...",
                                    step=0.1,
                                    className="mb-3"
                                )
                            ])
                        ]),
                        # Submit Button
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Predict Volume & Revenue Impact",
                                    id="submit-button",
                                    color="primary",
                                    size="lg",
                                    className="w-100"
                                )
                            ])
                        ])
                    ])
                ], className="mb-4"),
                # Model Results Section
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Model Results", className="card-title"),
                        html.Div(id="results-output", className="mt-3"),
                        html.Div(id="error-output", className="mt-3")
                    ])
                ])
            ], width=6),
            # Chart Section
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Price Change vs Revenue Impact Analysis", className="card-title"),
                        dcc.Graph(
                            id="price-volume-chart",
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '580px'}
                        )
                    ], style={"height": "650px"})
                ])
            ], width=6)
        ], className="mt-4")
    elif current_page == "tab-genie":
        return genie_tab()
    elif current_page == "tab-dashboard":
        return dashboard_tab()
    return html.Div("Tab not found.")

# Callback for submitting to model endpoint
@app.callback(
    [Output("results-output", "children"),
     Output("error-output", "children"),
     Output("loading-output", "children"),
     Output("prediction-history-store", "data")],
    [Input("submit-button", "n_clicks")],
    [State("product-dropdown", "value"),
     State("geography-dropdown", "value"),
     State("date-picker", "date"),
     State("price-percent-input", "value"),
     State("prediction-history-store", "data")],
    prevent_initial_call=True
)
def submit_model_callback(n_clicks, product_key, geography_key, date_key, price_percent, prediction_history):
    if not n_clicks:
        return "", "", "", prediction_history or []
    
    # Hardcoded serving endpoint name
    
    # Validate required inputs
    if not product_key or not geography_key:
        error_msg = dbc.Alert(
            "Please fill in all required fields (Product and Geography).",
            color="danger"
        )
        return "", error_msg, "", prediction_history or []
    
    try:
        # Convert date to string format if provided
        date_str = None
        if date_key:
            date_obj = datetime.strptime(date_key, "%Y-%m-%d").date()
            date_str = date_obj.strftime("%Y%m%d")  # Format as YYYYMMDD
        
        # Prepare the record for the model
        record = {
            "productKey": product_key,
            "geographyKey": geography_key
        }
        
        # Add optional fields if provided
        if date_str:
            record["dateKey"] = date_str
        if price_percent is not None:
            record["pricePercentChange"] = price_percent
        
        records = [record]
        
        # Print what's being sent to the endpoint (for debugging)
        print(f"Sending to endpoint '{model_serving_endpoint_name}':")
        print(f"Records: {json.dumps(records, indent=2)}")
        
        # Submit to Databricks model serving endpoint
        response = w.serving_endpoints.query(
            name=model_serving_endpoint_name,
            dataframe_records=records
        )
        
        # Print the response from the endpoint (for debugging)
        print(f"Response from endpoint '{model_serving_endpoint_name}':")
        print(f"Full response: {json.dumps(response.as_dict(), indent=2)}")
        print(f"Predictions: {json.dumps(response.predictions, indent=2)}")
        
        # Format the results for display
        predictions = response.predictions
        
        # Create a new result entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_result = {
            "timestamp": timestamp,
            "product_key": product_key,
            "geography_key": geography_key,
            "date_key": date_str,
            "price_percent_change": price_percent,
            "predictions": predictions
        }
        
        # Add to prediction history (keep only last 10 results)
        updated_history = (prediction_history or [])
        updated_history.insert(0, new_result)  # Add to front
        updated_history = updated_history[:10]  # Keep only last 10
        
        # Create results cards for only the last 3 predictions (but keep all in store)
        results_cards = []
        display_history = updated_history[:3]  # Only show last 3 predictions
        
        for i, result in enumerate(display_history):
            # Create card header with timestamp and input parameters
            card_header = f"Prediction #{len(updated_history) - i} - {result['timestamp']}"
            
            # Create card body with input parameters and predictions
            card_body_content = [
                dbc.Row([
                    dbc.Col([
                        html.H6("Input Parameters:", className="text-primary"),
                        html.P([
                            html.Strong("Product: "), f"Product {result['product_key']}", html.Br(),
                            html.Strong("Geography: "), f"Geography {result['geography_key']}", html.Br(),
                            html.Strong("Date: "), result['date_key'] or "Not specified", html.Br(),
                            html.Strong("Price % Change: "), 
                            f"{result['price_percent_change']}%" if result['price_percent_change'] is not None else "Not specified"
                        ], className="mb-3")
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.H6("Estimated Volume % Change:", className="text-success"),
                        html.H4(f"{round(result['predictions'][0], 2)}%", 
                               className="text-success font-weight-bold",
                               style={"background-color": "#f8f9fa", "padding": "15px", "border-radius": "5px", "text-align": "center"})
                    ], width=6),
                    dbc.Col([
                        html.H6("Estimated Revenue Impact:", className="text-primary"),
                        html.H4(f"{round(((1 + result['price_percent_change']/100) * (1 + result['predictions'][0]/100) - 1) * 100, 2)}%" if result['price_percent_change'] is not None else "N/A", 
                               className="text-primary font-weight-bold",
                               style={"background-color": "#f8f9fa", "padding": "15px", "border-radius": "5px", "text-align": "center"})
                    ], width=6)
                ])
            ]
            
            # Determine card color (most recent is success, others are light)
            card_color = "success" if i == 0 else "light"
            card_outline = True if i == 0 else False
            
            card = dbc.Card([
                dbc.CardHeader(card_header),
                dbc.CardBody(card_body_content)
            ], color=card_color, outline=card_outline, className="mb-3")
            
            results_cards.append(card)
        
        return results_cards, "", "", updated_history
        
    except Exception as e:
        error_msg = dbc.Alert([
            html.H5("Error occurred:", className="alert-heading"),
            html.P(f"Error details: {str(e)}"),
            html.P("Please check your endpoint name and Databricks credentials.")
        ], color="danger")
        
        return "", error_msg, "", prediction_history or []

# Callback to disable data loader interval when complete
@app.callback(
    Output("data-loader-interval", "disabled"),
    [Input("data-loader-interval", "n_intervals")]
)
def update_data_loader_interval(n_intervals):
    global data_loading_complete
    
    # Stop the interval if data loading is complete or we've reached max intervals
    disable_interval = data_loading_complete or n_intervals >= 30
    
    if data_loading_complete:
        print("Data loading complete - disabling interval")
    
    return disable_interval

# Callback to update the price vs volume change chart
@app.callback(
    Output("price-volume-chart", "figure"),
    [Input("prediction-history-store", "data"),
     Input("product-dropdown", "value"),
     Input("geography-dropdown", "value")],
    prevent_initial_call=True
)
def update_price_volume_chart_callback(prediction_history, selected_product, selected_geography):
    """Update the scatter plot showing price change vs volume change relationship with revenue impact"""
    
    if not prediction_history:
        # Return empty chart if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available yet.<br>Submit some predictions to see the chart!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis_title="Price Change (%)",
            yaxis_title="Volume Change (%)",
            template="plotly_white"
        )
        return fig
    
    # Filter prediction history to only include results for the currently selected product and geography
    filtered_history = []
    for result in prediction_history:
        if (result.get('product_key') == selected_product and 
            result.get('geography_key') == selected_geography):
            filtered_history.append(result)
    
    if not filtered_history:
        # Return empty chart if no data for current selection
        fig = go.Figure()
        fig.add_annotation(
            text=f"No prediction data for Product {selected_product} and Geography {selected_geography}.<br>Submit predictions for this combination to see the chart!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis_title="Price Change (%)",
            yaxis_title="Volume Change (%)",
            template="plotly_white"
        )
        return fig
    
    # Extract data for plotting and sort by price change
    data_points = []
    
    for result in filtered_history:
        if result.get('price_percent_change') is not None and result.get('predictions'):
            price_change = result['price_percent_change']
            volume_change = result['predictions'][0]  # First prediction value
            
            # Calculate revenue impact: ((1 + price_change/100) × (1 + volume_change/100) - 1) × 100
            revenue_impact = ((1 + price_change/100) * (1 + volume_change/100) - 1) * 100
            
            # Create hover text with additional info including revenue impact
            hover_text = (
                f"Product: {result['product_key']}<br>"
                f"Geography: {result['geography_key']}<br>"
                f"Date: {result.get('date_key', 'Not specified')}<br>"
                f"Revenue Impact: {revenue_impact:.2f}%<br>"
                f"Timestamp: {result['timestamp']}"
            )
            
            data_points.append({
                'price_change': price_change,
                'volume_change': volume_change,
                'revenue_impact': revenue_impact,
                'hover_text': hover_text
            })
    
    # Sort data points by price change to connect dots in order
    data_points.sort(key=lambda x: x['price_change'])
    
    # Extract sorted arrays for plotting
    price_changes = [point['price_change'] for point in data_points]
    volume_changes = [point['volume_change'] for point in data_points]
    revenue_impacts = [point['revenue_impact'] for point in data_points]
    hover_data = [point['hover_text'] for point in data_points]
    
    if not price_changes:
        # Return empty chart if no valid data points
        fig = go.Figure()
        fig.add_annotation(
            text=f"No data points with price change values for Product {selected_product} and Geography {selected_geography}.<br>Submit predictions with price changes to see the chart!",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis_title="Price Change (%)",
            yaxis_title="Volume Change (%)",
            template="plotly_white"
        )
        return fig
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add scatter plot with revenue impact as color scale
    fig.add_trace(go.Scatter(
        x=price_changes,
        y=volume_changes,
        mode='markers+lines',
        marker=dict(
            size=12,
            color=revenue_impacts,
            colorscale='RdYlGn',  # Red for negative, Green for positive revenue impact
            showscale=True,
            colorbar=dict(title="Revenue Impact (%)")
        ),
        line=dict(width=2, color='rgba(100, 100, 100, 0.3)'),
        hovertemplate='<b>Price Change:</b> %{x}%<br><b>Volume Change:</b> %{y}%<br>%{text}<extra></extra>',
        text=hover_data,
        name='Predictions'
    ))
    
    # Add trend line if we have multiple points
    if len(price_changes) > 1:
        # Calculate trend line using numpy polyfit equivalent
        z = np.polyfit(price_changes, volume_changes, 1)
        p = np.poly1d(z)
        
        x_trend = [min(price_changes), max(price_changes)]
        y_trend = [p(x) for x in x_trend]
        
        fig.add_trace(go.Scatter(
            x=x_trend,
            y=y_trend,
            mode='lines',
            line=dict(dash='dash', color='red', width=2),
            name='Trend Line',
            hovertemplate='Trend Line<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="Price Change (%)",
        yaxis_title="Volume Change (%)",
        template="plotly_white",
        margin=dict(l=60, r=60, t=40, b=60),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# --- Chatbot Callback ---
@app.callback(
    Output("chat-container", "children"),
    Output("chat-store", "data"),
    Input("send-button", "n_clicks"),
    State("chat-input", "value"),
    State("chat-store", "data"),
    prevent_initial_call=True
)
def update_chat(n_clicks, user_message, chat_history):
    if not user_message:
        return dash.no_update, chat_history
    chat_history = chat_history or []
    chat_history.append({"role": "user", "text": user_message})

    # Same processing logic as before...
    genie_query = f"""SELECT manufacturing.dynamic_pricing._genie_query(
      \"https://e2-demo-west.cloud.databricks.com/\",
      'dapi...',,
      {repr(user_message)},
      \"\"
    )"""

    response = w.statement_execution.execute_statement(
        warehouse_id=SQL_WAREHOUSE_ID,
        statement=genie_query,
        wait_timeout="45s",
        format=Format.JSON_ARRAY
    )
    all_rows = response.result.data_array if response.result and response.result.data_array else []
    genie_string = all_rows[0][0]
    pattern = re.compile(r'(\w[\w\s]*):\s*(.+)')
    matches = pattern.findall(genie_string)
    result_dict = {}
    for key, value in matches:
        result_dict[key.strip()] = value.strip()
    json_output_string = json.dumps(result_dict, indent=4)
    genie_result = json.loads(json_output_string)

    genie_result_str = str(genie_result).replace("'", "''")
    query = f"""
    select ai_query(
    'databricks-gpt-oss-120b',
    'convert and display this as markdown format. Show the Query Result in a table format: {genie_result_str}'
    )
    """
    response = w.statement_execution.execute_statement(
    warehouse_id=SQL_WAREHOUSE_ID,
    statement=query,
    wait_timeout="45s",
    format=Format.JSON_ARRAY
    )

    genie_result_markdown = response.result.data_array if response.result and response.result.data_array else []
    
    # Extract the markdown content from the response
    markdown_content = ""
    if genie_result_markdown and len(genie_result_markdown) > 0:
        markdown_content = genie_result_markdown[0][0] if genie_result_markdown[0] else ""
    else:
        # Fallback to original result if markdown conversion failed
        markdown_content = f"""
### Analysis Results

"""
        for key, value in genie_result.items():
            markdown_content += f"**{key}:** {value}\n\n"

    chat_history.append({"role": "bot", "text": markdown_content})

    # Build chat bubbles
    chat_bubbles = []
    for msg in chat_history:
        align = "left" if msg["role"] == "bot" else "right"
        color = "#e1e1e1" if msg["role"] == "bot" else "#d1e7dd"
        
        if msg["role"] == "bot":
            # Use dcc.Markdown for bot messages to render markdown
            chat_bubbles.append(
                html.Div(
                    dcc.Markdown(msg["text"]),
                    style={
                        "textAlign": align,
                        "background": color,
                        "borderRadius": "10px",
                        "padding": "8px",
                        "margin": "4px 0"
                    }
                )
            )
        else:
            # Keep user messages as plain text
            chat_bubbles.append(
                html.Div(msg["text"], style={
                    "textAlign": align,
                    "background": color,
                    "borderRadius": "10px",
                    "padding": "8px",
                    "margin": "4px 0"
                })
            )
    return chat_bubbles, chat_history

if __name__ == "__main__":
    app.run(
        host="0.0.0.0", 
        port=os.getenv("DATABRICKS_APP_PORT", "8000"),
        # debug=True
    )
