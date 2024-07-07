import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
import joblib
import os
from openai import OpenAI

# Load the combined CSV file
beverages_df = pd.read_csv('C:/Users/morit/OneDrive/Dokumente/ESADE/Term3/AI Prototypes/Assignment 2/data/beverages_combined.csv')
print("DataFrame loaded successfully")
print(beverages_df.info())

# Load the pre-trained model
model = joblib.load('C:/Users/morit/OneDrive/Dokumente/ESADE/Term3/AI Prototypes/Assignment 2/nutriscore_decision_tree_model.pkl')
# Filter data to include only valid Nutriscore grades
valid_grades = ['a', 'b', 'c', 'd', 'e']
beverages_df_filtered = beverages_df[beverages_df['Nutriscore Grade'].isin(valid_grades)]

# Create visualizations
fig_nutriscore = px.bar(
    beverages_df_filtered.sort_values('Nutriscore Grade'),  # Sort by Nutriscore Grade
    x='Nutriscore Grade',
    title='Distribution of Nutriscore Grades',
    color='Nutriscore Grade',
    category_orders={'Nutriscore Grade': ['a', 'b', 'c', 'd', 'e']},  # Ensure the order is correct
    color_discrete_map={
        'a': '#4caf50',  # Green
        'b': '#8bc34a',  # Light Green
        'c': '#ffeb3b',  # Yellow
        'd': '#ffc107',  # Amber
        'e': '#f44336'   # Red
    },
    hover_data=['Categories', 'Brands']
)

fig_nutriscore.update_layout(
    xaxis_title='Nutriscore Grade',
    yaxis_title='Count',
    title_x=0.5,
    template='plotly_white'
)

fig_ecoscore = px.bar(
    beverages_df, 
    x='Ecoscore Grade', 
    title='Distribution of Ecoscore Grades',
    color='Ecoscore Grade',
    category_orders={'Ecoscore Grade': ['a', 'b', 'c', 'd', 'e']}, 
    color_discrete_map={
        'a': '#4caf50',  # Green
        'b': '#8bc34a',  # Light Green
        'c': '#ffeb3b',  # Yellow
        'd': '#ffc107',  # Amber
        'e': '#f44336'   # Red
    },
    hover_data=['Categories', 'Brands']
)

fig_ecoscore.update_layout(
    xaxis_title='Ecoscore Grade',
    yaxis_title='Count',
    title_x=0.5,
    template='plotly_white'
)

# Convert non-numeric values to NaN for relevant columns
numeric_columns = ['Sugars (g)', 'Proteins (g)', 'Fat (g)', 'Saturated Fat (g)', 'Salt (g)', 'Energy (kcal)']
beverages_df[numeric_columns] = beverages_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in these columns
beverages_df_cleaned = beverages_df.dropna(subset=numeric_columns)

# Calculate relative nutritional values by country
avg_nutritional_values = beverages_df_cleaned.groupby('Countries')[numeric_columns].mean().reset_index()
avg_nutritional_values[numeric_columns] = avg_nutritional_values[numeric_columns].div(avg_nutritional_values[numeric_columns].mean())

# Create additional visualizations
fig_heatmap = px.imshow(
    avg_nutritional_values.set_index('Countries').T,
    labels=dict(x='Countries', y='Nutritional Metrics', color='Relative Value'),
    title='Heatmap of Relative Nutritional Values by Country'
)

fig_stacked_bar = px.bar(
    avg_nutritional_values.melt(id_vars='Countries', value_vars=numeric_columns),
    x='Countries',
    y='value',
    color='variable',
    title='Stacked Bar Chart of Nutritional Composition by Country'
)

# Aggregate the data to show top categories by energy
top_categories = beverages_df.groupby('Categories')['Energy (kcal)'].sum().nlargest(5).index
filtered_df = beverages_df[beverages_df['Categories'].isin(top_categories)]

fig_sunburst = px.sunburst(
    filtered_df,
    path=['Countries', 'Categories'],
    values='Energy (kcal)',
    title='Sunburst Chart for Top 5 Food Categories by Country'
)

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the layout of the app
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="Beverage Analysis Dashboard",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("Country Comparison Analysis", href="/country-comparison")),
            dbc.NavItem(dbc.NavLink("Individual Product Analysis", href="/product-analysis")),
        ]
    ),
    dbc.Container([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ], fluid=True, className="mt-4")
], fluid=True)

# Define the callback to update the content based on the current URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/country-comparison':
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader(html.H2("Country Comparison Analysis")),
                dbc.CardBody([
                    dcc.Graph(id='nutriscore-bar-chart', figure=fig_nutriscore),
                    dbc.Alert(
                        "Click on a Nutriscore or Ecoscore grade in the bar charts to see the average nutritional composition for products with that grade.",
                        color="info",
                        dismissable=True
                    ),
                    dcc.Graph(id='nutritional-composition-radar'),
                    dcc.Graph(id='ecoscore-bar-chart', figure=fig_ecoscore),
                    dcc.Graph(id='heatmap', figure=fig_heatmap),
                    dcc.Graph(id='stacked-bar', figure=fig_stacked_bar),
                    dcc.Graph(id='sunburst', figure=fig_sunburst),
                    html.Label("Select Nutritional Metrics:"),
                    dcc.Dropdown(
                        id='nutrition-metrics-dropdown',
                        options=[{'label': col, 'value': col} for col in numeric_columns],
                        value=numeric_columns[0],
                        multi=False
                    ),
                    dcc.Graph(id='nutrition-boxplot')
                ])
            ], className="mt-4")
        ])
    elif pathname == '/product-analysis':
        return dbc.Container([
            dbc.Card([
                dbc.CardHeader(html.H2("Individual Product Analysis")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='product-dropdown',
                                options=[{'label': prod, 'value': prod} for prod in beverages_df['Product Code'].unique()],
                                placeholder='Select a product',
                                multi=True
                            )
                        ], width=12)
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col([dcc.Graph(id='product-nutriscore-bar')], width=6),
                        dbc.Col([dcc.Graph(id='product-ecoscore-pie')], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([dcc.Graph(id='product-nutrition-radar')], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dcc.Textarea(
                                id='ai-input',
                                placeholder='Enter your query here...',
                                style={'width': '100%', 'height': 100},
                                className="form-control"
                            ),
                            dbc.Button('Get Insights', id='ai-button', color="primary", className="mt-3"),
                            html.Div(id='ai-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
                        ], width=12)
                    ])
                ])
            ], className="mt-4"),
            dbc.Card([
                dbc.CardHeader(html.H3("Predict Nutriscore for New Product")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Sugars (g):"),
                            dcc.Input(id='input-sugars', type='number', value=0, className="form-control"),
                            html.Br(),
                            dbc.Label("Proteins (g):"),
                            dcc.Input(id='input-proteins', type='number', value=0, className="form-control"),
                            html.Br(),
                            dbc.Label("Fat (g):"),
                            dcc.Input(id='input-fat', type='number', value=0, className="form-control"),
                            html.Br(),
                            dbc.Label("Saturated Fat (g):"),
                            dcc.Input(id='input-saturated-fat', type='number', value=0, className="form-control"),
                            html.Br(),
                            dbc.Label("Salt (g):"),
                            dcc.Input(id='input-salt', type='number', value=0, className="form-control"),
                            html.Br(),
                            dbc.Label("Energy (kcal):"),
                            dcc.Input(id='input-energy', type='number', value=0, className="form-control"),
                            html.Br(),
                            dbc.Button('Predict Nutriscore', id='predict-button', color="primary", className="mt-3"),
                            html.Div(id='prediction-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
                        ], width=6)
                    ])
                ])
            ], className="mt-4")
        ])
    else:
        return dbc.Container([html.H2("Welcome to the Beverage Analysis Dashboard")])

# Callback to update the nutrition box plot based on selected metric
@app.callback(
    Output('nutrition-boxplot', 'figure'),
    Input('nutrition-metrics-dropdown', 'value'),
    Input('ecoscore-bar-chart', 'clickData')
)
def update_nutrition_boxplot(selected_metric, click_data):
    filtered_df = beverages_df_cleaned
    if click_data:
        country = click_data['points'][0]['x']
        filtered_df = beverages_df_cleaned[beverages_df_cleaned['Countries'] == country]
    
    fig_boxplot = px.box(
        filtered_df, 
        x='Countries', 
        y=selected_metric,
        title=f'Distribution of {selected_metric} by Country',
        points=False
    )
    fig_boxplot.update_layout(
        xaxis_title='Countries',
        yaxis_title=selected_metric,
        title_x=0.5,
        template='plotly_white'
    )
    return fig_boxplot

# Callback to update the product details graph based on selected product
@app.callback(
    [Output('product-nutriscore-bar', 'figure'),
     Output('product-ecoscore-pie', 'figure'),
     Output('product-nutrition-radar', 'figure')],
    [Input('product-dropdown', 'value')]
)
def update_product_analysis(selected_products):
    if not selected_products:
        return {}, {}, {}

    filtered_data = beverages_df[beverages_df['Product Code'].isin(selected_products)]

    fig_nutriscore_bar = px.bar(
        filtered_data,
        x='Product Code',
        y='Nutriscore Score',
        color='Nutriscore Grade',
        title='Nutriscore Distribution for Selected Products'
    )

    fig_ecoscore_pie = px.pie(
        filtered_data,
        names='Ecoscore Grade',
        title='Ecoscore Distribution for Selected Products'
    )

    fig_nutrition_radar = px.line_polar(
        filtered_data.melt(id_vars=['Product Code'], value_vars=numeric_columns),
        r='value',
        theta='variable',
        color='Product Code',
        line_close=True,
        title='Nutritional Composition Radar Chart for Selected Products'
    )

    return fig_nutriscore_bar, fig_ecoscore_pie, fig_nutrition_radar

# Callback to predict Nutriscore for new product
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('input-sugars', 'value'),
        State('input-proteins', 'value'),
        State('input-fat', 'value'),
        State('input-saturated-fat', 'value'),
        State('input-salt', 'value'),
        State('input-energy', 'value')
    ]
)
def predict_nutriscore(n_clicks, sugars, proteins, fat, saturated_fat, salt, energy):
    if n_clicks is not None:
        # Create a DataFrame for the input features
        input_data = pd.DataFrame({
            'Sugars (g)': [sugars],
            'Proteins (g)': [proteins],
            'Fat (g)': [fat],
            'Saturated Fat (g)': [saturated_fat],
            'Salt (g)': [salt],
            'Energy (kcal)': [energy]
        })
        
        # Make the prediction
        prediction = model.predict(input_data)[0]
        
        return f"The predicted Nutriscore is: {prediction}"
    return ""

# Initialize OpenAI client
api_key = '' # Add your OpenAI API key here
client = OpenAI(api_key=api_key)

@app.callback(
    Output('ai-output', 'children'),
    [Input('ai-button', 'n_clicks')],
    [State('ai-input', 'value'),
     State('product-dropdown', 'value')]
)
def get_ai_insights(n_clicks, query, selected_products):
    if n_clicks is not None and query:
        product_info = ""

        if selected_products:
            product_data = beverages_df[beverages_df['Product Code'].isin(selected_products)]
            product_info = product_data.to_dict('records')

        full_query = f"{query}\nProduct data: {product_info}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_query}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip() # type: ignore
    return ""

# Callback to update the radar chart based on selected Nutriscore/Ecoscore grade
@app.callback(
    Output('nutritional-composition-radar', 'figure'),
    Input('nutriscore-bar-chart', 'clickData'),
    Input('ecoscore-bar-chart', 'clickData')
)
def update_nutritional_composition_radar(nutriscore_click, ecoscore_click):
    filtered_data = beverages_df
    if nutriscore_click:
        grade = nutriscore_click['points'][0]['x']
        filtered_data = beverages_df[beverages_df['Nutriscore Grade'] == grade]
    elif ecoscore_click:
        grade = ecoscore_click['points'][0]['x']
        filtered_data = beverages_df[beverages_df['Ecoscore Grade'] == grade]
    
    avg_nutritional_composition = filtered_data[numeric_columns].mean()
    fig_radar = px.line_polar(
        avg_nutritional_composition.reset_index(),
        r=avg_nutritional_composition.values,
        theta=avg_nutritional_composition.index,
        line_close=True,
        title='Average Nutritional Composition'
    )
    return fig_radar

# Run the app with Flask development server
if __name__ == '__main__':
    app.run_server(debug=True)
