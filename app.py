import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from helper_functions import create_per_capita_features, load_raw_global, load_local, \
    load_merged_daily_global, load_merged_daily_local, sieve, load_tidy_global, top_n_locations
from typing import Iterable
import re

external_stylesheets = ['https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Load all data
def load_data():
    """Load all processed COVID-19 data"""

    data = load_raw_global()
    tidy_global = load_tidy_global(data)
    merged_global = load_merged_daily_global(tidy_global)

    # Load local data, and daily data dictionaries
    state, county = load_local().values()
    merged_state = load_merged_daily_local(state)
    merged_county = load_merged_daily_local(county)

    # Create per capita features on merged data sets
    datasets = [merged_global, merged_state.drop(columns='fips'), merged_county.drop(columns='fips')]
    files = ['pop_countries.csv', 'pop_states.csv', 'pop_counties.csv']
    populations = [pd.read_csv('Data/' + file, index_col=0) for file in files]
    world_data, state_data, county_data = [create_per_capita_features(data, pop)
                                           for data, pop in zip(datasets, populations)]
    world_data['Location'] = world_data['Country/Region']
    state_data['Location'] = state_data['State']
    county_data['Location'] = county_data['County'] + ', ' + county_data['State']
    world_data.loc[world_data.Location == 'Georgia', 'Location'] = 'Georgia (country)'
    return world_data, state_data, county_data


world_data, state_data, county_data = load_data()
top10_countries = top_n_locations(world_data, feature='Confirmed')


def create_plot(content: str = 'Confirmed', countries: Iterable = top10_countries,
                states: Iterable = [], counties: Iterable = [],
                log_y: str = 'linear', per_capita: str = 'total',
                daily: str = 'cumulative'):
    """

    Parameters
    ----------
    content : str {'Confirmed', 'Deaths', 'Recovered'}
        Data to plot
    countries : list of strings
        Countries to plot
    states : list of strings
        States to plot
    counties : list of strings
        Counties to plot
    log_y : string {'linear', 'log'}
        y axis scale
    per_capita : string {'total', 'per capita'}
        Option to normalize data by population
    daily : string {'Cumulative', 'Daily'}
        Option to show daily difference data or cumulative
    Returns
    -----------
    fig : object
        The plot output
    """
    data = {'countries': sieve(world_data, 'Location', countries),
            'states': sieve(state_data, 'Location', states),
            'counties': sieve(county_data, 'Location', counties)}
    filtered_data = pd.concat(data.values(), join='outer')

    if daily == 'Daily':
        content = ' '.join([daily, content])
    if per_capita != 'total':
        content = ' '.join([content, 'per capita'])

    y_axis = log_y != 'linear'
    fig = px.line(filtered_data, x='Date', y=content, log_y=y_axis,
                  color='Location', hover_name='Location', title=content)

    return fig


def create_top_n():
    return dcc.Dropdown(
        options=[{'label': 'Top Countries (Deaths)'}]
    )


def create_daily_radio():
    return dcc.RadioItems(
        options=[{'label': val, 'value': val}
                 for val in ['Cumulative', 'Daily']],
        value='Cumulative',
        id='daily-radio',
        labelStyle={
            "display": "inline-block",
            "padding": "12px 12px 12px 50px",
        }
    )


def create_y_radio():
    return dcc.RadioItems(
        options=[
            {'label': 'log', 'value': 'log'},
            {'label': 'linear', 'value': 'linear'}
        ],
        value='linear',
        id='y-axis',
        labelStyle={
            "display": "inline-block",
            "padding": "12px 12px 12px 0px",
        }
    )


def create_content_radio():
    return dcc.RadioItems(
        options=[{'label': val, 'value': val}
                 for val in ['Confirmed', 'Deaths', 'Recovered']],
        value='Confirmed',
        id='content-radio',
        labelStyle={
            "display": "inline-block",
            "padding": "12px 12px 12px 0px",
        }
    )


def create_per_capita_element():
    return dcc.RadioItems(
        options=[{'label': 'Total', 'value': 'total'},
                 {'label': 'per capita', 'value': 'per capita'}
                 ],
        id='per-capita-radio',
        value='total',
        labelStyle={
            "display": "inline-block",
            "padding": "12px 12px 12px 0px",
        }
    )


def create_location_select(data, id, values):
    return dcc.Dropdown(
        options=[{'label': location, 'value': location}
                 for location in data['Location'].unique()],
        multi=True,
        id=id,
        value=values
    )


def create_left_card():
    """Panel with plot setting elements"""
    card = dbc.Card(
        [dbc.CardBody(
            [html.H3("Plot Settings", className="alert-secondary"),
             html.Br(),
             html.Div(
                 [html.H6('Content', className='alert-secondary'),
                  dbc.Row([
                      dbc.Col(create_content_radio()),
                      dbc.Col(create_daily_radio())]
                  ),
                  html.H6('y-Axis Scaling', className='alert-secondary'),
                  create_y_radio(),
                  html.H6('Normalization', className='alert-secondary'),
                  create_per_capita_element(),
                  html.H6('Choose Countries:', className='alert-secondary'),
                  create_location_select(world_data, 'countries-multiselect', top10_countries),
                  html.H6('Choose US States and Territories:', className='alert-secondary'),
                  create_location_select(state_data, 'states-multiselect', []),
                  html.H6('Choose US Counties:', className='alert-secondary'),
                  create_location_select(county_data, 'counties-multiselect', [])
                  ]
             )
             ]

        ),
        ]
    ),
    return card


def create_right_card():
    """Panel with plot """
    card = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H3("COVID-19 Time Series Comparisons", className="alert-secondary"),
                    html.Div(
                        dcc.Graph(figure=create_plot(),
                                  id='plot1')
                    )

                ],
            ),
        ],
    )
    return card


app.layout = html.Div(
    className='bg-light',
    children=[
        html.Div(
            [html.H1("COVID-19 Dashboard",
                     id="title",
                     className="text-white",
                     style={"margin-left": "3%"},
                     ),
             ],
            className="jumbotron bg-dark",
        ),
        html.Div(
            dbc.Row(
                [dbc.Col(create_left_card(), width=4),
                 dbc.Col(create_right_card(), width=8)
                 ],
            )
        )
    ])


@app.callback(
    Output('plot1', 'figure'),
    [Input('content-radio', 'value'),
     Input('countries-multiselect', 'value'),
     Input('states-multiselect', 'value'),
     Input('counties-multiselect', 'value'),
     Input('y-axis', 'value'),
     Input('per-capita-radio', 'value'),
     Input('daily-radio', 'value')

     ]
)
def update_plot(content: str,
                countries: Iterable,
                states: Iterable,
                counties: Iterable,
                log_y: str,
                per_capita: str,
                daily: str
                ):
    """
    Update plot based on figure

    Parameters
    ----------
    content : str {'Confirmed', 'Deaths', 'Recovered'}
        Data to plot
    countries : list of strings
        Countries to plot
    states : list of strings
        States to plot
    counties : list of strings
        Counties to plot
    log_y : string {'linear', 'log'}
        y axis scale
    per_capita : string {'total', 'per capita'}
        Option to normalize data by population
    daily : string {'cumulative', 'daily'}
        Option to check changes

    Returns
    -------
    fig: object
        The plot with updated
    """
    try:
        return create_plot(content=content,
                           countries=countries,
                           states=states,
                           counties=counties,
                           log_y=log_y,
                           per_capita=per_capita,
                           daily=daily
                           )
    except:
        raise PreventUpdate


@app.callback(
    [Output('countries-multiselect', 'options'),
     Output('states-multiselect', 'options'),
     Output('counties-multiselect', 'options'),
     Output('countries-multiselect', 'placeholder'),
     Output('states-multiselect', 'placeholder'),
     Output('counties-multiselect', 'placeholder')],
    [Input('content-radio', 'value'),
     Input('per-capita-radio', 'value'),
     Input('daily-radio', 'value')]
)
def update_location_options(content: str, per_capita: str, daily: str):
    """
    Update sorted options for choose location dropdowns

    Parameters
    ----------
    content : " str {'Confirmed', 'Deaths', 'Recovered'}
        Content option values
    per_capita : str {'total', 'per capita'}
        Normalization option values
    daily : str {'Cumulative', 'Daily'}
        Difference option values

    Returns
    -------
    country_options : list(dicts)
        New sorted countries options based on input
    state_options : list(dicts)
        New sorted states options based on inputs
    county_options : list(dicts)

    """
    field = content
    if per_capita == 'per capita':
        field = ' '.join([content, per_capita])
    if daily == 'Daily':
        field = ' '.join([daily, field])
    local_field = re.sub(r'Recovered', 'Confirmed', field)

    option_values = {
        'countries': top_n_locations(world_data, field, 3000),
        'states': top_n_locations(state_data, local_field, 3000),
        'counties': top_n_locations(county_data, local_field, 3000)
    }

    placeholders = {'countries': f'Sorted by {field}',
                    'states': f'Sorted by {local_field}',
                    'counties': f'Sorted by {local_field}'}
    options = {}
    for level in option_values:
        options[level] = [{'label': location, 'value': location}
                          for location in option_values[level]]

    return (options['countries'], options['states'], options['counties'],
            placeholders['countries'], placeholders['states'], placeholders['counties'])


if __name__ == '__main__':
    app.run_server(debug=True, port=3004)
