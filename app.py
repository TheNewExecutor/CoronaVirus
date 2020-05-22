import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import State, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from helper_functions import create_per_capita_features, load_raw_global, load_local, \
    load_merged_daily_global, load_merged_daily_local, sieve
from typing import List, Iterable, Tuple

# Load all data
@appcallback()
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
    world_data.loc[world_data.Location=='Georgia'].Location = 'Georgia (country)'
    return world_data, state_data, county_data


world_data, state_data, county_data = load_data()

def create_plot(content: str='Confirmed', countries :List[str], states: List[str]=[], counties: List[Tuple[str]]=[],
                log_y: bool=False, per_capita: bool=False):
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
    y_axis : string {'linear', 'log'}
        y axis scale
    per_capita : string {'yes', 'no'}
        Option to normalize data by population

    Returns
    -----------
    fig : object
        The plot output
    """
    data['countries'] = sieve(world_data, 'Location', countries)
    data['states'] = sieve(state_data, 'Location',states)
    data['counties'] = sieve(county_data, 'Location', counties)
    filtered_data = pd.concat(data.values(), axis=1)

    if per_capita:
        content = ' '.join([content, 'per capita'])
    fig = px.line(filtered_data, x='Date', y=content, log_y=log_y,
                      color='Location', hover_name='Location', title=content)

    return fig

def create_y_radio():
    return dcc.RadioItems(
        options=[
            {'label': 'log', 'value': True},
            {'label': 'linear', 'value': False}
        ]
    )

def create_content_radio():
    return dcc.RadioItems(
        options=[{'label': val, 'value': val}
                for val in ['Confirmed', 'Deaths', 'Recovered']]
    )

def create_per_capita_element():
    return dcc.RadioItems(
        options=[{'label': 'Total', 'value': False},
                 {'label': 'per capita', 'value': True}
                ]
    )

def create_county_select():
    pass

def create_state_select():
    pass

def create_country_select():
    pass



@appcallback(
    [Output('plot1', 'figure')],
    [Input('y-axis', 'value'),
     Input('content-radio', 'value'),
     Input('per-capita-radio', 'value'),
     Input('countries-multiselect', 'value'),
     Input('states-multiselect', 'value'),
     Input('counties-multiselect', 'value')]
)
def update_plot(content, countries, states, counties, log_y, per_capita):
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
    y_axis : string {'linear', 'log'}
        y axis scale
    per_capita : string {'yes', 'no'}
        Option to normalize data by population

    Returns
    -------
    fig: object
        The plot with updated
    """
    return create_plot(content, countries, states, counties, log_y, per_capita)


external_stylesheets = ['https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout(html.Div(
    className='row',
    children=[
    html.Div(id='row1',
             className='row',
             ),
    html.Div(id='row2',
             className='row'),
    html.Div(id='row3',
             className='row'),
    html.Div()
        ]
    )



)
