# Author: Steven Percy, updated by Yijin Li, Joe Brauch
# Date: 08/24/2022

import glob
import pandas as pd

def expand_input_df(df,to_expand,expand_options):
    #NOTE: in progress, only works for nodes where all years and techs are specified. DO NOT USE FOR OTHERS.
    '''
    This function expands a specified category of the input dataframe to all options in that category where inputs apply to 'all'.
    For any row of inputs applying to 'All' in the specified category, this function creates rows using those inputs for every
    option in that category. These inputs are overridden by any specified options in the input dataframe.

    Parameters:
        df: the input dataframe
        to_expand: the category to be expanded
        expand_options: list of all options for that category

    Returns:
        new, expanded dataframe
    '''
    new_df = pd.DataFrame()
    for op in expand_options:
        op_df = df.xs('All',level=to_expand,drop_level=False)
        op_df.index = op_df.index.remove_unused_levels().set_levels([op],level=to_expand)
        new_df = pd.concat([new_df,op_df])
    
    new_df.loc[df.drop(index='All',level='Node').index] = None
    return new_df.combine_first(df.drop(index='All',level='Node')).sort_index()


def get_gen_profiles(year, gen_techs, path, drop_capacity_below=False):
    '''
    Builds dataframes of generation profiles and capacity limits for each tech, node, and tranche (to be used as model inputs)
    '''
    files = glob.glob(path + "/*")
    
    gen_profiles = []
    capacity_limits = []

    for tech in gen_techs:
        #tech = tech.replace('_', ' ')
        tech_profiles = []
        tech_files = [f for f in files if tech in f and str(year) in f]
        nodes_from_files = []

        for f in tech_files:
            
            node = f.split('_Node_')[1].split('_')[0] #TODO: Either set formatting constraints or make flexible for user to input
            profile = pd.read_csv(f, index_col=0)

            if len(profile) > 0:
                nodes_from_files.append(node)
                profile.columns = [int(i) for i in profile.columns.astype(float)]

                if drop_capacity_below:
                    profile = profile.loc[:, profile.columns > drop_capacity_below]

                tech_profiles.append(profile)

        tech_profiles = pd.concat(tech_profiles, axis=1, keys=nodes_from_files)
        capacities = get_capacity_limits(tech_profiles)
        capacity_limits.append(capacities)
        tech_profiles = index_profiles(tech_profiles)
        gen_profiles.append(tech_profiles)

    gen_profiles_df = pd.concat(gen_profiles, axis=1, keys=gen_techs)
    capacity_limits_df = pd.concat(capacity_limits, axis=1, keys=gen_techs)
    gen_profiles_df.index = pd.to_datetime(str(year), yearfirst=True) + pd.to_timedelta(
        gen_profiles_df.index, unit='h')

    return gen_profiles_df, capacity_limits_df


def index_profiles(gen_profiles):
    '''
    Re-indexes columns for generation profiles to use numerical identifiers for tranches which can be referenced in the model
    '''
    gen_profiles = gen_profiles.T.reset_index()
    new_index = gen_profiles.groupby('level_0').cumcount()
    gen_profiles.level_1 = new_index
    gen_profiles = gen_profiles.set_index(['level_0', 'level_1']).T
    return gen_profiles


def get_capacity_limits(gen_profiles):
    '''
    Builds a table of capacity limits by tranche for each node from generation profile inputs
    '''
    capacities = pd.DataFrame([list(i) for i in (gen_profiles.columns)], columns=['node', 'capacity'])
    capacities = capacities[['node', 'capacity']]
    capacities.index = capacities.groupby('node').cumcount()
    capacities = capacities.pivot(columns='node')
    capacities = capacities.droplevel(0, axis=1)
    return capacities


def get_demand(time_periods, year,freq_in, freq_out, file_path):
    '''
    Processes demand input files into demand dataframe of desired frequncy that can be read by the model
    '''
    # NOTE: For now assumes daily demand!!
    # Read demand data
    demand = pd.read_csv(file_path,index_col=['Node','Period'])  
    
    # Reshape demand data
    demand = demand['h2_demand[kg/hr]'].unstack(level='Node')
    demand.index = pd.to_datetime(demand.index.str.replace(freq_in, '').astype(int),origin='12/31/{}'.format(year-1),unit=freq_in)
    demand.sort_index(inplace=True)

    #calculate the number of output periods for each input period
    periods_in_day = pd.Series(1,index=time_periods).resample(freq_out).first().resample(freq_in).count()
    
    #Adjust demand for daily periods
    demand = pd.concat([demand[d].values / periods_in_day for d in demand.columns], axis=1, keys=demand.columns)
    demand = demand.reindex(time_periods).ffill()

    return demand


def get_links(path):
    '''
    Builds tables for links, mapping of nodes to connected links, and link flow directions for model input
    '''
    links = pd.read_csv(path).set_index('Link')
    
    flow_direction = get_link_flow_direction(links)
    
    #build mapping of nodes to connect links, e.g. 'A': ['A to B', 'A to C', 'D to 'A', etc.]
    links_to_nodes = pd.concat([links['End node'], links['Start node']]).reset_index().set_index(0)
    links_to_nodes = dict(links_to_nodes.Link.groupby(links_to_nodes.index).apply(set))

    #Fill NaNs in links to avoid issues in model solving
    links.fillna(value=0, inplace=True)
    
    return links, flow_direction, links_to_nodes

def get_link_flow_direction(links):
    link_flow_direction = pd.DataFrame(index=links.index, columns=[], data=0)
    for link in link_flow_direction.index:
        link_flow_direction.loc[link,links.loc[link,'Start node']] = -1
        link_flow_direction.loc[link,links.loc[link,'End node']] = 1
    
    return link_flow_direction.fillna(0)

def get_build_cost_matrix(financial_data, gen_cost, prod_cost, stor_cost, year, nodes):
    '''
    This function returns a matrix with the annualized build cost of different technologies for each node (CAPEX + fixed OPEX)
    Note: this will build a table including all technologies specified in input files, even if those technologies are not specified
          as allowed technologies for a model run. (This is OK, the disallowed technologies will simply not be referenced in the
          model run.)
    '''
    build_cost = pd.DataFrame()

    for node in nodes:
        for tech in gen_cost.index.get_level_values('Tech').unique():
            capex = gen_cost.loc[(year,node,tech), 'CAPEX($/kW)']
            opex = gen_cost.loc[(year,node,tech), 'Fixed OPEX($/kW-yr)']
            recovery_time = financial_data.loc[(year,tech), 'Recovery_time (years)']
            wacc = financial_data.loc[(year,tech), 'WACC']
            interest = wacc
            build_cost.loc[tech, node] = get_annuity(capex, interest, recovery_time) + opex

        for tech in prod_cost.index.get_level_values('Tech').unique():
            capex = prod_cost.loc[(year,node,tech), 'CAPEX($/kW)']
            opex = prod_cost.loc[(year,node,tech), 'Fixed OPEX($/kW-yr)']
            recovery_time = financial_data.loc[(year,tech), 'Recovery_time (years)']
            wacc = financial_data.loc[(year,tech), 'WACC']
            interest = wacc
            build_cost.loc[tech, node] = get_annuity(capex, interest, recovery_time) + opex

        for tech in stor_cost.index.get_level_values('Tech').unique():
            capex = stor_cost.loc[(year,node,tech), 'CAPEX ($/kg)']
            opex = stor_cost.loc[(year,node,tech), 'Fixed OPEX ($/kg-yr)']
            recovery_time = financial_data.loc[(year,tech), 'Recovery_time (years)']
            wacc = financial_data.loc[(year,tech), 'WACC']
            interest = wacc
            build_cost.loc[tech, node] = get_annuity(capex, interest, recovery_time) + opex

    return build_cost


def get_annuity(capex, interest, years):
    '''
    The function calculates the annuitized costs for an upfront payment given an interest rate and payback years.
    '''
    an = capex  * (interest * (1 + interest) ** years) /((1 + interest) ** years - 1)
    
    return an


def get_node_tech_limits(capacities,node='A',tech = 'Terrestrial_Wind'):
    '''
    Builds a series of capacity limits by tranche for the specified node and technology.
    Series includes ONLY the tranches for that technology at that zone, so that the model is not able to build outside of those tranches.
    Returns an empty series if the technology is not allowed at that node.
    '''
    if node in capacities[tech].columns:
        max_capacity=capacities[(tech,node)]
    else:
        #return empty series if node does not exist in 'capacities'
        max_capacity=pd.Series(dtype='float64')
    return max_capacity.dropna() #dropna to remove tranches that do not exist for that technology at that node