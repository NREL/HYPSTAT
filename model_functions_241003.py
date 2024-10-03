# Author: Steven Percy, updated by Yijin Li, Joe Brauch
# Date: 08/24/2022

import glob
import pandas as pd

#TODO: rename variables as needed

def get_gen_profiles(year, techs, path, drop_capacity_below=False):
    files = glob.glob(path + "/*")
    
    all_renewable_profiles = []
    all_capacities = []

    for tech in techs:
        #tech = tech.replace('_', ' ')
        tech_profiles = []
        tech_files = [f for f in files if tech in f and str(year) in f]
        zones_from_files = []

        #TODO: potentially loop through explicit file format instead of reading available files??

        for f in tech_files:
            
            zone = f.split('_Node_')[1].split('_')[0] #TODO: Either set formatting constraints or make flexible for user to input
            profile = pd.read_csv(f, index_col=0)

            if len(profile) > 0:
                zones_from_files.append(zone)
                profile.columns = [int(i) for i in profile.columns.astype(float)]

                if drop_capacity_below:
                    profile = profile.loc[:, profile.columns > drop_capacity_below]

                tech_profiles.append(profile)

        tech_profiles = pd.concat(tech_profiles, axis=1, keys=zones_from_files)
        capacities = get_capacity_frame(tech_profiles)
        all_capacities.append(capacities)
        tech_profiles = index_profiles(tech_profiles)
        all_renewable_profiles.append(tech_profiles)

    all_renewable_profiles = pd.concat(all_renewable_profiles, axis=1, keys=techs)
    all_capacities = pd.concat(all_capacities, axis=1, keys=techs)
    all_renewable_profiles.index = pd.to_datetime(str(year), yearfirst=True) + pd.to_timedelta(
        all_renewable_profiles.index, unit='h')

    return all_renewable_profiles, all_capacities

def index_profiles(all_renewable_profiles):
    all_renewable_profiles = all_renewable_profiles.T.reset_index()
    new_index = all_renewable_profiles.groupby('level_0').cumcount()
    all_renewable_profiles.level_1 = new_index
    all_renewable_profiles = all_renewable_profiles.set_index(['level_0', 'level_1']).T
    return all_renewable_profiles

def get_capacity_frame(all_renewable_profiles):
    capacities = pd.DataFrame([list(i) for i in (all_renewable_profiles.columns)], columns=['zone', 'capacity'])
    capacities = capacities[['zone', 'capacity']]
    capacities.index = capacities.groupby('zone').cumcount()
    capacities = capacities.pivot(columns='zone')
    capacities = capacities.droplevel(0, axis=1)
    return capacities


def get_demand(all_renewable_profiles, year,freq_in, freq_out, file_path):
    # NOTE: For now assumes daily demand!!
    # Read demand data
    demand = pd.read_csv(file_path,index_col=['Node','Period'])  
    
    # Reshape demand data
    demand = demand['h2_demand[kg/hr]'].unstack(level='Node')
    demand.index = pd.to_datetime(demand.index.str.replace(freq_in, '').astype(int),origin='12/31/{}'.format(year-1),unit=freq_in)
    demand.sort_index(inplace=True)

    #calculate the number of output periods for each input period
    periods_in_day = all_renewable_profiles.resample(freq_out).first().resample(freq_in).count().iloc[:, 0]
    
    #Adjust demand for daily periods
    demand = pd.concat([demand[d].values / periods_in_day for d in demand.columns], axis=1, keys=demand.columns)
    demand = demand.reindex(all_renewable_profiles.resample(freq_out).first().index).ffill()

    return demand


def get_links(path):
    links = pd.read_csv(path).set_index('Link')
    
    
    flow_direction = get_link_flow_direction(links.index, separator=' to ')
    
    links_to_zones = pd.concat([links['End node'], links['Start node']]).reset_index().set_index(0)
    links_to_zones = dict(links_to_zones.Link.groupby(links_to_zones.index).apply(set))

    #Fill NaNs in links to avoid issues in model solving
    links.fillna(value=0, inplace=True)
    
    return links, flow_direction, links_to_zones


def get_link_flow_direction(links, separator=' to '):
    link_flow_direction = pd.DataFrame(index=links, columns=[], data=0)

    for link in links:
        fn, tn = link.split(separator)
        link_flow_direction.loc[link, fn] = -1
        link_flow_direction.loc[link, tn] = 1

    return link_flow_direction.fillna(0)



def get_build_cost_matrix(financial_data, RE_costs, H2_prod_cost, H2_storage_cost, year, all_zones):
    '''
    This function returns a matrix with the annualized build cost for each node
    '''
    
    build_cost = pd.DataFrame()

    for zone in all_zones:
        for tech in RE_costs.index.get_level_values('Tech').unique():
            capex = RE_costs.loc[(year,zone,tech), 'CAPEX($/kW)']
            opex = RE_costs.loc[(year,zone,tech), 'Fix OPEX($/kW-yr)']
            recovery_time = financial_data.loc[(year,tech), 'Recovery_time (years)']
            wacc_nominal = financial_data.loc[(year,tech), 'WACC']
            interest = wacc_nominal
            build_cost.loc[tech, zone] = get_annuity(capex, interest, recovery_time) + opex

        for tech in H2_prod_cost.index.get_level_values('Tech').unique():
            capex = H2_prod_cost.loc[(year,zone,tech), 'CAPEX($/kW)']
            opex = H2_prod_cost.loc[(year,zone,tech), 'OPEX($/kW-yr)']
            recovery_time = financial_data.loc[(year,tech), 'Recovery_time (years)']
            wacc_nominal = financial_data.loc[(year,tech), 'WACC']
            interest = wacc_nominal
            build_cost.loc[tech, zone] = get_annuity(capex, interest, recovery_time) + opex

        for tech in H2_storage_cost.index.get_level_values('Tech').unique():
            capex = H2_storage_cost.loc[(year,zone,tech), 'CAPEX ($/kg)']
            opex = H2_storage_cost.loc[(year,zone,tech), 'Fixed OPEX ($/kg-yr)']
            recovery_time = financial_data.loc[(year,tech), 'Recovery_time (years)']
            wacc_nominal = financial_data.loc[(year,tech), 'WACC']
            interest = wacc_nominal
            build_cost.loc[tech, zone] = get_annuity(capex, interest, recovery_time) + opex

    return build_cost#.set_index('Tech',drop=True)


def get_annuity(capex, interest, years):
    '''
    The function provides the annual repayment formula to calculate the annual payment amount for your loan. 
    '''
    #if interest>1:
    #    interest = interest / 100  # conversion from %
    an = capex  * (interest * (1 + interest) ** years) /((1 + interest) ** years - 1)
    
    return an

def get_producers(capacities,zone='A',tech = 'Terrestrial_Wind'):
    '''
    This function allows for the creation of a list of the max capacity of all profile for a node
    It handles the case where a zone can't have a particular technology by returning an empty series
    '''
    if zone in capacities[tech].columns:
        max_capacity=capacities[(tech,zone)]
    else:
        #return empty series if node does not exist in 'capacities'
        max_capacity=pd.Series(dtype='float64')
    return max_capacity.dropna()

def get_producers_tech(tech_capacities,zone='A'):
    '''
    This function allows for the creation of a list of the max capacity of specific technology for a node
    It handles the case where a zone can't have a particular technology by returning an empty series
    '''
    if zone in tech_capacities.columns:
        max_capacity=tech_capacities[(zone)]
    else:
        #return empty series if node does not exist in 'capacities'
        max_capacity=pd.Series(dtype='float64')
    return max_capacity.dropna()