# Author: Steven Percy, updated by Yijin Li, Joe Brauch
# Date: 08/24/2022

import glob
import pandas as pd


def get_renewable_profiles(year, techs, path, drop_capacity_below=False):
    files = glob.glob(path + "/*")
    
    all_renewable_profiles = []
    all_capacities = []

    for tech in techs:
        tech = tech.replace('_', ' ')
        tech_profiles = []
        tech_files = [f for f in files if tech in f and str(year) in f]
        zones_from_files = []

        for f in tech_files:
            
            zone = f.split('_Node_')[1].split('_')[0]
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


def get_demand(all_renewable_profiles, year, daily_demand, freq, file_path):
    # Read demand data
    demand = pd.read_csv(file_path)
    demand = demand.loc[demand['Year'] == year]
    

    # Calculate total demand
    demand['total_demand'] = demand[['Fueling-Station Demand [kg/yr]', 'Non-Fueling-Station Demand [kg/yr]']].sum(axis=1)

    # Reshape demand data
    demand = demand.set_index(['Period', 'Network ID'])['total_demand'].unstack().sort_index()
    demand.index = demand.index.str.replace('P', '').astype(int)
    demand = demand.sort_index()

    # Adjust for daily demand
    if daily_demand:
        #demand.index = demand.index.astype(str) + '-' + str(year)
        #demand.index = pd.to_datetime(demand.index, dayfirst=True, format='%d-%Y')

        #Calculate periods in a month
        periods_in_day = all_renewable_profiles.resample(freq).first().resample('D').count().iloc[:, 0]
        
        # Adjust demand for daily periods
        demand = pd.concat([demand[d].values / periods_in_day for d in demand.columns], axis=1, keys=demand.columns)
        demand = demand.reindex(all_renewable_profiles.resample(freq).first().index).ffill()
        #demand.index = pd.to_datetime(demand.index.astype(str) + '-' + str(year), dayfirst=True)

    return demand


def get_links(path):
    links = pd.read_csv(path).set_index('Link')
    #links = links[links['Pipeline allowed']=='Y']
    
    all_zones = links[['End node', 'Start node']].stack().unique()
    #links = links.loc[~links['Delivery Method'].isna()]
    
    flow_direction = get_link_flow_direction(links.index, separator=' to ')
    flow_direction = flow_direction.reindex(all_zones, axis=1).fillna(0)
    
    links_to_zones = pd.concat([links['End node'], links['Start node']]).reset_index().set_index(0)
    links_to_zones = dict(links_to_zones.Link.groupby(links_to_zones.index).apply(set))
    #links['Tech'] = 'Pipeline'
    #link_distances = pd.concat([links['Tech'],links['Pipeline distance [km]']],axis=1)
    link_distances = links['Pipeline distance [km]']
    max_cap = (links['Pipeline max capacity [kg/hr]']).unique()
    
    return flow_direction, links, all_zones, links_to_zones

def get_truck_links(path):

    links = pd.read_csv(path).set_index('Link')
    links = links[links['Truck allowed']=='Y']

    all_zones = links[['End node', 'Start node']].stack().unique()
    

    flow_direction = get_link_flow_direction(links.index, separator=' to ')
    flow_direction = flow_direction.reindex(all_zones, axis=1).fillna(0)

    links_to_zones = pd.concat([links['End node'], links['Start node']]).reset_index().set_index(0)
    links_to_zones = dict(links_to_zones.Link.groupby(links_to_zones.index).apply(set))

    links['Tech'] = 'Truck'
    link_distances = pd.concat([links['Tech'],links['Truck distance [km]']],axis=1)
    print(link_distances)
    max_cap = (links['Truck max capacity [kg/hr]']).unique()

    return flow_direction, links, all_zones, links_to_zones, link_distances, max_cap

def get_all_links(path):
    
    links = pd.read_csv(path).set_index('Link')
    
    all_zones = links[['End node', 'Start node']].stack().unique()
    
    flow_direction = get_link_flow_direction(links.index, separator=' to ')
    flow_direction = flow_direction.reindex(all_zones, axis=1).fillna(0)
    
    links_to_zones = pd.concat([links['End node'], links['Start node']]).reset_index().set_index(0)
    links_to_zones = dict(links_to_zones.Link.groupby(links_to_zones.index).apply(set))

    
    return flow_direction, links, all_zones, links_to_zones

def get_link_flow_direction(links, separator=' to '):
    link_flow_direction = pd.DataFrame(index=links, columns=[], data=0)

    for link in links:
        fn, tn = link.split(separator)
        link_flow_direction.loc[link, fn] = -1
        link_flow_direction.loc[link, tn] = 1

    return link_flow_direction.fillna(0)



def get_build_cost_matrix(FinancialFiles_paths, REcostFiles_paths, ProductioncostFiles_paths, H2StorageFile_paths, year, all_zones):
    '''
    This function returns a matrix with the annualized build cost for each node
    '''

    def read_file(path):

        file = pd.read_csv(path[0])
        file = file[file['Year']==year]
        file.drop(columns=['Year'], inplace=True)

        return file
    
    financials = read_file(FinancialFiles_paths)

    RE_technology_build = read_file(REcostFiles_paths)
    multiply_numeric = lambda x: x * 1.1 if pd.api.types.is_numeric_dtype(x) else x # convert to euros

    RE_technology_build = RE_technology_build.applymap(multiply_numeric) 

    H2_technology_build = read_file(ProductioncostFiles_paths)
    H2_technology_build = H2_technology_build.applymap(multiply_numeric)

    H2_storage_build = read_file(H2StorageFile_paths)
    

    RE_technology_build = RE_technology_build.merge(financials, how='left', left_on='Tech', right_on='Tech').set_index('Tech',drop=True)
    H2_technology_build = H2_technology_build.merge(financials, how='left', left_on='Tech', right_on='Tech').set_index('Tech',drop=True)
    H2_storage_build = H2_storage_build.merge(financials, how='left', left_on='Tech', right_on='Tech').set_index('Tech',drop=True)

    tech_cost_build = pd.concat([RE_technology_build,H2_technology_build,H2_storage_build])

    build_cost = pd.DataFrame()
    #for tech in tech_cost_build.index: #tech_cost_build['Tech'].unique():
    for zone in all_zones:
        for tech in RE_technology_build.index:
            capex = RE_technology_build.loc[tech, 'CAPEX($/kW)']
            opex = RE_technology_build.loc[tech, 'Fix OPEX($/kW-yr)']
            recovery_time = RE_technology_build.loc[tech, 'Recovery_time (years)']
            wacc_nominal = float(RE_technology_build.loc[tech, 'WACC_Nominal'])
            interest = wacc_nominal
            build_cost.loc[tech, zone] = get_annuity(capex, interest, recovery_time) + opex

        for tech in H2_technology_build.index:
            capex = H2_technology_build.loc[tech, 'CAPEX($/kW)']
            opex = H2_technology_build.loc[tech, 'OPEX($/kW-yr)']
            recovery_time = H2_technology_build.loc[tech, 'Recovery_time (years)']
            wacc_nominal = float(H2_technology_build.loc[tech, 'WACC_Nominal'])
            interest = wacc_nominal
            build_cost.loc[tech, zone] = get_annuity(capex, interest, recovery_time) + opex

        for tech in H2_storage_build.index:
            capex = H2_storage_build.loc[tech, 'CAPEX (€/kg)']
            opex = H2_storage_build.loc[tech, 'OPEX (€/kg)']
            recovery_time = H2_storage_build.loc[tech, 'Recovery_time (years)']
            wacc_nominal = float(H2_storage_build.loc[tech, 'WACC_Nominal']) 
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