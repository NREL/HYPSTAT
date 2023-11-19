# Author: Steven Percy, updated by Yijin Li, Joe Brauch
# Date: 08/24/2022

import glob
import pandas as pd


def get_renewable_profiles(year, techs, path, drop_capacity_below=False):
    files = glob.glob(path + "/*")
    all_renewable_profiles = []
    all_capacities = []

    for tech in techs:
        #tech = tech.replace('_', ' ')
        tech_profiles = []
        tech_files = [f for f in files if tech in f and str(year) in f]
        zones_from_files = []

        for f in tech_files:
            zone = f.split('_Zone')[1].split('_')[0]
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
    demand = pd.read_csv(file_path, sep='\t')
    demand = demand.loc[demand['Year'] == year]

    # Calculate total demand
    demand['total_demand'] = demand[['Fueling-Station Demand [kg]', 'Non-Fueling-Station Demand [kg]']].sum(axis=1)

    # Reshape demand data
    demand = demand.set_index(['Period', 'Network ID'])['total_demand'].unstack().sort_index()
    demand.index = demand.index.str.replace('M', '').astype(int)
    demand = demand.sort_index()

    # Adjust for daily demand
    if daily_demand:
        demand.index = '1-' + demand.index.astype(str) + '-' + str(year)
        demand.index = pd.to_datetime(demand.index, dayfirst=True)

        # Calculate periods in a month
        periods_in_month = all_renewable_profiles.resample(freq).first().resample('MS').count().iloc[:, 0]

        # Adjust demand for daily periods
        demand = pd.concat([demand[d] / periods_in_month for d in demand.columns], axis=1, keys=demand.columns)
        demand = demand.reindex(all_renewable_profiles.resample(freq).first().index).ffill()

    return demand

def get_links(path, unconstrained=False):
    if unconstrained:
        links = pd.read_csv('../Test case data/unlimited_links.csv').set_index('Link')
    else:
        links = pd.read_csv(path).set_index('Link')

    all_zones = links[['End zone', 'Start zone']].stack().unique()
    links = links.loc[~links['Delivery Method'].isna()]

    flow_direction = get_link_flow_direction(links.index, separator=' to ')
    flow_direction = flow_direction.reindex(all_zones, axis=1).fillna(0)

    links_to_zones = pd.concat([links['End zone'], links['Start zone']]).reset_index().set_index(0)
    links_to_zones = dict(links_to_zones.Link.groupby(links_to_zones.index).apply(set))

    return flow_direction, links, all_zones, links_to_zones

def get_link_flow_direction(links, separator=' to '):
    link_flow_direction = pd.DataFrame(index=links, columns=[], data=0)

    for link in links:
        fn, tn = link.split(separator)
        link_flow_direction.loc[link, fn] = -1
        link_flow_direction.loc[link, tn] = 1

    return link_flow_direction.fillna(0)

