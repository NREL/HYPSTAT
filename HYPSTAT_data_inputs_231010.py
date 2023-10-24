###################################################
# Data inputs (input processing)
###################################################

techs = ['Terrestrial_Wind','Offshore_Wind','Solar'] 

year = 2050 #dummy line

scenario='Mid_Demand'

if year == 2050:
    techs = ['Terrestrial_Wind','Offshore_Wind','Solar','Nuclear'] 


supply_path='/projects/h2nyserda/NYSERDA_Scenario_Tool/Temporal/NREL_data/Temporal_Supply_Files_with_Clustering_Limited_final_221118'
all_renewable_profiles,capacities=get_renewable_profiles(year=year,techs = techs, drop_capacity_below=False,path=supply_path)

demand=get_demand(all_renewable_profiles,year=year, daily_demand=True, scenario=scenario)

if h != 1: #TODO: think about time resolution, for now leave as minimum 1 hour
    all_renewable_profiles = all_renewable_profiles.resample('{}h'.format(h)).sum()
    demand = demand.resample('{}h'.format(h)).sum()

#path = 'Temporal/NREL_data/Network_configurations/'+str(year)+'_'+scenario+'_links.csv'
path = '/projects/h2nyserda/NYSERDA_Scenario_Tool/NYSERDA_Scenario_Tool/Temporal/NREL_data/Network_configurations/'+str(year)+'_'+scenario+'_links.csv'
link_flow_direction, links, all_zones,links_to_zones=get_links(path,scenario=scenario,year=year,unconstrained=True)

#add column in links for operating cost

#right now, set all to $0.06/kg

comp_cost = 0.06 #$/kg/100-mi of transmission
truck_cost = 1.25 #$/kg/100-mi of truck transmission
truck_size_limit = 100*1000/24 #kg/hr
#will need to update below filepath to run on HPC!
link_distances = pd.read_csv('/projects/h2nyserda/NYSERDA_Scenario_Tool/Temporal/NREL_data/link_distances.csv',index_col='Zone')
#link_distances = pd.read_csv('Temporal/NREL_data/link_distances.csv',index_col='Zone')

link_opex = [comp_cost*(link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]/100) for link in links.index]
#add in $1.25/kg per 100-mi
links.insert(4,'Transmission Opex ($/kg)',link_opex)

""" 
for link in links.index:
    if links.loc[link,'Delivery Method']=='Truck':
        links.loc[link,'Capacity (kg/hr)'] = 100*1000/24
        links.loc[link,'Transmission Opex ($/kg)'] = truck_cost*(link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]/100) """

for zone in all_zones:
    if zone not in links_to_zones:
        links_to_zones[zone] = set()

'''links['Capacity (kg/hr)'] *= 2 #testing increasing link capacity to avoid unserved hydrogen
links.loc['HI to J','Capacity (kg/hr)'] *= 6 ''' #don't need this block for unconstrained network
h2_conversion_efficiency=51
build_cost=get_build_cost_matrix(year=year,file='Build_Cost_Inputs_elec_cost_conservative.csv',all_zones=all_zones,includ_interconnection_cost = True)
#build_cost.loc['Tank_Storage','NJ1'] *= 1.1 old, from testing

PTC = dict()
techs = ['Terrestrial_Wind','Offshore_Wind','Solar'] 
for tech in techs:
    if tech == 'Offshore_Wind':
        PTC[tech] = 0 #0.04
    else:
        PTC[tech] = 0.024

ITC = {
    'Terrestrial_Wind': 0,
    'Offshore_Wind': 0.04,
    'Solar': 0
}

cavern_capacities = dict()
for zone in all_zones:
    cavern_capacities[zone] = 0

tank_capacities = dict()
for zone in all_zones:
    tank_capacities[zone] = 'inf'

#disallow tank storage in HI and J
tank_capacities['HI'] = 0
tank_capacities['J'] = 0

#cavern_capacities['CS'] = 'inf'
cavern_capacities['CS'] = 8000000
#cavern_capacities['A'] = 'inf'
Total_cavern_capacity = 'inf'
#Total_cavern_capacity = 16000000

## nuclear hydrogen inputs:
include_nuclear_hydrogen=False
nuclear_hydrogen_years=[2050]
nuclear_zones=['CN']
#capacity_factor=0.890507
maximum_nuclear_capacity=1255.8 * 1000 #KW
Nuclear_LCOE=30 #$/MWh
Nuclear_LCOE=Nuclear_LCOE/1000 #$/kWh
nuclear_h2_conversion_efficiency = 35.7

#import controls
max_imports = demand.sum().sum()/2
import_zones = {'CS'} #zones which can import hydrogen

overserved_cost = 5 #$/kg