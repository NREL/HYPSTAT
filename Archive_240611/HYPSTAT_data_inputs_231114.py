###################################################
# Data inputs (input processing)
###################################################

import pandas as pd
import yaml
from Model_Functions_v2_111423 import *


# Specify the path to YAML file
yaml_file_path = 'HYPSTAT_scenario.yaml'

# Read the YAML file
with open(yaml_file_path, 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)
 
# Extract yaml data into variables
year = yaml_data.get('studiedyear',None)
firstYear = yaml_data.get('firstYear', None)
lastYear = yaml_data.get('lastYear', None)
timeWindow = yaml_data.get('timeWindow', None)
scenario = yaml_data.get('scenario', None)
truck_size_limit = yaml_data.get('truck_size_limit', None)

interest = yaml_data.get('interest',None)
payback_years = yaml_data.get('payback_years',None)
payback_years_tank = yaml_data.get('payback_years_tank', None)
payback_years_electrolyzer = yaml_data.get('payback_years_electrolyzer',None)

max_imports_ratio = yaml_data.get('max_imports_ratio',None)
import_zones = yaml_data.get('import_zones',None)
overserved_cost = yaml_data.get('hydrogen_overserved_cost',None)
Storage_charge_limit =yaml_data.get('Storage_charge_limit', None)
Storage_discharge_limit =yaml_data.get('Storage_discharge_limit', None)
Storage_charge_cost =yaml_data.get('Storage_charge_cost', None)
Total_storage_capacity =yaml_data.get('Total_storage_capacity', None)

start = yaml_data.get('start', None)
end = yaml_data.get('end', None) 


# Files
demandFiles_paths = yaml_data.get('demandFile', [])
REcostFiles_paths = yaml_data.get('REcostFile', [])
ProductioncostFiles_paths = yaml_data.get('ProductioncostFile', [])
H2DeliveryFile_paths = yaml_data.get('H2DeliveryFile', [])
H2StorageFile_paths = yaml_data.get('H2StorageFile', [])
SupplycurveFolder_paths = yaml_data.get('SupplycurveFolder', [])
Networks_paths = yaml_data.get('NetworksFiles', [])
IncentiveFiles_paths = yaml_data.get('IncentiveFiles', [])
StorageCapacityFiles_paths = yaml_data.get('StorageCapacityFiles', [])


# create the year and tech dictionary:
REcostFiles = pd.concat([pd.read_csv(path) for path in REcostFiles_paths], ignore_index=True)
year_tech_dict = {}
for year, group in REcostFiles.groupby('Year'):
    year_tech_dict[year] = list(group['Tech'])
techs = year_tech_dict[year]

H2StorageFile = pd.concat([pd.read_csv(path) for path in H2StorageFile_paths], ignore_index=True)
year_stor_tech_dict = {}
for year, group in H2StorageFile.groupby('Year'):
    year_tech_dict[year] = list(group['Tech'])
stor_techs = year_stor_tech_dict[year]


REcostFiles = pd.concat([pd.read_csv(path) for path in REcostFiles_paths], ignore_index=True)
year_tech_dict = {}
for year, group in REcostFiles.groupby('Year'):
    year_tech_dict[year] = list(group['Tech'])
techs = year_tech_dict[year]

H2_prod_cost = pd.read_csv(ProductioncostFiles_paths[0])
h2_conversion_efficiency = float(H2_prod_cost[H2_prod_cost['Year']==year]['efficiency(kWh/kg)'])
BigM =yaml_data.get('BigM', None)

# Supply Curve and Renewable Capacity
all_renewable_profiles,capacities=get_renewable_profiles(year=year,techs = year_tech_dict[year],path=SupplycurveFolder_paths[0],drop_capacity_below=False)

# Demand
demand=get_demand(all_renewable_profiles,year=year, daily_demand=True,freq='h', file_path=demandFiles_paths[0])

# Networks
link_flow_direction, links, all_zones,links_to_zones=get_links(path=Networks_paths[0])

for zone in all_zones:
    if zone not in links_to_zones:
        links_to_zones[zone] = set()

# Transmission Cost
delivery_cost=pd.read_csv(H2DeliveryFile_paths[0])
delivery_cost[delivery_cost['Year']==year]
cost_dict = dict(zip(delivery_cost['Delivery Method'], delivery_cost['OPEX ($/kg)']))
links['Transmission Opex ($/kg)'] = links['Delivery Method'].map(cost_dict) * links['Link Distance']/100

#build cost
build_cost=get_build_cost_matrix(interest,payback_years, payback_years_tank,payback_years_electrolyzer, REcostFiles_paths,ProductioncostFiles_paths,H2StorageFile_paths,year,all_zones)

#IRA incentives
RE_incentives = pd.read_csv(IncentiveFiles_paths[0])
RE_incentives = RE_incentives[RE_incentives['Year']==year]
ITC = dict(zip(RE_incentives['Tech'], RE_incentives['ITC ($/kg)'].fillna(0)))
PTC = dict(zip(RE_incentives['Tech'], RE_incentives['PTC ($/kg)'].fillna(0)))

#Stoage Capacity Limitation
cavern_capacities = pd.read_csv(StorageCapacityFiles_paths[0])
cavern_capacities = dict(zip(cavern_capacities['Zone'], cavern_capacities['capacity'].fillna(0)))
tank_capacities = pd.read_csv(StorageCapacityFiles_paths[1])
tank_capacities = dict(zip(tank_capacities['Zone'], tank_capacities['capacity'].fillna(0)))

# Max Imports Amount
max_imports = demand.sum().sum()*max_imports_ratio

#Time controls
all_renewable_profiles=all_renewable_profiles.loc[start+str(year): end+str(year)]
year_ratio=len((all_renewable_profiles).resample('d').first())/365 # used to estimate total costs for year. Just used for testing. 
max_imports *= year_ratio

build_cost=build_cost*year_ratio
