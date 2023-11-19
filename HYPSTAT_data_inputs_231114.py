###################################################
# Data inputs (input processing)
###################################################

import pandas as pd
import yaml
from Model_Functions_v2_111423 import *
year = 2030
# Specify the path to YAML file
yaml_file_path = 'HYPSTAT_scenario.yaml'

# Read the YAML file
with open(yaml_file_path, 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)
 
# Extract data into variables
firstYear = yaml_data.get('firstYear', None)
lastYear = yaml_data.get('lastYear', None)
timeWindow = yaml_data.get('timeWindow', None)
scenario = yaml_data.get('scenario', None)
truck_size_limit = yaml_data.get('truck_size_limit', None)

demandFiles_paths = yaml_data.get('demandFile', [])
REcostFiles_paths = yaml_data.get('REcostFile', [])
ProductioncostFiles_paths = yaml_data.get('ProductioncostFile', [])
SupplycurveFolder_paths = yaml_data.get('SupplycurveFolder', [])
Networks_paths = yaml_data.get('NetworksFiles', [])
LinkdistanceFile_paths = yaml_data.get('LinkdistanceFile', [])

# create the year and tech dictionary:
REcostFiles = pd.concat([pd.read_csv(path) for path in REcostFiles_paths], ignore_index=True)
year_tech_dict = {}
for year, group in REcostFiles.groupby('Year'):
    year_tech_dict[year] = list(group['Tech'])


# Supply Curve and Renewable Capacity
all_renewable_profiles,capacities=get_renewable_profiles(year=year,techs = year_tech_dict[year],path=SupplycurveFolder_paths[0],drop_capacity_below=False)

# Demand
demand=get_demand(all_renewable_profiles,year=year, daily_demand=True,freq='h', file_path=demandFiles_paths[0])

