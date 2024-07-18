'''
Authors: Joe Brauch, Yijin Li, Steven Percy

This file contains the model object for the HYPSTAT model
'''

import time
import os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path  # Import Path to handle file paths
import copy


from pyomo.environ import *

from Model_Functions_v3_240701 import *

#TODO: Update units for PTC and ITC if they should be $/kWh

class HYPSTAT:
        
    def __init__(self, yaml_file_path='HYPSTAT_scenario.yaml'):
        self.yaml_file_path = yaml_file_path
        self.load_inputs()

    def load_inputs(self,optimize_pipelines=False,Pipeline_Exists=None):
        
        yaml_path = Path(self.yaml_file_path)
        with yaml_path.open('r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        
        self.year = yaml_data.get('studiedyear',None)
        self.time_periods = pd.date_range(str(self.year)+'-01-01 00:00:00', periods=8760, freq="H") 
        self.firstYear = yaml_data.get('firstYear', None)
        self.lastYear = yaml_data.get('lastYear', None)
        self.timeWindow = yaml_data.get('timeWindow', None)

        self.max_imports_ratio = yaml_data.get('max_imports_ratio', None)
        self.min_exports_ratio = yaml_data.get('min_exports_ratio', None)
        self.import_zones = yaml_data.get('import_zones', None)
        self.export_zones = yaml_data.get('export_zones', None)
        self.overserved_cost = yaml_data.get('hydrogen_overserved_cost', None)
        self.grid_prices_by_loc = yaml_data.get('grid_prices_by_loc', False) #if missing, defaults to national grid prices

        self.start = yaml_data.get('start', None)
        self.end = yaml_data.get('end', None)

        self.coarse_resolution = yaml_data.get('coarse_resolution', None)
        self.fine_resolution = yaml_data.get('fine_resolution', None)

        self.demandFiles_paths = yaml_data.get('demandFile', [])
        self.FinancialFiles_paths = yaml_data.get('FinancialFile', [])
        self.REcostFiles_paths = yaml_data.get('REcostFile', [])
        
        #hourly electricty profile
        self.REcostProfileFiles_paths = yaml_data.get('REcostProfileFile', [])
        
        self.ProductioncostFiles_paths = yaml_data.get('ProductioncostFile', [])
        self.H2DeliveryFile_paths = yaml_data.get('H2DeliveryFile', [])
        self.H2StorageFile_paths = yaml_data.get('H2StorageFile', [])
        self.SupplycurveFolder_paths = yaml_data.get('SupplycurveFolder', [])
        self.Networks_paths = yaml_data.get('NetworksFiles', [])
        self.Pipeline_opex_override_path = yaml_data.get('PipelineCostOverideFile',[])
        self.IncentiveFiles_paths = yaml_data.get('IncentiveFiles', [])
        self.StorageCapacityFiles_paths = yaml_data.get('StorageCapacityFiles', [])
        self.StorageLimitsFiles_paths = yaml_data.get('StorageLimitsFiles', [])

        self.REcostFiles = pd.concat([pd.read_csv(path) for path in self.REcostFiles_paths], ignore_index=True)
        #self.REcostProfileFiles = pd.concat([pd.read_csv(path) for path in self.REcostProfileFiles_paths], ignore_index=True)
        #self.REcostProfileFiles.fillna(0,inplace=True)
        #self.REcostProfileFiles.index = self.time_periods
        self.techs = (self.REcostFiles.loc[self.REcostFiles['Year']==self.year])['Tech'].unique()
        #TODO: maybe have technologies as manual input

        self.H2StorageFile = pd.concat([pd.read_csv(path) for path in self.H2StorageFile_paths], ignore_index=True)
    
        self.stor_techs = (self.H2StorageFile.loc[self.H2StorageFile['Year']==self.year])['Tech'].unique()
        self.Storage_charge_cost = dict(zip(self.H2StorageFile['Tech'], self.H2StorageFile['OPEX ($/kg)'])) #NOTE: only charge cost, but labeled as opex
               
        self.all_renewable_profiles, self.capacities = get_renewable_profiles(year=self.year, techs=self.techs, path=self.SupplycurveFolder_paths[0], drop_capacity_below=False) #[RE profiles: CF (kWh/kW)/hr for each hour; capacities: MW] TODO: make capacities kW?
        self.demand = get_demand(self.all_renewable_profiles, year=self.year, daily_demand=True, freq='h', file_path=self.demandFiles_paths[0])
        
        self.link_flow_direction, self.links, self.all_zones, self.links_to_zones = get_links(self.Networks_paths[0])

        #Fill NaNs in links to avoid issues in model solving
        self.links.fillna(value=0, inplace=True)
        
        self.delivery_cost = pd.read_csv(self.H2DeliveryFile_paths[0])
        self.delivery_cost = (self.delivery_cost[self.delivery_cost['Year'] == self.year]).set_index('Tech')
        self.links['Pipeline Opex ($/kg)'] = self.delivery_cost.loc['Pipeline','Variable OPEX ($/kg/100-km)'] * \
                                                  self.links['Pipeline distance [km]'] / 100

        self.links['Truck LCOT ($/kg)'] = self.delivery_cost.loc['Truck','LCOT ($/kg/100-km)'] * \
                                                  self.links['Truck distance [km]'] / 100

        self.pipeline_fixed_capex = self.delivery_cost.loc['Pipeline','Fixed CAPEX ($/km)']
        self.pipeline_capacity_capex = self.delivery_cost.loc['Pipeline','Capacity CAPEX ($/(kg/hr)/km)']  
        self.pipeline_fixed_opex_frac = self.delivery_cost.loc['Pipeline','Fixed OPEX (fraction of CAPEX/yr)']      

        #TODO: figure out better input structure for pipeline opex
        if self.Pipeline_opex_override_path is not None and self.Pipeline_opex_override_path.count(None) != len(self.Pipeline_opex_override_path):
            self.pipeline_opex_overrides = pd.concat([pd.read_csv(path) for path in self.Pipeline_opex_override_path], ignore_index=True)
            self.pipeline_opex_overrides.set_index('Link',drop=True,inplace=True)
            print(self.pipeline_opex_overrides)
            for link in self.links.index:
                if link in self.pipeline_opex_overrides.index:
                    self.links.loc[link,'Pipeline Opex ($/kg)'] = self.pipeline_opex_overrides.loc[link,'Pipeline OPEX override [$/kg]']

        #TODO: update this so that it also has an index for time, and make sure indexing matches in objective function
        
        
        
        
        
        ######RE OPEX--manual input for now for testing; TODO: coordinate with Yijin's inputs
        # BUILD ELECTRICITY GEN OPEX TABLE
        #   The electricity gen opex table is indexed for each zone and technology and is indexed for time intervals to allow for variable pricing (e.g., wholesale grid).
        #   A constant variable opex (in currency/kWh) is automatically loaded from the technology financials file. 
        #   If provided, profile files for electricity gen opex will override the constant value in the financial files--i.e., the variable OPEX value in the technology financials file is NOT used.
                    
        multi_index = pd.MultiIndex.from_product([self.all_zones,self.techs])
        self.re_opex = pd.DataFrame(data=0, index=self.time_periods, columns=multi_index)

        # fill electricity gen opex table with constant variable opex values from REcostFiles
        for z in self.all_zones:
            for tech in self.techs:
                self.re_opex.loc[:,(z,tech)] = self.REcostFiles.set_index(['Year','Tech']).loc[(self.year,tech),'Variable OPEX ($/kWh)']

        # override electricity gen opex for technologies with time profiles (e.g., grid representing hourly wholesale pricing)
        #   Note: REcostProfileFiles must start with the name of the technology that they should be overriding. If multiple files for a single technology are provided,
        #   only the first will be used. If a file does not start with the name of the technology, that file will not be used.
                
        # TODO: add locational pricing
        for z in self.all_zones:
            for tech in self.techs:
                if tech not in self.REcostProfileFiles_paths.keys():
                    continue
                self.REcostProfileFiles = pd.read_csv(self.REcostProfileFiles_paths[tech])
                self.REcostProfileFiles.fillna(0,inplace=True)
                self.REcostProfileFiles.index = self.time_periods
                if self.grid_prices_by_loc: #CAN ONLY BE FALSE RIGHT NOW
                    self.re_opex.loc[:,(z,tech)] = self.REcostProfileFiles[z]/1000 # [$/kWh]
                else:
                    self.re_opex.loc[:,(z,tech)] = self.REcostProfileFiles['Price ($/MWh)']/1000 #TODO: remove 1000 and figure out way to input or specify column name
                
        #year is removed
        self.build_cost = get_build_cost_matrix(self.FinancialFiles_paths,self.REcostFiles_paths,self.ProductioncostFiles_paths,self.H2StorageFile_paths,self.year, self.all_zones) #[EUR/kW for generators and electrolyzers, EUR/kg wk cap for storage]
        
        self.H2_prod_cost = pd.read_csv(self.ProductioncostFiles_paths[0])
        self.h2_conversion_efficiency = float(self.H2_prod_cost[self.H2_prod_cost['Year']==self.year]['efficiency(kWh/kg)']) #[kWh/kg]
        self.compressor_charge = float(self.H2_prod_cost[self.H2_prod_cost['Year']==self.year]['Variable OPEX ($/kg)']) #[kWh/kg]

        self.RE_incentives = pd.read_csv(self.IncentiveFiles_paths[0])
        self.RE_incentives = self.RE_incentives[self.RE_incentives['Year'] == self.year]
        self.ITC = dict(zip(self.RE_incentives['Tech'], self.RE_incentives['ITC ($/kg)'].fillna(0))) #[EUR/kWh]
        self.PTC = dict(zip(self.RE_incentives['Tech'], self.RE_incentives['PTC ($/kg)'].fillna(0))) #[EUR/kWh]
        
        self.storage_capacities = pd.read_csv(self.StorageCapacityFiles_paths[0]).set_index('Node')
        self.storage_capacities = self.storage_capacities.applymap(lambda x: 'inf' if str(x).strip("'") == 'inf' else pd.to_numeric(x, errors='coerce'))
        self.storage_capacities = {index: row.to_dict() for index, row in self.storage_capacities.iterrows()} #[kg wk capacity]

        self.H2_Storage_Limit = pd.read_csv(self.StorageLimitsFiles_paths[0])
        self.H2_Storage_Limit['capacity'] = self.H2_Storage_Limit['capacity'].apply(lambda x: 'inf' if str(x).strip("'") == 'inf' else pd.to_numeric(x, errors='coerce'))
        self.Total_storage_capacity = dict(zip(self.H2_Storage_Limit['Storage Tech'], self.H2_Storage_Limit['capacity'])) #[kg wk capacity]

        self.max_imports = self.demand.sum().sum() * self.max_imports_ratio
        self.min_exports = self.demand.sum().sum() * self.min_exports_ratio #NOTE: ratio or amount could be changed
        self.Cost_of_Unserved_H2 = yaml_data.get('Cost_of_Unserved_H2', None) #[EUR/kg]
        self.Storage_charge_limit = dict(zip(self.H2_Storage_Limit['Storage Tech'],self.H2_Storage_Limit['charge limit'].apply(lambda x: 'inf' if str(x).strip("'") == 'inf' else pd.to_numeric(x, errors='coerce'))))
        self.Storage_discharge_limit = dict(zip(self.H2_Storage_Limit['Storage Tech'],self.H2_Storage_Limit['discharge limit'].apply(lambda x: 'inf' if str(x).strip("'") == 'inf' else pd.to_numeric(x, errors='coerce'))))
        self.storage_limits = yaml_data.get('storage_limits', None)
      

        self.all_renewable_profiles = self.all_renewable_profiles.loc[self.start + str(self.year): self.end + str(self.year)]
        self.year_ratio = len(self.all_renewable_profiles.resample('d').first()) / 365
        self.max_imports *= self.year_ratio
        self.build_cost *= self.year_ratio

        if optimize_pipelines:
            #first iteration
            self.h=self.coarse_resolution # [hrs/period] this specifies the number of hours in each time interval.  It allows the conversion of kg/h when multiple hours are in each timestep.  if resampling data please change this value 
            print('Optimizing pipelines')
            print()
        else:
            #second iteration, hourly run
            self.h=self.fine_resolution # [hrs/period]
            print('Performing full optimization')
            print()

        if Pipeline_Exists is not None:
            #eliminates certain links based on a previous round of optimization (or user input???)
            threshold = 0.1
            for link in Pipeline_Exists.index:
                if Pipeline_Exists[link] < threshold:
                    #close to 0, pipeline doesn't exist
                    self.links.loc[link,'Pipeline max capacity [kg/hr]'] = 0
                    self.links.loc[link,'Pipeline allowed'] = 'N'


        if self.h != 1: #TODO: think about time resolution, for now leave as minimum 1 hour
            self.all_renewable_profiles = self.all_renewable_profiles.resample('{}h'.format(self.h)).sum() #[period CF, i.e., (kWh/kW)/period]
            self.demand = self.demand.resample('{}h'.format(self.h)).sum() # [kg/period]
            self.re_opex = self.re_opex.resample('{}h'.format(self.h)).mean() # [EUR/kWh, average]

    
    def load_model(self,optimize_pipelines=False):
        self.m = ConcreteModel()

        #Define time and node index sets
        self.t_dict = {ind + 1: k for ind, k in enumerate(self.all_renewable_profiles.index)}

        ### INDEX SETS ###
        self.m.T = Set(initialize=(sorted(list(self.t_dict.keys()))), ordered=True) # sort t_dict
        self.m.Zones = Set(initialize=list(self.all_zones), ordered=True)
        self.m.Links = Set(initialize=list(self.links.index), ordered=True)
        self.m.Gen_Techs = Set(initialize=list(self.techs), ordered=True)
        self.m.Stor_Techs = Set(initialize=list(self.stor_techs),ordered=True)
        self.m.Gen_Tranches = Set(initialize=list(self.capacities.index), ordered=True) # This set defines the index for different resource qualities in a single location

        '''
        TODO: SETS THAT ARE MISSING
        -H2 production techs
        -tank-based transmission techs
        -Pipelines? Maybe extra tranches
        '''

        #TODO: HOW to deal with self.h boolean and conditioning 

        ### PARAMETERS ###
        self.m.Cost_of_unserved_H2 = Param(initialize=self.Cost_of_Unserved_H2) # [EUR/kg] We assume that a kg of unserved hydrogen has an economic cost of 10eX.
        self.m.H2_conversion_efficiency = Param(initialize=self.h2_conversion_efficiency) # [kW/kg H2]
        self.Storage_charge_limit = dict(zip(self.H2_Storage_Limit['Storage Tech'],self.H2_Storage_Limit['charge limit'].apply(lambda x: 'inf' if str(x).strip("'") == 'inf' else pd.to_numeric(x, errors='coerce')))) #[% of wk cap per hour]
        self.Storage_discharge_limit = dict(zip(self.H2_Storage_Limit['Storage Tech'],self.H2_Storage_Limit['discharge limit'].apply(lambda x: 'inf' if str(x).strip("'") == 'inf' else pd.to_numeric(x, errors='coerce')))) #[% of wk cap per hour]
        self.storage_charge_limit_values = {tech: 'inf' if isinstance(value, str) else value * self.h for tech, value in self.Storage_charge_limit.items()} #[% of wk cap per period or INF]
        self.storage_discharge_limit_values = {tech: 'inf' if isinstance(value, str) else value * self.h for tech, value in self.Storage_discharge_limit.items()} #[% of wk cap per period or INF]

        # Use the dictionaries to initialize the parameters
        self.m.Storage_charge_limit = Param(self.m.Stor_Techs, initialize=self.storage_charge_limit_values,within=Any) #[% of wk cap per period or INF]
        self.m.Storage_discharge_limit = Param(self.m.Stor_Techs, initialize=self.storage_discharge_limit_values,within=Any) #[% of working cap per period or INF]
        self.m.Storage_charge_cost = Param(self.m.Stor_Techs,initialize={str(st): self.Storage_charge_cost[st] for st in self.m.Stor_Techs},within=Reals) #[EUR/kg]
        self.m.Total_storage_capacity = Param(self.m.Stor_Techs, initialize={str(st): self.Total_storage_capacity[st] for st in self.m.Stor_Techs},within=Any) #[kg or INF]

        self.m.Truck_flow_allowed = Param(self.m.Links,initialize={str(lk): 1 if self.links.loc[lk,'Truck allowed']=='Y' else 0 for lk in self.m.Links}) #0 if not allowed for each link, 1 if allowed (multiplier for capacity limits)
        self.m.Pipeline_flow_allowed = Param(self.m.Links,initialize={str(lk): 1 if self.links.loc[lk,'Pipeline allowed']=='Y' else 0 for lk in self.m.Links})

        ### CAPACITY VARIABLES ###
        self.m.Gen_Capacity = Var(self.m.Gen_Techs,self.m.Zones,self.m.Gen_Tranches, domain=NonNegativeReals, initialize=0)  #[kW]
        self.m.Electrolyser_Capacity = Var(self.m.Zones, domain=NonNegativeReals) #[kW] #TODO: split into different types of production (maybe rename)
        self.m.Storage_Capacity = Var(self.m.Stor_Techs,self.m.Zones,domain=NonNegativeReals) #[kg] wk capacity
        self.m.Pipeline_Capacity = Var(self.m.Links,domain=NonNegativeReals) # [kg/period] pipeline capacity for each zone
        if optimize_pipelines:
            self.m.Pipeline_Exists = Var(self.m.Links,domain=Binary) #whether or not a pipeline exists
        
        
        ### PRODUCTION AND FLOW VARIABLES ###

        #TODO: will need to add variable for specific flows of electricity to specific production types

        # Electricity
        self.m.Electricity_Potential  = Var(self.m.Gen_Techs, self.m.T, self.m.Zones)  #[kWh] potential to generate electricity from RE (or grid) 
        self.m.Electricity_Used = Var(self.m.Gen_Techs, self.m.T, self.m.Zones) #[kWh] electricity used for hydrogen production
        self.m.Electricity_Curtailed = Var(self.m.Gen_Techs,self.m.T, self.m.Zones, domain=NonNegativeReals) #[kWh] electricity production

        # Hydrogen
        self.m.H2_Production = Var(self.m.T, self.m.Zones, domain=NonNegativeReals)   #[kg/period] #TODO: split into different types of production
        self.m.H2_Demand_Met = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) #[kg/period]
        self.m.H2_Unserved = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) #[kg/period],bounds=(0,0) )   #kg/per interval
        self.m.H2_Overserved = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) #[kg/period] ,bounds=(0,0) )  
        self.m.H2_Imports = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) #[kg/period] imported hydrogen, kg/interval
        self.m.H2_Exports = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) #[kg/period] exported hydrogen, kg/interval

        # Storage
        self.m.Storage_Charge = Var(self.m.Stor_Techs,self.m.T,self.m.Zones) #[kg]
        self.m.Storage_Level = Var(self.m.Stor_Techs,self.m.T,self.m.Zones,domain=NonNegativeReals) #[kg]
        self.m.Storage_Charge_OPEX = Var(self.m.Stor_Techs,self.m.T,self.m.Zones, domain=NonNegativeReals) #[EUR/period]
        
        # Transmission
        self.m.Link_Flow = Var(self.m.T, self.m.Links)  #[kg]
        self.m.Pipeline_Flow = Var(self.m.T, self.m.Links) #[kg]
        self.m.Truck_Flow = Var(self.m.T,self.m.Links) #[kg]
        self.m.Pipeline_OPEX = Var(self.m.T, self.m.Links, domain=NonNegativeReals) #[€/period] #TODO: rename all of these to specify OPEX
        self.m.Truck_Cost = Var(self.m.T, self.m.Links, domain=NonNegativeReals) #[€/period] #TODO: rename all of these to specify OPEX, TODO: generalize to all tank-based transmission
        
        

    def load_constraints(self,optimize_pipelines=False):
        # Load pipeline flow incidence matrix
        def link_rule(m, link, zone):
            return float(self.link_flow_direction.loc[link, zone])

        self.m.link_flow_direction = Param(self.m.Links, self.m.Zones, initialize=link_rule)

        ### DEMAND CONSTRAINTS ###
        print('Setting up demand constraints...')

        # Option 1: Meet demand over course of day
        def demand_rule(m, t, zone):
            if t%24==1:
                return (sum(m.H2_Demand_Met[t_step,zone] for t_step in range(t,t+24)) == sum(self.demand.loc[self.t_dict[t_step],zone] for t_step in range(t,t+24))) #TODO: May need to revise to consider 'h' value
            else:
                return Constraint.Skip

        # Option 2: Meet demand in each hour
        def demand_rule_2(m, t, zone):
            return (m.H2_Demand_Met[t,zone] == self.demand.loc[self.t_dict[t],zone]) #[kg/period]

        #self.m.demand_constraint = Constraint(m.T, m.Zones, rule=demand_rule)
        self.m.demand_constraint = Constraint(self.m.T, self.m.Zones, rule=demand_rule_2)

        ### IMPORT CONSTRAINTS ###
        print('Setting up import constraints...')

        # Limit quantity of total imports
        def total_imports_rule(m):
            return sum(sum(m.H2_Imports[t,zone] for t in m.T) for zone in m.Zones) <= self.max_imports #[kg]
        
        self.m.total_imports_constraint = Constraint(rule=total_imports_rule)

        # Restrict which zones are allowed to import
        def zone_imports_rule(m, t, zone):
            if zone in self.import_zones:
                return Constraint.Skip
            else:
                return m.H2_Imports[t,zone] == 0 #[kg/period]
        
        self.m.zone_imports_constraint = Constraint(self.m.T,self.m.Zones,rule=zone_imports_rule)

        ### EXPORT CONSTRAINTS ###
        print('Setting up export constraints...')

        # Ensure that model meets a minimum quota of exports
        def total_exports_rule(m):
            return (sum(sum(m.H2_Exports[t,zone] for t in m.T) for zone in m.Zones)) >= self.min_exports #[kg]

        self.m.total_exports_constraint = Constraint(rule=total_exports_rule)
        # TODO: potentially incorporate specific export quotas for each zone

        # Restrict which zones are allowed to export
        def zone_exports_rule(m, t, zone):
            if zone in self.export_zones:
                return Constraint.Skip
            else:
                return m.H2_Exports[t,zone] == 0 #[kg/period]
        
        self.m.zone_exports_constraint = Constraint(self.m.T,self.m.Zones,rule=zone_exports_rule)

        ### STORAGE CONSTRAINTS ###
        print('Setting up storage constraints...')

        # TODO: clarify point within the hour that we mean for storage, transmission, etc.

        # Constraint storage level based on inflow/outflow of storage
        def Storage_mutli_period_rule(m, stor_tech, t, zone):
            if t == m.T.first():
                return m.Storage_Level[stor_tech,t,zone] == m.Storage_Level[stor_tech,m.T.last(),zone] + m.Storage_Charge[stor_tech,t,zone] #[kg]
            else:
                return (m.Storage_Level[stor_tech, t, zone] == m.Storage_Level[stor_tech, t-1, zone]  + m.Storage_Charge[stor_tech, t, zone]) #[kg]

        self.m.Storage_multi_period_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_mutli_period_rule)

        # Constrain storage charge speed
        def Storage_charge_rule(m, stor_tech, t, zone):
            if m.Storage_charge_limit[stor_tech]=='inf':
                return Constraint.Skip
                #Could also use just a really large number but probably leave as skip constraint
            else:
                return (m.Storage_Charge[stor_tech, t, zone] <= m.Storage_charge_limit[stor_tech]*m.Storage_Capacity[stor_tech, zone]) #[kg]
        
        self.m.Storage_charge_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_charge_rule)

        # Constrain storage discharge speed
        def Storage_discharge_rule(m, stor_tech, t, zone):
            if m.Storage_discharge_limit[stor_tech]=='inf':
                return Constraint.Skip
            else:
                return (m.Storage_Charge[stor_tech, t, zone] >= -m.Storage_discharge_limit[stor_tech]*m.Storage_Capacity[stor_tech, zone]) #[kg]

        self.m.Storage_discharge_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_discharge_rule)

        
        # TODO: note in documentation that storage capacities need to be working capacities
        # Constrain storage level to be within capacity
        def Storage_level_rule(m, stor_tech, t, zone):
            return (m.Storage_Level[stor_tech, t, zone] <= m.Storage_Capacity[stor_tech, zone]) #[kg]

        self.m.Storage_level_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_level_rule)

        #TODO: consolidate all of this into some sort of input table with limits for each storage type and zone
        #TODO: consider if we want to still have a total storage constraint

        # Constrain storage capacity within each zone (e.g., for geologic limitations)
        def Storage_capacity_cap_rule(m, stor_tech, zone):
            cap = self.storage_capacities[stor_tech][zone]
            if cap=='inf':
                return Constraint.Skip
            else:
                return (m.Storage_Capacity[stor_tech, zone] <= cap) #[kg]

        self.m.Storage_capacity_cap_constraint = Constraint(self.m.Stor_Techs, self.m.Zones, rule=Storage_capacity_cap_rule)

        # Constrain total storage capacity among all storage of a given type
        def Total_storage_capacity_rule(m, stor_tech):
            if m.Total_storage_capacity[stor_tech]=='inf':
                return Constraint.Skip
            else:
                return (sum(m.Storage_Capacity[stor_tech, zone] for zone in m.Zones) <= m.Total_storage_capacity[stor_tech]) #[kg]

        self.m.Total_storage_capacity_constraint = Constraint(self.m.Stor_Techs, rule=Total_storage_capacity_rule)

        ### LINK FLOW CONSTRAINTS ###
        print('Setting up flow constraints...')
        # Pipeline flow (update this later to be part of the variable bound)

        #TODO: have upper bound on pipeline size as an input

        # Constrain pipeline sizes based on binary variables OR based on inputs from the previous iteration
        if optimize_pipelines:
            def pipeline_exists_rule(m, link):
                #bigM = self.bigM *self.h #kg/period, upper bound on pipeline capacity (~7000 TPD)
                return m.Pipeline_Capacity[link] <= self.links.loc[link,'Pipeline max capacity [kg/hr]']*self.h*m.Pipeline_flow_allowed[link]*m.Pipeline_Exists[link] #[kg/period]
            
            self.m.pipeline_exists_constraint = Constraint(self.m.Links,rule=pipeline_exists_rule)
        
        else:
            #not optimizing pipelines, limit sizes according to links

            #TODO: think about how pipeline capacity limits (i.e., which pipelines exist) is passed in the second iteration

            def pipeline_size_rule(m, link):
                return m.Pipeline_Capacity[link] <= self.links.loc[link,'Pipeline max capacity [kg/hr]']*self.h*m.Pipeline_flow_allowed[link] #[kg/period]
            
            self.m.pipeline_size_constraint = Constraint(self.m.Links, rule=pipeline_size_rule)

        # Set total link flow for mass balance based on pipeline and truck flow
        
        def total_link_flow_rule(m, t, link):
            return m.Link_Flow[t, link] == m.Pipeline_Flow[t, link] + m.Truck_Flow[t, link]  #[kg]

        self.m.total_link_flow_constraint = Constraint(self.m.T, self.m.Links, rule=total_link_flow_rule)
        
        
        # TODO: generalize into tank-based transmission
        # TODO: think about if we want to incorporate capex/fixed opex or keep as levelized cost (via opex)

        # Limit truck flows according to size limit
        def forward_max_truck_rule(m, t, link):
            return m.Truck_Flow[t,link] <= self.links.loc[link,'Truck max capacity [kg/hr]']*self.h*m.Truck_flow_allowed[link] #[kg/period]
        
        self.m.forward_max_truck_constraint = Constraint(self.m.T, self.m.Links, rule=forward_max_truck_rule)

        def reverse_max_truck_rule(m, t, link):
            return m.Truck_Flow[t,link] >= -self.links.loc[link,'Truck max capacity [kg/hr]']*self.h*m.Truck_flow_allowed[link] #[kg/period]

        self.m.reverse_max_truck_constraint = Constraint(self.m.T, self.m.Links, rule=reverse_max_truck_rule)

        # Limit pipeline flows according to pipeline capacities
        def pipeline_forward_flow_rule(m, t, link):
            return m.Pipeline_Flow[t, link] <= m.Pipeline_Capacity[link] #[kg/period]
            
        self.m.pipeline_forward_flow_constraint = Constraint(self.m.T, self.m.Links, rule=pipeline_forward_flow_rule)

        def pipeline_reverse_flow_rule(m, t, link):
            return m.Pipeline_Flow[t, link] >= -1*m.Pipeline_Capacity[link] #[kg/period]

        self.m.pipeline_reverse_flow_constraint = Constraint(self.m.T, self.m.Links, rule=pipeline_reverse_flow_rule)

        ### PRODUCTION CONSTRAINTS ###
        print('Setting up electricity and hydrogen production constraints...')
        
        # Define the electricity production potential at each node in kwh
        def Electricity_Potential_rule(m, tech, t, zone):
            return m.Electricity_Potential[tech, t, zone] == sum(self.all_renewable_profiles.loc[self.t_dict[t],(tech,zone,tranche)] * m.Gen_Capacity[tech,zone,tranche]  for tranche in get_producers(self.capacities,zone=zone,tech = tech).index ) #[kWh, from kW * (kWh/kW)/period for period]

        self.m.Electricity_Potential_constraint = Constraint(self.m.Gen_Techs, self.m.T, self.m.Zones, rule=Electricity_Potential_rule)

        # Constrain the capacity of electricity generators
        def Gen_build_limit_rule(m, tech, zone, tranche):
            capacity = get_producers(self.capacities,zone=zone,tech=tech)
            if tranche in capacity.index:
                if not capacity.loc[tranche]==capacity.loc[tranche]:
                    capacity.loc[tranche] = 0
                return  (m.Gen_Capacity[tech,zone,tranche] <= capacity.loc[tranche] * 1000) # [kW, capacity must be in MW] #TODO: specify units in inputs to avoid this 1000
            else:
                return Constraint.Skip

        self.m.Renewable_build_limit_constraint = Constraint(self.m.Gen_Techs, self.m.Zones, self.m.Gen_Tranches, rule=Gen_build_limit_rule)

        # TODO: recast curtailment and electrolyzer capacity based on generalized electricity/production flows
        # Constrain curtailed electricty to less than production
        def Curtailed_electricity_limit_rule(m, tech, t, zone):
            return m.Electricity_Curtailed[tech, t, zone] <= m.Electricity_Potential[tech, t, zone] #[kWh/period]
        
        self.m.Curtailed_electricity_limit_constraint = Constraint(self.m.Gen_Techs, self.m.T, self.m.Zones, rule=Curtailed_electricity_limit_rule) #can't curtailed more electricity than is produced

        # Set amount of electricity used based on curtailed electricity
        def Electricity_Used_rule(m, tech, t, zone):
            return (m.Electricity_Used[tech, t, zone] == m.Electricity_Potential[tech, t, zone] - m.Electricity_Curtailed[tech, t, zone]) #[kWh/period]

        self.m.Electricity_Used_constraint = Constraint(self.m.Gen_Techs, self.m.T, self.m.Zones, rule=Electricity_Used_rule)

        # Constrain electricity used based on electrolyzer capacity
        def Electrolyser_Capacity_Limit_rule(m, t, zone):
            return (sum(m.Electricity_Used[tech, t, zone] for tech in m.Gen_Techs)) <= m.Electrolyser_Capacity[zone]*self.h #[kWh/period]

        self.m.electrolyser_capacity_limit_constraint = Constraint(self.m.T, self.m.Zones, rule=Electrolyser_Capacity_Limit_rule)

        # New H2 production build at nodes
        # electricity is in kWh
        # TODO: generalize to various H2 production pathways (focused on electrolysis)
            # TODO: will need to think about how renewable production matches up with H2 production
            # TODO: create variables for specification of electricity flows from each RE tech to each H2 production tech
        
        # Set H2 production
        def H2_Production_rule(m, t, zone):
            return m.H2_Production[t, zone] == (sum(m.Electricity_Used[tech, t, zone] for tech in m.Gen_Techs))/m.H2_conversion_efficiency #[kg/period]

        self.m.H2_Production_constraint = Constraint(self.m.T, self.m.Zones, rule=H2_Production_rule)

        ### POWER BALANCE ###
        print('Setting up power balance constraints...')

        # Set the hydrogen balance
        def H2_Balance_rule(m, t, zone):
            links_to_this_zone=self.links_to_zones[zone]
            return (m.H2_Production[t, zone] + m.H2_Unserved[t, zone] + sum(m.Link_Flow[t, link] * m.link_flow_direction[link, zone] for link in links_to_this_zone) + m.H2_Imports[t,zone] == \
                    sum(m.Storage_Charge[stor_tech, t, zone] for stor_tech in m.Stor_Techs) + m.H2_Demand_Met[t, zone] + m.H2_Overserved[t, zone] + m.H2_Exports[t, zone]) #[kg/period]

        self.m.H2_Balance_constraint = Constraint(self.m.T, self.m.Zones, rule=H2_Balance_rule)

        ### TRANSMISSION AND STORAGE COSTS ###

        print('Setting up transmission cost constraints...')
        #TODO: work out in inputs somewhere how to add multipliers for links or set link costs in some specific way (or disallow links)
        
        # Set and truck costs
        def Forward_Pipeline_OPEX_rule(m, t, link):
            return (m.Pipeline_OPEX[t,link] >= m.Pipeline_Flow[t,link]*self.links.loc[link,'Pipeline Opex ($/kg)']) #[EUR/period]

        self.m.Forward_Pipeline_OPEX_constraint = Constraint(self.m.T, self.m.Links, rule=Forward_Pipeline_OPEX_rule)

        def Reverse_Pipeline_OPEX_rule(m, t, link):
            return (m.Pipeline_OPEX[t,link] >= -m.Pipeline_Flow[t,link]*self.links.loc[link,'Pipeline Opex ($/kg)']) #[EUR/period]

        self.m.Reverse_Pipeline_OPEX_constraint = Constraint(self.m.T, self.m.Links, rule=Reverse_Pipeline_OPEX_rule)

        def Forward_Truck_Cost_rule(m, t, link): #TODO: generalize to all tank transport
            return (m.Truck_Cost[t,link] >= m.Truck_Flow[t,link]*self.links.loc[link,'Truck LCOT ($/kg)']) #[EUR/period] for now, would like to remove /100 and just handle in inputs
        
        self.m.Forward_Truck_Cost_constraint = Constraint(self.m.T, self.m.Links, rule=Forward_Truck_Cost_rule)

        def Reverse_Truck_Cost_rule(m, t, link):
            return (m.Truck_Cost[t,link] >= -m.Truck_Flow[t,link]*self.links.loc[link,'Truck LCOT ($/kg)']) #[EUR/period] for now, would like to remove /100 and just handle in inputs
        
        self.m.Reverse_Truck_Cost_constraint = Constraint(self.m.T, self.m.Links, rule=Reverse_Truck_Cost_rule)

        print('Setting up storage cost constraints...')
        # TODO: allow for variable opex on discharge if desired by user

        #Set storage charge costs
        def Storage_Charge_OPEX_rule(m, stor_tech, t, zone):
            return (m.Storage_Charge_OPEX[stor_tech,t,zone] >= m.Storage_Charge[stor_tech,t,zone]*m.Storage_charge_cost[stor_tech]) #[EUR/period]

        self.m.Storage_Charge_OPEX_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_Charge_OPEX_rule)

        #print(self.re_opex.index)
        #print(self.re_opex.columns)
        #print(self.m.T)

        print('Setting up objective function...')
        #### Objective function

        def get_CRF(interest=0.1, years=30):
            return (interest * (1 + interest) ** years) /((1 + interest) ** years - 1)

        
        def objective_rule(m):
            
            Gen_Stor_CAPEX = sum(
                sum(m.Storage_Capacity[stor_tech, zone] * self.build_cost.loc['{}'.format(stor_tech),zone] for stor_tech in m.Stor_Techs) + #storage technology
                m.Electrolyser_Capacity[zone] * self.build_cost.loc['PEM Electrolyzer',zone] + #electrolyzers TODO: make this not hard coded
                sum(sum(m.Gen_Capacity[tech,zone,tranche] * self.build_cost.loc[tech,zone] for tech in m.Gen_Techs) for tranche in m.Gen_Tranches) #generators
            for zone in m.Zones)
            
            Pipeline_raw_CAPEX = sum(
                ((m.Pipeline_Capacity[link]/self.h)*self.pipeline_capacity_capex + self.pipeline_fixed_capex*(m.Pipeline_Exists[link] if optimize_pipelines else 1))*self.year_ratio*self.links.loc[link,'Pipeline distance [km]']
            for link in m.Links)

            Pipeline_CAPEX = Pipeline_raw_CAPEX*get_CRF() + Pipeline_raw_CAPEX*self.pipeline_fixed_opex_frac #hard-coded for now, TODO: make pipeline opex as part of inputs

            Transmission_OPEX = sum(
                sum(m.Pipeline_OPEX[t,link] for t in m.T) + #pipeline opex
                sum(m.Truck_Cost[t,link] for t in m.T) #truck costs
            for link in m.Links)
            
            Gen_Stor_OPEX = sum(
                sum(sum(m.Storage_Charge_OPEX[stor_tech,t,zone] for t in m.T) for stor_tech in m.Stor_Techs) + #storage opex
                sum(sum((m.Electricity_Used[tech, t, zone])*self.re_opex.loc[self.t_dict[t],(zone,tech)] for t in m.T) for tech in m.Gen_Techs) #gen opex
            for zone in m.Zones)

            Incentives = (
                -sum(sum(sum((m.Electricity_Used[tech, t, zone])*self.PTC[tech] for zone in m.Zones) for t in m.T) for tech in m.Gen_Techs) + #PTC effects
                -sum(sum(sum((m.Electricity_Potential[tech,t,zone])*self.ITC[tech] for zone in m.Zones) for t in m.T) for tech in m.Gen_Techs)
             ) #ITC effects

            Penalties = sum(
                sum(m.H2_Unserved[t, zone] * m.Cost_of_unserved_H2 for t in m.T) + #unserved hydrogen
                sum(m.H2_Overserved[t, zone] * self.overserved_cost for t in m.T) #overserved hydrogen
            for zone in m.Zones)

            return Gen_Stor_CAPEX + Pipeline_CAPEX + Transmission_OPEX + Gen_Stor_OPEX + Incentives + Penalties
                    
            #log_infeasible_constraints(m)
        self.m.objective = Objective(rule=objective_rule, sense=minimize)
            
    def solve_model(self,optimize_pipelines=False,solver='glpk'):
        stream_solver=True
        print('Solving')
        if solver=='gurobi':
            opt = SolverFactory("gurobi", solver_io="python")
        elif solver=='glpk':
            opt = SolverFactory('glpk')
        else:
            raise ValueError('Solver specified is not allowed!')
        results_final = opt.solve(self.m, tee=stream_solver)
        
        if optimize_pipelines:
            self.Pipeline_Exists = pd.Series(self.m.Pipeline_Exists.extract_values(), index=self.m.Pipeline_Exists.extract_values().keys())
    
    def two_step_solve(self,solver='glpk'):

        start = time.time()

        #solve first, optimizing for pipelines
        self.load_inputs(optimize_pipelines=True)
        self.load_model(optimize_pipelines=True)
        self.load_constraints(optimize_pipelines=True)
        self.solve_model(optimize_pipelines=True,solver=solver)

        mid = time.time()

        #solve second, not optimizing
        self.load_inputs(optimize_pipelines=False,Pipeline_Exists=self.Pipeline_Exists)
        self.load_model(optimize_pipelines=False)
        self.load_constraints(optimize_pipelines=False)
        self.solve_model(optimize_pipelines=False,solver=solver)

        stop = time.time()

        print()
        print('### MODEL SOLVE COMPLETE ###')
        print('Initial solve took {} seconds; full solve took {} seconds. Total time: {} seconds'.format(mid-start,stop-mid,stop-start))
        print()

    def write_outputs(self,results_dir):

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        else:
            con = input("results_dir '{}' already exists. Do you want to overwrite? (y/n)".format(results_dir))
            while True:
                if con=='n':
                    con2 = input("Press enter to abort (no results writing) or enter a new directory name to write results to")
                    if con2 == '':
                        return
                    else:
                        return self.write_outputs(con2)
                elif con == 'y':
                    break
                else:
                    con = input('Invalid input. Please respond (y/n)')


        #collect useful parameters and write to csv for future reference
        params = dict()
        params['year_ratio'] = self.year_ratio
        params['max_imports'] = self.max_imports
        params['h'] = self.h
        params['obj'] = self.m.objective()

        print(params['obj'])

        pd.Series(params).to_csv(results_dir+'/params.csv')
        self.build_cost.to_csv(results_dir+'/build_cost.csv')
        PTC_data = pd.DataFrame.from_dict(self.PTC,orient='index')
        PTC_data.to_csv(results_dir+'/PTC.csv')
        ITC_data = pd.DataFrame.from_dict(self.ITC,orient='index')
        ITC_data.to_csv(results_dir+'/ITC.csv')

      
        
        Storage_Capacity = pd.Series(self.m.Storage_Capacity.extract_values(), index=self.m.Storage_Capacity.extract_values().keys()).unstack()
        Renewable_Capacity = pd.Series(self.m.Gen_Capacity.extract_values(), index=self.m.Gen_Capacity.extract_values().keys()).unstack()
        Zone_Capacities=Renewable_Capacity.sum(1).unstack()
        Electrolyser_Capacity = pd.Series(self.m.Electrolyser_Capacity.extract_values(), index=self.m.Electrolyser_Capacity.extract_values().keys())
        Zone_Capacities.loc['PEM_Electrolyser']=Electrolyser_Capacity
        Zone_Capacities = pd.concat([Zone_Capacities,Storage_Capacity],axis=0)

        H2_Overserved = pd.Series(self.m.H2_Overserved.extract_values(), index=self.m.H2_Overserved.extract_values().keys()).unstack()
        H2_Overserved.index=list(self.t_dict.values())
        H2_Overserved.to_csv(results_dir+'/H2_overserved.csv')

        Link_Flow = pd.Series(self.m.Link_Flow.extract_values(), index=self.m.Link_Flow.extract_values().keys()).unstack()
        Link_Flow.index=list(self.t_dict.values())
        Link_Flow.to_csv(results_dir+'/Link_Flow.csv')

        Pipeline_Flow = pd.Series(self.m.Pipeline_Flow.extract_values(), index=self.m.Pipeline_Flow.extract_values().keys()).unstack()
        Pipeline_Flow.index=list(self.t_dict.values())
        Pipeline_Flow.to_csv(results_dir+'/Pipeline_Flow.csv')

        Storage_Level = pd.Series(self.m.Storage_Level.extract_values(), index=self.m.Storage_Level.extract_values().keys()).unstack()
        Storage_Level.index=pd.MultiIndex.from_tuples([(i[0],self.t_dict[i[1]]) for i in Storage_Level.index],names=Storage_Level.index.names)
        Storage_Level.to_csv(results_dir+'/Storage_Level.csv')

        Renewable_Production = pd.Series(self.m.Electricity_Potential.extract_values(), index=self.m.Electricity_Potential.extract_values().keys()).unstack()
        Renewable_Production = pd.DataFrame(Renewable_Production)
        Renewable_Production_raw = copy.copy(Renewable_Production)
        Renewable_Production_raw.index = pd.MultiIndex.from_tuples([(i[0],self.t_dict[i[1]]) for i in Renewable_Production_raw.index],names=Renewable_Production_raw.index.names)
        Renewable_Production_raw.to_csv(results_dir+'/Renewable_Production_raw.csv')
        Renewable_Production_Tech = Renewable_Production.groupby(level=0).sum()
        Renewable_Production = Renewable_Production.groupby(level=1).sum()
        Renewable_Production.index=list(self.t_dict.values())
        #Renewable_Production.columns = list(techs)
        Renewable_Production.to_csv(results_dir+'/Renewable_Production.csv')
        Renewable_Production_Tech.to_csv(results_dir+'/Renewable_Production_Tech_Total.csv')

        Renewable_Capacity = pd.Series(self.m.Gen_Capacity.extract_values(), index=self.m.Gen_Capacity.extract_values().keys()).unstack()
        Renewable_Capacity.to_csv(results_dir+'/Renewable_Capacity.csv')

        Storage_Charge = pd.Series(self.m.Storage_Charge.extract_values(), index=self.m.Storage_Charge.extract_values().keys()).unstack()
        Storage_Charge.index=pd.MultiIndex.from_tuples([(i[0],self.t_dict[i[1]]) for i in Storage_Charge.index],names=Storage_Charge.index.names)
        Storage_Charge.to_csv(results_dir+'/Storage_Discharge_Charge.csv')

        H2_Unserved = pd.Series(self.m.H2_Unserved.extract_values(), index=self.m.H2_Unserved.extract_values().keys()).unstack()
        H2_Unserved.index=list(self.t_dict.values())
        H2_Unserved.to_csv(results_dir+'/H2_Unserved.csv')

        Zone_Capacities.loc['H2_Unserved_Capacity']=H2_Unserved.max()
        Zone_Capacities.to_csv(results_dir+'/Zone_Capacities.csv')

        H2_Production = pd.Series(self.m.H2_Production.extract_values(), index=self.m.H2_Production.extract_values().keys()).unstack()
        H2_Production.index=list(self.t_dict.values())
        H2_Production.to_csv(results_dir+'/Hydrogen_Production.csv')

        Curtailed_Renewable_Production = pd.Series(self.m.Electricity_Curtailed.extract_values(), index=self.m.Electricity_Curtailed.extract_values().keys()).unstack()
        Curtailed_Renewable_Production = pd.DataFrame(Curtailed_Renewable_Production)
        Curtailed_Renewable_Production_raw = copy.copy(Curtailed_Renewable_Production)
        Curtailed_Renewable_Production_raw.index = pd.MultiIndex.from_tuples([(i[0],self.t_dict[i[1]]) for i in Curtailed_Renewable_Production_raw.index],names=Curtailed_Renewable_Production_raw.index.names)
        Curtailed_Renewable_Production_raw.to_csv(results_dir+'/Curtailed_Renewable_Production_raw.csv')
        Curtailed_Renewable_Production_Tech = Curtailed_Renewable_Production.groupby(level = 0).sum()
        Curtailed_Renewable_Production = Curtailed_Renewable_Production.groupby(level = 1).sum()
        Curtailed_Renewable_Production.index=list(self.t_dict.values())
        Curtailed_Renewable_Production.to_csv(results_dir+'/Curtailed_Renewable_Production.csv')
        Curtailed_Renewable_Production_Tech.to_csv(results_dir+'/Curtailed_Renewable_Production_Tech_Total.csv')

        Pipeline_Capacity = pd.Series(self.m.Pipeline_Capacity.extract_values(), index=self.m.Pipeline_Capacity.extract_values().keys())    
        Pipeline_Capacity.to_csv(results_dir+'/Pipeline_Capacity.csv')

        Demand_Met = pd.Series(self.m.H2_Demand_Met.extract_values(),index=self.m.H2_Demand_Met.extract_values().keys()).unstack()
        Demand_Met.index=list(self.t_dict.values())
        Demand_Met.to_csv(results_dir+'/Demand_Met.csv')

        H2_Imports = pd.Series(self.m.H2_Imports.extract_values(),index=self.m.H2_Imports.extract_values().keys()).unstack()
        H2_Imports.index = list(self.t_dict.values())
        H2_Imports.to_csv(results_dir+'/H2_Imports.csv')

        Truck_Flow = pd.Series(self.m.Truck_Flow.extract_values(),index=self.m.Truck_Flow.extract_values().keys()).unstack()
        Truck_Flow.index = list(self.t_dict.values())
        Truck_Flow.to_csv(results_dir+'/Truck_Flow.csv') 

        Truck_Cost = pd.Series(self.m.Truck_Cost.extract_values(),index=self.m.Truck_Cost.extract_values().keys()).unstack()
        Truck_Cost.index = list(self.t_dict.values())
        Truck_Cost.to_csv(results_dir+'/Truck_Cost.csv')    

        self.Pipeline_Exists.to_csv(results_dir+'/Pipeline_Exists.csv')


test = HYPSTAT(yaml_file_path='Case_Study/Case_Study_Scenario.yaml')
test.two_step_solve(solver='glpk')
test.write_outputs('Case_Study/Outputs/active_test')
print('Done!')

'''
GUIDE TO TEST CASES:

        test1: baseline, uses Nuclear in B to meet demand in B,C,and D; E and F self supply with grid (all truck transport)
        test2: more expensive trucking (uses pipelines instead of trucks to move H2)
        test3: back to normal truck costs, make nuclear more expensive (uses only grid, one pipeline link)
        test4: restrict grid in Zone B to force either Nuclear, RE, or transmission (moves grid to A and adds pipeline)
        test5: make nuclear cheaper, grid back to Zone B (uses Nuclear)
        test6: limit nuclear capacity to force combo of nuclear and some other technology (uses nuclear + grid)
        test7: add PTC to get some RE
        test8: limit grid in B to try to get something in Zone A
        test9: limit onshore wind in B to try to get something in Zone A
        test10: limit grid in A to see if wind/solar is hybridized (Not hybridized, but more interesting truck network)
        test11: updating geologic storage to see if it's used with capacity allowed (no more RE in Zone A cause storage allows for grid use in D)
        test12: wind back to Zone B
        test13: trying summer months to see if any solar is used (yes, in F to supply some of E)
        test14: recreate test 10
        test15: limit geologic storage but allow some; small amount of wind in B
        test16: add ITC for offshore wind to try to get to use, also adding 1% of CAPEX as fixed OPEX for pipelines
        test17: tuning ITC down for offshore wind to try to get some production back in A
        test18: tune ITC back up but limit offshore wind capacity
        test19: reduce solar costs and increase solar PTC to get some solar build
        '''