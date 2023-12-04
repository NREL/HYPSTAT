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


from pyomo.environ import *

from Model_Functions_v3 import *
from Model_Functions_v2_111423 import *


class HYPSTAT:
        
    def __init__(self, yaml_file_path='HYPSTAT_scenario.yaml'):
        self.yaml_file_path = yaml_file_path
        self.load_inputs()

    def load_inputs(self,optimize_pipelines=False,Pipeline_Exists=None):
        yaml_path = Path(self.yaml_file_path)
        with yaml_path.open('r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        
        self.year = yaml_data.get('studiedyear',None)
        self.firstYear = yaml_data.get('firstYear', None)
        self.lastYear = yaml_data.get('lastYear', None)
        self.timeWindow = yaml_data.get('timeWindow', None)
        self.scenario = yaml_data.get('scenario', None)
        self.truck_size_limit = yaml_data.get('truck_size_limit', None)

        self.interest = yaml_data.get('interest', None)
        self.payback_years = yaml_data.get('payback_years', None)
        self.payback_years_tank = yaml_data.get('payback_years_tank', None)
        self.payback_years_electrolyzer = yaml_data.get('payback_years_electrolyzer', None)

        self.max_imports_ratio = yaml_data.get('max_imports_ratio', None)
        self.import_zones = yaml_data.get('import_zones', None)
        print(self.import_zones)
        self.overserved_cost = yaml_data.get('hydrogen_overserved_cost', None)

        self.start = yaml_data.get('start', None)
        self.end = yaml_data.get('end', None)

        self.demandFiles_paths = yaml_data.get('demandFile', [])
        self.REcostFiles_paths = yaml_data.get('REcostFile', [])
        self.ProductioncostFiles_paths = yaml_data.get('ProductioncostFile', [])
        self.H2DeliveryFile_paths = yaml_data.get('H2DeliveryFile', [])
        self.H2StorageFile_paths = yaml_data.get('H2StorageFile', [])
        self.SupplycurveFolder_paths = yaml_data.get('SupplycurveFolder', [])
        self.Networks_paths = yaml_data.get('NetworksFiles', [])
        self.IncentiveFiles_paths = yaml_data.get('IncentiveFiles', [])
        self.StorageCapacityFiles_paths = yaml_data.get('StorageCapacityFiles', [])
        self.StorageLimitsFiles_paths = yaml_data.get('StorageLimitsFiles', [])

        self.REcostFiles = pd.concat([pd.read_csv(path) for path in self.REcostFiles_paths], ignore_index=True)
        self.year_tech_dict = self.create_year_tech_dict(self.REcostFiles)
        self.techs = self.year_tech_dict[self.year]

        self.H2StorageFile = pd.concat([pd.read_csv(path) for path in self.H2StorageFile_paths], ignore_index=True)
        self.year_stor_tech_dict = self.create_year_tech_dict(self.H2StorageFile)
        self.stor_techs = self.year_stor_tech_dict[self.year]

        self.all_renewable_profiles, self.capacities = get_renewable_profiles(year=self.year,techs = self.techs,path=self.SupplycurveFolder_paths[0],drop_capacity_below=False)
        self.demand = get_demand(self.all_renewable_profiles,year=self.year, daily_demand=True,freq='h', file_path=self.demandFiles_paths[0])
        self.link_flow_direction, self.links, self.all_zones, self.links_to_zones = get_links(self.Networks_paths[0])

        self.delivery_cost = pd.read_csv(self.H2DeliveryFile_paths[0])
        self.delivery_cost = self.delivery_cost[self.delivery_cost['Year'] == self.year]
        self.cost_dict = dict(zip(self.delivery_cost['Delivery Method'], self.delivery_cost['OPEX ($/kg)']))
        self.links['Transmission Opex ($/kg)'] = self.links['Delivery Method'].map(self.cost_dict) * \
                                                  self.links['Link Distance'] / 100
        
        #### WAYS TO GENEREALIZE truck and link distances 
        self.truck_cost = self.cost_dict['Truck']
        link_distance_path = 'Inputs/networks/link_distances.csv' #TODO: maybe incorporate into get_links function above?
        self.link_distances = pd.read_csv(link_distance_path,index_col='Zone')
        print(self.truck_cost)

        ######RE OPEX--manual input for now for testing; TODO: coordinate with Yijin's inputs
        self.re_opex = pd.DataFrame(data=0,index=self.techs,columns=self.all_zones)
        self.re_opex.loc['Grid'] = 0.2 #$/kWh (LCOE)

        self.build_cost = get_build_cost_matrix(self.interest,self.payback_years, self.payback_years_tank,self.payback_years_electrolyzer, self.REcostFiles_paths,self.ProductioncostFiles_paths,self.H2StorageFile_paths,self.year,self.all_zones)

        self.H2_prod_cost = pd.read_csv(self.ProductioncostFiles_paths[0])
        self.h2_conversion_efficiency = float(self.H2_prod_cost[self.H2_prod_cost['Year']==self.year]['efficiency(kWh/kg)'])

        self.RE_incentives = pd.read_csv(self.IncentiveFiles_paths[0])
        self.RE_incentives = self.RE_incentives[self.RE_incentives['Year'] == self.year]
        self.ITC = dict(zip(self.RE_incentives['Tech'], self.RE_incentives['ITC ($/kg)'].fillna(0)))
        self.PTC = dict(zip(self.RE_incentives['Tech'], self.RE_incentives['PTC ($/kg)'].fillna(0)))

        
        self.storage_capacities = pd.read_csv(self.StorageCapacityFiles_paths[0]).set_index('Zone')
        self.storage_capacities = self.storage_capacities.applymap(lambda x: 'inf' if str(x).strip("'") == 'inf' else pd.to_numeric(x, errors='coerce'))
        self.storage_capacities = {index: row.to_dict() for index, row in self.storage_capacities.iterrows()}

        self.H2_Storage_Limit = pd.read_csv(self.StorageLimitsFiles_paths[0])
        self.H2_Storage_Limit['capacity'] = self.H2_Storage_Limit['capacity'].astype(str)
        self.Total_storage_capacity = dict(zip(self.H2_Storage_Limit['Storage Tech'], self.H2_Storage_Limit['capacity'].str.strip("'").replace('inf', 'inf')))

        print(self.Total_storage_capacity)
        self.max_imports = self.demand.sum().sum() * self.max_imports_ratio
        self.Cost_of_Unserved_H2 = yaml_data.get('Cost_of_Unserved_H2', None)
        #self.Storage_charge_limit =yaml_data.get('Storage_charge_limit', None)
        self.Storage_charge_limit = dict(zip(self.H2_Storage_Limit['Storage Tech'],self.H2_Storage_Limit['charge limit']))
        #self.Storage_discharge_limit =yaml_data.get('Storage_discharge_limit', None)
        self.Storage_discharge_limit = dict(zip(self.H2_Storage_Limit['Storage Tech'],self.H2_Storage_Limit['discharge limit']))
        #self.Storage_charge_cost =yaml_data.get('Storage_charge_cost', None)
        self.Storage_charge_cost = dict(zip(self.H2_Storage_Limit['Storage Tech'],self.H2_Storage_Limit['charge_cost ($/kg)']))
        
        self.bigM = yaml_data.get('bigM',None)
        print(self.bigM)

        self.all_renewable_profiles = self.all_renewable_profiles.loc[self.start + str(self.year): self.end + str(self.year)]
        self.year_ratio = len(self.all_renewable_profiles.resample('d').first()) / 365
        self.max_imports *= self.year_ratio
        self.build_cost *= self.year_ratio

        if optimize_pipelines:
            #first iteration
            self.h=24 # this specifies the number of hours in each time interval.  It allows the conversion of kg/h when multiple hours are in each timestep.  if resampling data please change this value 
            print('Optimizing pipelines')
            print()
        else:
            #second iteration, hourly run
            self.h=2
            print('Performing full optimization')
            print()

        if Pipeline_Exists is not None:
            #eliminates certain links based on a previous round of optimization (or user input???)
            threshold = 0.1
            for link in Pipeline_Exists.index:
                if Pipeline_Exists[link] < threshold:
                    #close to 0, pipeline doesn't exist
                    self.links.loc[link,'Capacity (kg/hr)'] = 0
            print(self.links)

        if self.h != 1: #TODO: think about time resolution, for now leave as minimum 1 hour
            self.all_renewable_profiles = self.all_renewable_profiles.resample('{}h'.format(self.h)).sum()
            self.demand = self.demand.resample('{}h'.format(self.h)).sum()

        

    def create_year_tech_dict(self, REcostFiles):
        year_tech_dict = {}
        for year, group in REcostFiles.groupby('Year'):
            year_tech_dict[year] = list(group['Tech'])
        return year_tech_dict
    
    def load_model(self,optimize_pipelines=False):
        self.m = ConcreteModel()

        #Define time and node index sets
        self.t_dict = {ind + 1: k for ind, k in enumerate(self.all_renewable_profiles.index)}

        #setup model index sets
        self.m.T = Set(initialize=(sorted(list(self.t_dict.keys()))), ordered=True) # sort t_dict
        self.m.Zones = Set(initialize=list(self.all_zones), ordered=True)
        self.m.Links = Set(initialize=list(self.links.index), ordered=True)
        self.m.Techs = Set(initialize=list(self.techs), ordered=True) #TODO: rename for more specific to RE generators
        self.m.Stor_Techs = Set(initialize=list(self.stor_techs),ordered=True)
        
        # This set defined the index for the renewable profiles.
        self.m.Renewable_producers = Set(initialize=list(self.capacities.index), ordered=True) #TODO: rename to be more clear (e.g., tranches)
        print(list(self.capacities))

        '''
        TODO: SETS THAT ARE MISSING
        -H2 production techs
        -tank-based transmission techs

        -Pipelines? Maybe extra tranches
        '''


        #cost_parameters: #TODO: maybe just convert to input variables (question for Steven?)
        self.m.Cost_of_Unserved_H2 = Param(initialize=self.Cost_of_Unserved_H2) # We assume that a kg of unserved hydrogen has an economic cost of 10eX.
        self.m.h2_conversion_efficiency = Param(initialize=self.h2_conversion_efficiency) # kW/kg.h2
        #self.m.Cavern_charge_limit = Param(initialize=(0.03/24)*self.h) #% of capacity per hour, from 3%/day
        #self.m.Cavern_discharge_limit = Param(initialize=(0.1/24)*self.h) #% of capacity per hour, from 10%/day
        
        ##### HOW to deal with self.h
        
        #self.m.Storage_charge_limit = Param(self.m.Stor_Techs, initialize=self.Storage_charge_limit*self.h)
        #self.m.Storage_discharge_limit = Param(self.m.Stor_Techs, initialize=self.Storage_discharge_limit*self.h)
        self.m.Storage_charge_limit = Param(self.m.Stor_Techs, initialize={'Cavern':(0.03/24)*self.h,'Tank':'inf'})
        self.m.Storage_discharge_limit = Param(self.m.Stor_Techs, initialize={'Cavern':(0.1/24)*self.h,'Tank':'inf'})
        

        #self.m.Tank_charge_cost = Param(initialize=0.06) #$/kg, opex for loading H2 storage tank
        #self.m.Cavern_charge_cost = Param(initialize=0.06) #$/kg, opex for loading salt cavern storage
        self.m.Storage_charge_cost = Param(self.m.Stor_Techs,initialize={str(st): self.Storage_charge_cost[st] for st in self.m.Stor_Techs})
        print(self.Storage_charge_cost)
        #if self.Total_cavern_capacity != 'inf':
        #    self.m.Total_cavern_capacity = Param(initialize = self.Total_cavern_capacity) # kg
        #self.m.Total_storage_capacity = Param(self.m.Stor_Techs, initialize={st: self.Total_storage_capacity[st] for st in self.m.Stor_Techs})
        #self.m.Total_storage_capacity = Param(self.m.Stor_Techs, initialize=self.Total_storage_capacity)
        self.m.Total_storage_capacity = Param(self.m.Stor_Techs, initialize={str(st): self.Total_storage_capacity[st] for st in self.m.Stor_Techs})

        #Time varying variables

        #TODO: will need to add variable for specific flows of electricity to specific production types
        self.m.Storage_Discharge_Charge = Var(self.m.Stor_Techs,self.m.T,self.m.Zones)
        self.m.Storage_Level = Var(self.m.Stor_Techs,self.m.T,self.m.Zones,domain=NonNegativeReals)
        self.m.Storage_Capacity = Var(self.m.Stor_Techs,self.m.Zones,domain=NonNegativeReals)
        
        self.m.Link_Flow = Var(self.m.T, self.m.Links)  # (kg/per interval i.e. kg/h)
        self.m.Pipeline_Flow = Var(self.m.T, self.m.Links) #(kg/per interval, i.e., kg/h) Pipeline flow
        self.m.Truck_Flow = Var(self.m.T,self.m.Links)
        self.m.Electricity_Potential  = Var(self.m.Techs, self.m.T, self.m.Zones)  #kWh 
        self.m.Electricity_Used = Var(self.m.Techs, self.m.T, self.m.Zones) #kWh
        self.m.Renewable_Capacity = Var(self.m.Techs,self.m.Zones,self.m.Renewable_producers, domain=NonNegativeReals)  #kw
        self.m.Hydrogen_Production = Var(self.m.T, self.m.Zones, domain=NonNegativeReals)   #kg/per interval #TODO: split into different types of production
        self.m.H2_Unserved = Var(self.m.T, self.m.Zones, domain=NonNegativeReals)#,bounds=(0,0) )   #kg/per interval
        self.m.H2_Overserved = Var(self.m.T, self.m.Zones, domain=NonNegativeReals)#,bounds=(0,0) )  
        self.m.Electrolyser_Capacity = Var(self.m.Zones, domain=NonNegativeReals) #TODO: split into different types of production (maybe rename)
        self.m.Electricity_Curtailed = Var(self.m.Techs,self.m.T, self.m.Zones, domain=NonNegativeReals) 
        self.m.H2_Imports = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) # imported hydrogen, kg/interval

        self.m.Demand_Met = Var(self.m.T, self.m.Zones, domain=NonNegativeReals)# kg per interval #TODO: rename H2_demand_met or similar
        self.m.Pipeline_Cost = Var(self.m.T, self.m.Links, domain=NonNegativeReals) #$/kg #TODO: rename all of these to specify OPEX
        self.m.Truck_Cost = Var(self.m.T, self.m.Links, domain=NonNegativeReals) #$/kg #TODO: rename all of these to specify OPEX, TODO: generalize to all tank-based transmission
        
        self.m.Storage_Cost = Var(self.m.Stor_Techs,self.m.T,self.m.Zones, domain=NonNegativeReals)

        self.m.Pipeline_Capacity = Var(self.m.Links, domain = NonNegativeReals) # kg/period pipeline capacity for each zone
        if optimize_pipelines:
            self.m.Pipeline_Exists = Var(self.m.Links,domain=Binary) #whether or not a pipeline exists

    def load_constraints(self,optimize_pipelines=False):
        # pipeline flow incidence matrix
            def link_rule(m, link, zone):
                return float(self.link_flow_direction.loc[link, zone])

            self.m.link_flow_direction = Param(self.m.Links, self.m.Zones, initialize=link_rule)

            print('Setting up storage constraints...')

            # Meet demand over course of day

            def demand_rule(m, t, zone):
                if t%24==1:
                    return (sum(m.Demand_Met[t_step,zone] for t_step in range(t,t+24)) == sum(self.demand.loc[self.t_dict[t_step],zone] for t_step in range(t,t+24))) #TODO: May need to revise to consider 'h' value
                else:
                    return Constraint.Skip

            def demand_rule_2(m, t, zone):
                return (m.Demand_Met[t,zone] == self.demand.loc[self.t_dict[t],zone])

            #m.demand_constraint = Constraint(m.T, m.Zones, rule=demand_rule)
            self.m.demand_constraint = Constraint(self.m.T, self.m.Zones, rule=demand_rule_2)

            # Limit imports
            def total_imports_rule(m):
                return sum(sum(m.H2_Imports[t,zone] for t in m.T) for zone in m.Zones) <= self.max_imports
            
            self.m.total_imports_constraint = Constraint(rule=total_imports_rule)

            # Tabling imports for now; #TODO: think about exports
            def zone_imports_rule(m, t, zone):
                if zone in self.import_zones:
                    return Constraint.Skip
                else:
                    return m.H2_Imports[t,zone] == 0
            
            self.m.zone_imports_constraint = Constraint(self.m.T,self.m.Zones,rule=zone_imports_rule)

            # Storage Charge Discharge, multi-period;
            # TODO: clarify point within the hour that we mean for storage, transmission, etc.
            def Storage_mutli_period_rule(m, stor_tech, t, zone):
                if t == m.T.first():
                    return m.Storage_Level[stor_tech,t,zone] == m.Storage_Level[stor_tech,m.T.last(),zone] + m.Storage_Discharge_Charge[stor_tech,t,zone]
                else:
                    return (m.Storage_Level[stor_tech, t, zone] == m.Storage_Level[stor_tech, t-1, zone]  + m.Storage_Discharge_Charge[stor_tech, t, zone])

            self.m.Storage_multi_period_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_mutli_period_rule)

            # charge discharge speed
            def Storage_charge_rule(m, stor_tech, t, zone):
                if m.Storage_charge_limit[stor_tech]=='inf':
                    return Constraint.Skip
                    #Could also use just a really large number but probably leave as skip constraint
                else:
                    return (m.Storage_Discharge_Charge[stor_tech, t, zone] <= m.Storage_charge_limit[stor_tech]*m.Storage_Capacity[stor_tech, zone])
            
            self.m.Storage_charge_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_charge_rule)

            def Storage_discharge_rule(m, stor_tech, t, zone):
                if m.Storage_discharge_limit[stor_tech]=='inf':
                    return Constraint.Skip
                else:
                    return (m.Storage_Discharge_Charge[stor_tech, t, zone] >= -m.Storage_discharge_limit[stor_tech]*m.Storage_Capacity[stor_tech, zone])

            self.m.Storage_discharge_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_discharge_rule)

            # Storage Level
            # TODO: note in documentation that storage capacities need to be working capacities

            def Storage_level_rule(m, stor_tech, t, zone):
                return (m.Storage_Level[stor_tech, t, zone] <= m.Storage_Capacity[stor_tech, zone])

            self.m.Storage_level_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_level_rule)

            #TODO: consolidate all of this into some sort of input table with limits for each storage type and zone
            #TODO: consider if we want to still have a total storage constraint

            def Storage_capacity_cap_rule(m, stor_tech, zone):
                cap = self.storage_capacities[stor_tech][zone]
                if cap=='inf':
                    return Constraint.Skip
                else:
                    return (m.Storage_Capacity[stor_tech, zone] <= cap)

            self.m.Storage_capacity_cap_constraint = Constraint(self.m.Stor_Techs, self.m.Zones, rule=Storage_capacity_cap_rule)


            def Total_storage_capacity_rule(m, stor_tech):
                if m.Total_storage_capacity[stor_tech]=='inf':
                    return Constraint.Skip
                else:
                    return (sum(m.Storage_Capacity[stor_tech, zone] for zone in m.Zones) <= m.Total_storage_capacity[stor_tech])

            self.m.Total_storage_capacity_constraint = Constraint(self.m.Stor_Techs, rule=Total_storage_capacity_rule)
            
            print('Setting up flow constraints...')
            # Pipeline flow (update this later to be part of the variable bound)

            #TODO: have upper bound on pipeline size as an input
            if optimize_pipelines:
                def pipeline_exists_rule(m, link):
                    bigM = self.bigM *self.h #kg/period, upper bound on pipeline capacity (~7000 TPD)
                    return m.Pipeline_Capacity[link] <= bigM*m.Pipeline_Exists[link]
                
                self.m.pipeline_exists_constraint = Constraint(self.m.Links,rule=pipeline_exists_rule)
            
            else:
                #not optimizing pipelines, limit sizes according to links

                #TODO: think about how pipeline capacity limits (i.e., which pipelines exist) is passed in the second iteration

                def pipeline_size_rule(m, link):
                    return m.Pipeline_Capacity[link] <= self.links.loc[link,'Capacity (kg/hr)']*self.h
                
                self.m.pipeline_size_constraint = Constraint(self.m.Links, rule=pipeline_size_rule)

            # Total link flow
            def total_link_flow_rule(m, t, link):
                return m.Link_Flow[t, link] == m.Pipeline_Flow[t, link] + m.Truck_Flow[t, link]

            self.m.total_link_flow_constraint = Constraint(self.m.T, self.m.Links, rule=total_link_flow_rule)
            
            # TODO: generalize into tank-based transmission
            # TODO: think about if we want to incorporate capex/fixed opex or keep as levelized cost (via opex)
            def forward_max_truck_rule(m, t, link):
                return m.Truck_Flow[t,link] <= self.truck_size_limit*self.h #kg/period
            
            self.m.forward_max_truck_constraint = Constraint(self.m.T, self.m.Links, rule=forward_max_truck_rule)

            def reverse_max_truck_rule(m, t, link):
                return m.Truck_Flow[t,link] >= -self.truck_size_limit*self.h #kg/period
            
            self.m.reverse_max_truck_constraint = Constraint(self.m.T, self.m.Links, rule=reverse_max_truck_rule)

            def pipeline_forward_flow_rule(m, t, link):
                return m.Pipeline_Flow[t, link] <= m.Pipeline_Capacity[link]
                
            self.m.pipeline_forward_flow_constraint = Constraint(self.m.T, self.m.Links, rule=pipeline_forward_flow_rule)

            def pipeline_reverse_flow_rule(m, t, link):
                return m.Pipeline_Flow[t, link] >= -1*m.Pipeline_Capacity[link]

            self.m.pipeline_reverse_flow_constraint = Constraint(self.m.T, self.m.Links, rule=pipeline_reverse_flow_rule)

            print('Setting up renewable zone constraints...')
            # Renewable production - this defines the renewable production at each node  in kwh
            def Renewable_Production_rule(m, tech, t, zone):
                return m.Electricity_Potential[tech, t, zone] == sum(self.all_renewable_profiles.loc[self.t_dict[t],(tech,zone,producer)] * m.Renewable_Capacity[tech,zone,producer]  for producer in get_producers(self.capacities,zone=zone,tech = tech).index )

            self.m.Electricity_Potential_constraint = Constraint(self.m.Techs, self.m.T, self.m.Zones, rule=Renewable_Production_rule)

            # Renewable build limit  - sets the capacity of each REZ to be less than a value 
            def Renewable_build_limit_rule(m, tech, zone, producer):
                capacity = get_producers(self.capacities,zone=zone,tech=tech)
                if producer in capacity.index:
                    if not capacity.loc[producer]==capacity.loc[producer]:

                        capacity.loc[producer] = 0
                    return  (m.Renewable_Capacity[tech,zone,producer] <= capacity.loc[producer] * 1000) #TODO: specify units in inputs to avoid this 1000
                else:
                    return Constraint.Skip

            self.m.Renewable_build_limit_constraint = Constraint(self.m.Techs, self.m.Zones, self.m.Renewable_producers, rule=Renewable_build_limit_rule)

            # TODO: recast curtailment and electrolyzer capacity based on generalized electricity/production flows
            def Curtailed_electricity_limit_rule(m, tech, t, zone):
                return m.Electricity_Curtailed[tech, t, zone] <= m.Electricity_Potential[tech, t, zone]
            
            self.m.Curtailed_electricity_limit_constraint = Constraint(self.m.Techs, self.m.T, self.m.Zones, rule=Curtailed_electricity_limit_rule) #can't curtailed more electricity than is produced

            def Electricity_Used_rule(m, tech, t, zone):
                return (m.Electricity_Used[tech, t, zone] == m.Electricity_Potential[tech, t, zone] - m.Electricity_Curtailed[tech, t, zone])

            self.m.Electricity_Used_constraint = Constraint(self.m.Techs, self.m.T, self.m.Zones, rule=Electricity_Used_rule)

            # cap renewable production at node to electrolyser capacity
            #TODO: rename this for clarity (could recast for easier understanding but probably not necessary)
            def electrolyser_capacity_limit_rule(m, t, zone):
                return (sum(m.Electricity_Used[tech, t, zone] for tech in m.Techs)) <= m.Electrolyser_Capacity[zone]*self.h

            self.m.electrolyser_capacity_limit_constraint = Constraint(self.m.T, self.m.Zones, rule=electrolyser_capacity_limit_rule)

            # New H2 production build at nodes
            # electricity is in kWh
            # TODO: generalize to various H2 production pathways (focused on electrolysis)
                # TODO: will need to think about how renewable production matches up with H2 production
                # TODO: create variables for specification of electricity flows from each RE tech to each H2 production tech
            def H2_Production_rule(m, t, zone):
                return m.Hydrogen_Production[t, zone] == (sum(m.Electricity_Used[tech, t, zone] for tech in m.Techs))/m.h2_conversion_efficiency

            self.m.H2_Production_constraint = Constraint(self.m.T, self.m.Zones, rule=H2_Production_rule)

            print('Setting up power balance constraints...')
            # H2 balance considering link flows
            def H2_Balance_rule(m, t, zone):
                links_to_this_zone=self.links_to_zones[zone]
                return (m.Hydrogen_Production[t, zone] + m.H2_Unserved[t, zone] + sum(m.Link_Flow[t, link] * m.link_flow_direction[link, zone] for link in links_to_this_zone) + m.H2_Imports[t,zone] == \
                        sum(m.Storage_Discharge_Charge[stor_tech, t, zone] for stor_tech in m.Stor_Techs) + m.Demand_Met[t, zone] + m.H2_Overserved[t, zone] )

            self.m.H2_Balance_constraint = Constraint(self.m.T, self.m.Zones, rule=H2_Balance_rule)

            print('Setting up transmission cost constraints...')
            #TODO: work out in inputs somewhere how to add multipliers for links or set link costs in some specific way (or disallow links)
            def Forward_Pipeline_Cost_rule(m, t, link):
                return (m.Pipeline_Cost[t,link] >= m.Pipeline_Flow[t,link]*self.links.loc[link,'Transmission Opex ($/kg)'])

            self.m.Forward_Pipeline_Cost_constraint = Constraint(self.m.T, self.m.Links, rule=Forward_Pipeline_Cost_rule)

            def Reverse_Pipeline_Cost_rule(m, t, link):
                return (m.Pipeline_Cost[t,link] >= -m.Pipeline_Flow[t,link]*self.links.loc[link,'Transmission Opex ($/kg)'])

            self.m.Reverse_Pipeline_Cost_constraint = Constraint(self.m.T, self.m.Links, rule=Reverse_Pipeline_Cost_rule)

            def Forward_Truck_Cost_rule(m, t, link): #TODO: generalize to all tank transport
                return (m.Truck_Cost[t,link] >= m.Truck_Flow[t,link]*self.truck_cost*(self.link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]/100))
            
            self.m.Forward_Truck_Cost_constraint = Constraint(self.m.T, self.m.Links, rule=Forward_Truck_Cost_rule)

            def Reverse_Truck_Cost_rule(m, t, link):
                return (m.Truck_Cost[t,link] >= -m.Truck_Flow[t,link]*self.truck_cost*(self.link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]/100))
            
            self.m.Reverse_Truck_Cost_constraint = Constraint(self.m.T, self.m.Links, rule=Reverse_Truck_Cost_rule)

            print('Setting up storage cost constraints...')
            # TODO: allow for variable opex on discharge if desired by user

            def Storage_Cost_rule(m, stor_tech, t, zone):
                return (m.Storage_Cost[stor_tech,t,zone] >= m.Storage_Discharge_Charge[stor_tech,t,zone]*m.Storage_charge_cost[stor_tech])

            self.m.Storage_Cost_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Zones, rule=Storage_Cost_rule)


            print('Setting up objective function...')
            #### Objective function

            def get_CRF(interest=self.interest, years=self.payback_years):
                return (interest * (1 + interest) ** years) /((1 + interest) ** years - 1)
            '''
            def objective_rule(m):
                return (sum(
                        sum(m.Storage_Capacity[stor_tech, zone] * self.build_cost.loc['{}_Storage'.format(stor_tech),zone] for stor_tech in m.Stor_Techs) + #cost of storage build
                        m.Electrolyser_Capacity[zone] * self.build_cost.loc['PEM_Electrolyser',zone] + #cost of electrolyser TODO: think about how to reference costs with inputs, generalize for H2 production
                        sum(m.H2_Unserved[t, zone] * m.Cost_of_Unserved_H2 for t in m.T) + #penilty for unserved H2 TODO: think about inputs for overserved/underserved penalties
                        sum(m.H2_Overserved[t, zone] * self.overserved_cost for t in m.T) for zone in m.Zones) + #penilty for overserved H2
                        sum(sum(m.Pipeline_Cost[t,link] for t in m.T) for link in m.Links) + #opex for pipelines
                        sum(sum(m.Truck_Cost[t,link] for t in m.T) for link in m.Links) + #levelized cost for trucks
                        #TODO: consider variable opex for generation techs (e.g., to use LCOE)
                        #TODO: consider variable opex for hydrogen production techs
                        #TODO: consider CAPEX for tank-based based
                        #TODO: convert pipeline costs into inputs
                        sum((((m.Pipeline_Capacity[link]/self.h)*18.86 + 2122612*(m.Pipeline_Exists[link] if optimize_pipelines else 1))*self.year_ratio*self.link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]) for link in m.Links)*get_CRF() + #TODO: take CRF out of this and make sure it is inputs, like for all other techs
                        sum(sum(sum(m.Storage_Cost[stor_tech,t,zone] for t in m.T) for zone in m.Zones) for stor_tech in m.Stor_Techs) +
                        sum(sum(sum(m.Renewable_Capacity[tech,zone,producer] * self.build_cost.loc[tech,zone] for tech in m.Techs) for zone in m.Zones) for producer in m.Renewable_producers)  + # renewable build

                        #electricity opex
                        sum(sum(sum((m.Electricity_Used[tech, t, zone])*self.re_opex.loc[tech,zone] for t in m.T) for tech in m.Techs) for zone in m.Zones) +
                        # could loop through just grid connected zones or something like that

                        #TODO: handle PTC/ITC in inputs
                        -sum(sum(sum((m.Electricity_Used[tech, t, zone])*self.PTC[tech] for zone in m.Zones) for t in m.T) for tech in m.Techs) + #PTC effects
                        -sum(sum(sum((m.Electricity_Potential[tech,t,zone])*self.ITC[tech] for zone in m.Zones) for t in m.T) for tech in m.Techs) #ITC effects
                )
                        
                #log_infeasible_constraints(m)
            '''
            
            def objective_rule(m):
                return (sum(
                        sum(m.Storage_Capacity[stor_tech, zone] * self.build_cost.loc['{}'.format(stor_tech),zone] for stor_tech in m.Stor_Techs) + #cost of storage build
                        m.Electrolyser_Capacity[zone] * self.build_cost.loc['PEM Electrolyzer',zone] + #cost of electrolyser TODO: think about how to reference costs with inputs, generalize for H2 production
                        sum(m.H2_Unserved[t, zone] * m.Cost_of_Unserved_H2 for t in m.T) + #penilty for unserved H2 TODO: think about inputs for overserved/underserved penalties
                        sum(m.H2_Overserved[t, zone] * self.overserved_cost for t in m.T) for zone in m.Zones) + #penilty for overserved H2
                        sum(sum(m.Pipeline_Cost[t,link] for t in m.T) for link in m.Links) + #opex for pipelines
                        sum(sum(m.Truck_Cost[t,link] for t in m.T) for link in m.Links) + #levelized cost for trucks
                        #TODO: consider variable opex for generation techs (e.g., to use LCOE)
                        #TODO: consider variable opex for hydrogen production techs
                        #TODO: consider CAPEX for tank-based based
                        #TODO: convert pipeline costs into inputs
                        sum((((m.Pipeline_Capacity[link]/self.h)*18.86 + 2122612*(m.Pipeline_Exists[link] if optimize_pipelines else 1))*self.year_ratio*self.link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]) for link in m.Links)*get_CRF() + #TODO: take CRF out of this and make sure it is inputs, like for all other techs
                        sum(sum(sum(m.Storage_Cost[stor_tech,t,zone] for t in m.T) for zone in m.Zones) for stor_tech in m.Stor_Techs) +
                        sum(sum(sum(m.Renewable_Capacity[tech,zone,producer] * self.build_cost.loc[tech,zone] for tech in m.Techs) for zone in m.Zones) for producer in m.Renewable_producers)  + # renewable build

                        #electricity opex
                        sum(sum(sum((m.Electricity_Used[tech, t, zone])*self.re_opex.loc[tech,zone] for t in m.T) for tech in m.Techs) for zone in m.Zones) +
                        # could loop through just grid connected zones or something like that

                        #TODO: handle PTC/ITC in inputs
                        -sum(sum(sum((m.Electricity_Used[tech, t, zone])*self.PTC[tech] for zone in m.Zones) for t in m.T) for tech in m.Techs) + #PTC effects
                        -sum(sum(sum((m.Electricity_Potential[tech,t,zone])*self.ITC[tech] for zone in m.Zones) for t in m.T) for tech in m.Techs) #ITC effects
                )
                        
                #log_infeasible_constraints(m)
            self.m.objective = Objective(rule=objective_rule, sense=minimize)
            
    def solve_model(self,optimize_pipelines=False):
        stream_solver=True
        print('Solving')
        #opt = SolverFactory("gurobi", solver_io="python")
        opt = SolverFactory('glpk')
        results_final = opt.solve(self.m, tee=stream_solver)
        
        if optimize_pipelines:
            self.Pipeline_Exists = pd.Series(self.m.Pipeline_Exists.extract_values(), index=self.m.Pipeline_Exists.extract_values().keys())
    
    def two_step_solve(self):

        start = time.time()

        #solve first, optimizing for pipelines
        self.load_inputs(optimize_pipelines=True)
        self.load_model(optimize_pipelines=True)
        self.load_constraints(optimize_pipelines=True)
        self.solve_model(optimize_pipelines=True)

        mid = time.time()

        #solve second, not optimizing
        self.load_inputs(optimize_pipelines=False,Pipeline_Exists=self.Pipeline_Exists)
        self.load_model(optimize_pipelines=False)
        self.load_constraints(optimize_pipelines=False)
        self.solve_model(optimize_pipelines=False)

        stop = time.time()

        print()
        print('### MODEL SOLVE COMPLETE ###')
        print('Initial solve took {} seconds; full solve took {} seconds. Total time: {} seconds'.format(mid-start,stop-mid,stop-start))
        print()

    def write_outputs(self,results_dir):

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        else:
            con = input("results_dir '{}' already exists. Do you want to overwrite? (y/n)")
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
        Renewable_Capacity = pd.Series(self.m.Renewable_Capacity.extract_values(), index=self.m.Renewable_Capacity.extract_values().keys()).unstack()
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

        Renewable_Capacity = pd.Series(self.m.Renewable_Capacity.extract_values(), index=self.m.Renewable_Capacity.extract_values().keys()).unstack()
        Renewable_Capacity.to_csv(results_dir+'/Renewable_Capacity.csv')

        Storage_Discharge_Charge = pd.Series(self.m.Storage_Discharge_Charge.extract_values(), index=self.m.Storage_Discharge_Charge.extract_values().keys()).unstack()
        Storage_Discharge_Charge.index=pd.MultiIndex.from_tuples([(i[0],self.t_dict[i[1]]) for i in Storage_Discharge_Charge.index],names=Storage_Discharge_Charge.index.names)
        Storage_Discharge_Charge.to_csv(results_dir+'/Storage_Discharge_Charge.csv')

        H2_Unserved = pd.Series(self.m.H2_Unserved.extract_values(), index=self.m.H2_Unserved.extract_values().keys()).unstack()
        H2_Unserved.index=list(self.t_dict.values())
        H2_Unserved.to_csv(results_dir+'/H2_Unserved.csv')

        Zone_Capacities.loc['H2_Unserved_Capacity']=H2_Unserved.max()
        Zone_Capacities.to_csv(results_dir+'/Zone_Capacities.csv')

        Hydrogen_Production = pd.Series(self.m.Hydrogen_Production.extract_values(), index=self.m.Hydrogen_Production.extract_values().keys()).unstack()
        Hydrogen_Production.index=list(self.t_dict.values())
        Hydrogen_Production.to_csv(results_dir+'/H2_Production.csv')

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

        Demand_Met = pd.Series(self.m.Demand_Met.extract_values(),index=self.m.Demand_Met.extract_values().keys()).unstack()
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


test = HYPSTAT()
test.two_step_solve()
test.write_outputs('Test Cases/test_case_inputs')
print('Done!')

'''
GUIDE TO TEST CASES:

For comparison: obj_test_case, test_case_correct_results, test_case_outputs (should all match)
                test_case_no_imports is a version with no imports (vs. 1/4 imports in test case)

                obj_test_case and test_case_no_imports have the objective function value so better for comparison

NOTE: everything before test_case_gen_stor (and including test_case_gen_stor_cavern_error) has an error in the MB that was fixed with test_case_gen_stor! So test_case_gen_stor should be new point of comparison

Test cases:
        test_case_pipelines: version of the model with specific variable created for pipelines...should have extra output for pipeline flow, but other than that outputs should match
        test_case_pipelines_no_imports: version of the no_imports test case with pipelines
        test_case_cons_mb: consolidated mass balance (should match the test case with imports)
        test_case_cons_mb2: consolidated mass balance with removal of extraneous variables
        test_case_cons_mb_FIXED: consolidated mass balance with storage discharge/charge error fixed
        test_case_gen_stor: cavern and H2 storage converted into generic storage (1 variable) with index/set for storage types
            Note: orders/names have changed but zone builds in this are the same as test_case_cons_mb_FIXED.
            CAN BE USED AS POINT OF COMPARISON NOW
        test_case_gen_stor_cavern_error: same as gen_stor, but with an artificial error using cavern storage discharge charge in the first hour storage constraint to mimic old test cases
        test_case_gen_stor2: same as test_case_gen_stor with some extraneous/commented code removed for cleaning. Results should be identical.
        test_case_re_opex: testing the adding of opex to RE, i.e., for grid electricity. This test has grid electricity at $1000/kWh, so no grid electricity should be used and results should be identical to previous case.
        test_case_grid: testing using RE opex to have a reasonable grid price ($0.2/kWh) (for competetion between grid and RE). SHOULD NOT HAVE COMPARABLE RESULTS TO BEFORE, but obj value should be less than before
        test_case_grid2: same as above but with cheaper electricity so that it is used ($0.03/kWh)
        test_case_grid3: same as above, with mid-price electricity ($0.05/kWh)
        test_case_recast_elec_prod: same as test_case_grid3, but with electricity recast to include a specific variable for amount of used electricity (more intuitive)
        test_case_recast_elec_prod2: same as test_case_grid but with electricity recast as above (but should build no grid electricity)
        test_case_inputs: Yijin made test case with new inputs generalized and replaced
        '''