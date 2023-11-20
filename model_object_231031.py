'''
Authors: Joe Brauch, Yijin Li, Steven Percy

This file contains the model object for the HYPSTAT model
'''

import time
import os
import pandas as pd
import numpy as np


from pyomo.environ import *

from Model_Functions_v3 import *


class HYPSTAT:
    def __init__(self):
        pass

    def load_inputs(self,optimize_pipelines=False,Pipeline_Exists=None):
        techs = ['Terrestrial_Wind','Offshore_Wind','Solar'] 

        self.year = 2050 #dummy line

        self.techs = ['Terrestrial_Wind','Offshore_Wind','Solar']  #TODO: convert to inputs (booleans for different tech types or flexible?)

        supply_path = '../Test case data/Supply profiles'
        self.all_renewable_profiles,self.capacities=get_renewable_profiles(year=self.year,techs=self.techs, drop_capacity_below=False,path=supply_path)

        demand_path = '../Test case data/Demand profiles/Mid_Demand_demand.tsv' #TODO: convert to csv, not tsv
        self.demand=get_demand(self.all_renewable_profiles,year=self.year, daily_demand=True, path=demand_path)

        links_path = '../Test case data/unlimited_links.csv'
        self.link_flow_direction, self.links, self.all_zones,self.links_to_zones=get_links(links_path)

        #add column in links for operating cost

        #right now, set all to $0.06/kg

        comp_cost = 0.06 #$/kg/100-mi of transmission
        self.truck_cost = 1.25 #$/kg/100-mi of truck transmission
        self.truck_size_limit = 100*1000/24 #kg/hr

        link_distance_path = '../Test case data/link_distances.csv' #TODO: maybe incorporate into get_links function above?
        self.link_distances = pd.read_csv(link_distance_path,index_col='Zone')

        link_opex = [comp_cost*(self.link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]/100) for link in self.links.index]
        #add in $1.25/kg per 100-mi
        self.links.insert(4,'Transmission Opex ($/kg)',link_opex)


        for zone in self.all_zones:
            if zone not in self.links_to_zones:
                self.links_to_zones[zone] = set()

        self.h2_conversion_efficiency=51
        self.build_cost=get_build_cost_matrix(year=self.year,file='Build_Cost_Inputs_elec_cost_conservative.csv',all_zones=self.all_zones,includ_interconnection_cost = True)

        self.PTC = dict()
        for tech in techs:
            if tech == 'Offshore_Wind':
                self.PTC[tech] = 0 #0.04
            else:
                self.PTC[tech] = 0.024

        self.ITC = {
            'Terrestrial_Wind': 0,
            'Offshore_Wind': 0.04,
            'Solar': 0
        }

        self.cavern_capacities = dict()
        for zone in self.all_zones:
            self.cavern_capacities[zone] = 0

        self.tank_capacities = dict()
        for zone in self.all_zones:
            self.tank_capacities[zone] = 'inf'

        #disallow tank storage in HI and J
        self.tank_capacities['HI'] = 0
        self.tank_capacities['J'] = 0

        #cavern_capacities['CS'] = 'inf'
        self.cavern_capacities['CS'] = 8000000
        #cavern_capacities['A'] = 'inf'
        self.Total_cavern_capacity = 'inf'
        #Total_cavern_capacity = 16000000

        #import controls
        self.max_imports = 0 #self.demand.sum().sum()/4
        self.import_zones = {'F'} #zones which can import hydrogen

        self.overserved_cost = 5 #$/kg

        #Time controls
        self.all_renewable_profiles=self.all_renewable_profiles.loc['1 Jan '+str(self.year): '7 Jan '+str(self.year)]

        self.year_ratio=len((self.all_renewable_profiles).resample('d').first())/365 # used to estimate total costs for year. Just used for testing. 
        self.max_imports *= self.year_ratio

        self.build_cost=self.build_cost*self.year_ratio

        # Controls for iteration with or without explicit pipeline optimization

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

        # OLD NUCLEAR CONTROLS - TO BE REMOVED #
        """ ## nuclear hydrogen inputs:
        include_nuclear_hydrogen=False
        nuclear_hydrogen_years=[2050]
        nuclear_zones=['CN']
        #capacity_factor=0.890507
        maximum_nuclear_capacity=1255.8 * 1000 #KW
        Nuclear_LCOE=30 #$/MWh
        Nuclear_LCOE=Nuclear_LCOE/1000 #$/kWh
        nuclear_h2_conversion_efficiency = 35.7 """
        

    def load_model(self,optimize_pipelines=False):
        self.m = ConcreteModel()

        #Define time and node index sets
        self.t_dict = {ind + 1: k for ind, k in enumerate(self.all_renewable_profiles.index)}

        #setup model index sets
        self.m.T = Set(initialize=(sorted(list(self.t_dict.keys()))), ordered=True) # sort t_dict
        self.m.Zones = Set(initialize=list(self.all_zones), ordered=True)
        self.m.Links = Set(initialize=list(self.links.index), ordered=True)
        self.m.Techs = Set(initialize=list(self.techs), ordered=True) #TODO: rename for more specific to RE generators
        
        # This set defined the index for the renewable profiles.
        self.m.Renewable_producers = Set(initialize=list(self.capacities.index), ordered=True) #TODO: rename to be more clear (e.g., tranches)

        '''
        TODO: SETS THAT ARE MISSING
        -H2 production techs
        -storage techs
        -tank-based transmission techs

        -Pipelines? Maybe extra tranches
        '''


        #cost_parameters: #TODO: maybe just convert to input variables (question for Steven?)
        self.m.Cost_of_Unserved_H2 = Param(initialize=10e3) # We assume that a kg of unserved hydrogen has an economic cost of 10eX.
        self.m.h2_conversion_efficiency = Param(initialize=self.h2_conversion_efficiency) # kW/kg.h2
        self.m.Cavern_charge_limit = Param(initialize=(0.03/24)*self.h) #% of capacity per hour, from 3%/day
        self.m.Cavern_discharge_limit = Param(initialize=(0.1/24)*self.h) #% of capacity per hour, from 10%/day
        self.m.Tank_charge_cost = Param(initialize=0.06) #$/kg, opex for loading H2 storage tank
        self.m.Cavern_charge_cost = Param(initialize=0.06) #$/kg, opex for loading salt cavern storage
        if self.Total_cavern_capacity != 'inf':
            self.m.Total_cavern_capacity = Param(initialize = self.Total_cavern_capacity) # kg


        #Time varying variables

        #TODO: will need to add variable for specific flows of electricity to specific production types
        self.m.Zone_H2_Demand_Storage = Var(self.m.T, self.m.Zones)#, domain=NonNegativeReals)  #kg/per interval i.e. kg/h #TODO: (?) eliminate this variable and consolodate into mass balance
        self.m.H2_Storage_Discharge_Charge = Var(self.m.T, self.m.Zones) # can be positive or negative  (kg/per interval i.e. kg/h)
        self.m.Cavern_Storage_Discharge_Charge = Var(self.m.T, self.m.Zones) # can be positive or negative  (kg/per interval i.e. kg/h) #TODO: consolodate into single variable with additional index for storage tech
        self.m.H2_Storage_Level = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) #(kg) 
        self.m.Cavern_Storage_Level = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) #(kg) #TODO: consolodate into single variable with additional index for storage tech
        self.m.H2_Storage_Capacity = Var(self.m.Zones, domain=NonNegativeReals)  #(kg)
        self.m.Cavern_Storage_Capacity = Var(self.m.Zones, domain=NonNegativeReals) #(kg) #TODO: consolodate into single variable with additional index for storage tech
        self.m.Link_Flow = Var(self.m.T, self.m.Links)  # (kg/per interval i.e. kg/h) #TODO: create a variable for pipeline flow (maybe remove Link_Flow)
        self.m.Pipeline_Flow = Var(self.m.T, self.m.Links) #(kg/per interval, i.e., kg/h) Pipeline flow
        self.m.Truck_Flow = Var(self.m.T,self.m.Links)
        self.m.Renewable_Production  = Var(self.m.Techs, self.m.T, self.m.Zones)  #kWh 
        self.m.Renewable_Capacity = Var(self.m.Techs,self.m.Zones,self.m.Renewable_producers, domain=NonNegativeReals)  #kw
        self.m.Hydrogen_Production = Var(self.m.T, self.m.Zones, domain=NonNegativeReals)   #kg/per interval #TODO: split into different types of production
        self.m.H2_Unserved = Var(self.m.T, self.m.Zones, domain=NonNegativeReals)#,bounds=(0,0) )   #kg/per interval
        self.m.H2_Overserved = Var(self.m.T, self.m.Zones, domain=NonNegativeReals)#,bounds=(0,0) )  
        self.m.Electrolyser_Capacity = Var(self.m.Zones, domain=NonNegativeReals) #TODO: split into different types of production (maybe rename)
        self.m.Curtailed_Renewable_Production = Var(self.m.Techs,self.m.T, self.m.Zones, domain=NonNegativeReals) 
        self.m.H2_Imports = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) # imported hydrogen, kg/interval

        self.m.Demand_Met = Var(self.m.T, self.m.Zones, domain=NonNegativeReals)# kg per interval #TODO: rename H2_demand_met or similar
        self.m.Pipeline_Cost = Var(self.m.T, self.m.Links, domain=NonNegativeReals) #$/kg #TODO: rename all of these to specify OPEX
        self.m.Truck_Cost = Var(self.m.T, self.m.Links, domain=NonNegativeReals) #$/kg #TODO: rename all of these to specify OPEX, TODO: generalize to all tank-based transmission
        self.m.H2_Storage_Cost = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) #$/kg for charging storage #TODO: rename all of these to specify OPEX
        self.m.Cavern_Storage_Cost = Var(self.m.T, self.m.Zones, domain=NonNegativeReals) #$/kg for charging storage #TODO: rename all of these to specify OPEX, TODO: consolodate storage into single variable with storage tech index set

        self.m.Pipeline_Capacity = Var(self.m.Links, domain = NonNegativeReals) # kg/period pipeline capacity for each zone
        if optimize_pipelines:
            self.m.Pipeline_Exists = Var(self.m.Links,domain=Binary) #whether or not a pipeline exists

        # OLD NUCLEAR CONTROLS - TO BE REMOVED #
        '''
        self.m.Nuclear_Zones = Set(initialize=list(nuclear_zones), ordered=True) #TODO: delete nuclear zones (set up as an input as part of different tech types)
        self.m.nuclear_h2_conversion_efficiency = Param(initialize=nuclear_h2_conversion_efficiency) # kW/kg.h2
        self.m.Nuclear_LCOE = Param(initialize=Nuclear_LCOE) #$/Kwh

        if include_nuclear_hydrogen  & (year in nuclear_hydrogen_years): #TODO: remove nuclear technology
            nuclear_profiles = all_renewable_profiles['Nuclear']

        #hourly capacity factor 
        self.m.Nuclear_Production  = Var(self.m.T, self.m.Nuclear_Zones)  #kWh 
        #m.Nuclear_Capacity = Var(m.Techs,m.Nuclear_Zones, domain=NonNegativeReals,bounds=(0, maximum_nuclear_capacity)) # Bound to the maximum nuclear capacity
        self.m.Nuclear_H2_Electrolyser_Capacity = Var(self.m.Nuclear_Zones, domain=NonNegativeReals, bounds=(0, maximum_nuclear_capacity)) # Bound to the maximum nuclear capacity
        self.m.Nuclear_H2_Production= Var(self.m.T, self.m.Nuclear_Zones, domain=NonNegativeReals) #kg
        '''

    def load_constraints(self,optimize_pipelines=False):
        # pipeline flow incidence matrix
            def link_rule(m, link, zone):
                return float(self.link_flow_direction.loc[link, zone])

            self.m.link_flow_direction = Param(self.m.Links, self.m.Zones, initialize=link_rule)

            print('Setting up storage constraints...')
            ## Storage_demand; adding in imports here
            ## A negative value of H2_Storage_Discharge_Charge is equiv to production; 
            # TODO: choose either fixed demand or Demand_Met variable and consolodate to 1 storage and demand rule
            def storage_demand_rule(m, t, zone):
                    return ( m.Zone_H2_Demand_Storage[t,zone] == self.demand.loc[self.t_dict[t], zone] + m.H2_Storage_Discharge_Charge[t,zone] + m.Cavern_Storage_Discharge_Charge[t,zone] - m.H2_Imports[t,zone])

            def storage_demand_rule_2(m, t, zone):
                    return ( m.Zone_H2_Demand_Storage[t,zone] == m.Demand_Met[t,zone] + m.H2_Storage_Discharge_Charge[t,zone] + m.Cavern_Storage_Discharge_Charge[t,zone] - m.H2_Imports[t,zone])

            self.m.storage_demand_constraint = Constraint(self.m.T, self.m.Zones, rule=storage_demand_rule_2)

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

            # Storage Charge Discharge, multi-period; TODO: figure out why mass balance wasn't quite closing and fix
            # TODO: clarify point within the hour that we mean for storage, transmission, etc.
            # TODO: consolodate into single storage rule with multiple storage techs
            def H2_storage_multi_period_rule(m, t, zone):
                if t == m.T.first():
                    return m.H2_Storage_Level[t, zone]==m.H2_Storage_Level[m.T.last(), zone] + m.Cavern_Storage_Discharge_Charge[t,zone]
                else:
                    return (m.H2_Storage_Level[t, zone] == m.H2_Storage_Level[t - 1,zone]  + m.H2_Storage_Discharge_Charge[t, zone])

            self.m.H2_storage_multi_period_constraint = Constraint(self.m.T, self.m.Zones, rule=H2_storage_multi_period_rule)

            def Cavern_storage_multi_period_rule(m, t, zone):
                if t == m.T.first():
                    return m.Cavern_Storage_Level[t, zone]==m.Cavern_Storage_Level[m.T.last(), zone] + m.Cavern_Storage_Discharge_Charge[t,zone]
                else:
                    return (m.Cavern_Storage_Level[t, zone] == m.Cavern_Storage_Level[t - 1,zone]  + m.Cavern_Storage_Discharge_Charge[t, zone])

            self.m.Cavern_storage_multi_period_constraint = Constraint(self.m.T, self.m.Zones, rule=Cavern_storage_multi_period_rule)

            # Cavern charge discharge speed
            # TODO: consolidate into single storage
            def Cavern_charge_rule(m, t, zone):
                return (m.Cavern_Storage_Discharge_Charge[t, zone] <= m.Cavern_charge_limit*m.Cavern_Storage_Capacity[zone])

            self.m.Cavern_charge_constraint = Constraint(self.m.T, self.m.Zones, rule=Cavern_charge_rule)

            def Cavern_discharge_rule(m, t, zone):
                return (m.Cavern_Storage_Discharge_Charge[t, zone] >= -m.Cavern_discharge_limit*m.Cavern_Storage_Capacity[zone])

            self.m.Cavern_discharge_constraint = Constraint(self.m.T, self.m.Zones, rule=Cavern_discharge_rule)

            # Storage Level
            # TODO: consolidate into single storage
            # TODO: note in documentation that storage capacities need to be working capacities
            def H2_storage_level_rule(m, t, zone):
                return (m.H2_Storage_Level[t, zone] <= m.H2_Storage_Capacity[zone])

            self.m.H2_storage_level_constraint = Constraint(self.m.T, self.m.Zones, rule=H2_storage_level_rule)

            def Cavern_storage_level_rule(m, t, zone):
                return (m.Cavern_Storage_Level[t, zone] <= m.Cavern_Storage_Capacity[zone])

            self.m.Cavern_storage_level_constraint = Constraint(self.m.T, self.m.Zones, rule=Cavern_storage_level_rule)

            #TODO: consolidate all of this into some sort of input table with limits for each storage type and zone
            #TODO: consider if we want to still have a total storage constraint
            def Tank_capacity_rule(m,zone):
                cap = self.tank_capacities[zone]
                if cap=='inf': #TODO: see if there is an np.infinity that this could be replaced with?
                    return Constraint.Skip
                else:
                    return (m.H2_Storage_Capacity[zone] <= cap)

            self.m.Tank_capacity_constraint = Constraint(self.m.Zones, rule=Tank_capacity_rule)

            def Cavern_capacity_rule(m, zone):
                cap = self.cavern_capacities[zone]
                if cap=='inf':
                    return Constraint.Skip
                else:
                    return (m.Cavern_Storage_Capacity[zone] <= cap)

            self.m.Cavern_capacity_constraint = Constraint(self.m.Zones, rule=Cavern_capacity_rule)

            def Total_cavern_capacity_rule(m):
                return (sum(m.Cavern_Storage_Capacity[zone] for zone in m.Zones) <= m.Total_cavern_capacity)

            if self.Total_cavern_capacity != 'inf':
                self.m.Total_cavern_capacity_constraint = Constraint(rule=Total_cavern_capacity_rule)

            
            print('Setting up flow constraints...')
            # Pipeline flow (update this later to be part of the variable bound)

            #TODO: have upper bound on pipeline size as an input
            if optimize_pipelines:
                def pipeline_exists_rule(m, link):
                    bigM = 300000*self.h #kg/period, upper bound on pipeline capacity (~7000 TPD)
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

            def pipeline_forward_flow_rule(m, t, link): #TODO: recast specifically with pipelines
                return m.Pipeline_Flow[t, link] <= m.Pipeline_Capacity[link]
                
            self.m.pipeline_forward_flow_constraint = Constraint(self.m.T, self.m.Links, rule=pipeline_forward_flow_rule)

            def pipeline_reverse_flow_rule(m, t, link):
                return m.Pipeline_Flow[t, link] >= -1*m.Pipeline_Capacity[link]

            self.m.pipeline_reverse_flow_constraint = Constraint(self.m.T, self.m.Links, rule=pipeline_reverse_flow_rule)

            print('Setting up renewable zone constraints...')
            # Renewable production - this defines the renewable production at each node  in kwh
            def Renewable_Production_rule(m, tech, t, zone):
                return m.Renewable_Production[tech, t, zone] == sum(self.all_renewable_profiles.loc[self.t_dict[t],(tech,zone,producer)] * m.Renewable_Capacity[tech,zone,producer]  for producer in get_producers(self.capacities,zone=zone,tech = tech).index )

            self.m.Renewable_Production_constraint = Constraint(self.m.Techs, self.m.T, self.m.Zones, rule=Renewable_Production_rule)

            # Renewable build limit  - sets the capacity of each REZ to be less than a value 
            def Renewable_build_limit_rule(m, tech, zone, producer):
                # 
                capacity = get_producers(self.capacities,zone=zone, tech = tech)
                
                if producer in capacity.index:
                    if not capacity.loc[producer]==capacity.loc[producer]:

                        capacity.loc[producer] = 0
                    #print(capacity.loc[producer])
                    return  (m.Renewable_Capacity[tech,zone,producer] <= capacity.loc[producer] * 1000) #TODO: specify units in inputs to avoid this 1000
                else:
                    return Constraint.Skip

            self.m.Renewable_build_limit_constraint = Constraint(self.m.Techs, self.m.Zones, self.m.Renewable_producers, rule=Renewable_build_limit_rule)

            # TODO: recast curtailment and electrolyzer capacity based on generalized electricity/production flows
            def Curtailed_electricity_limit_rule(m, tech, t, zone):
                return m.Curtailed_Renewable_Production[tech, t, zone] <= m.Renewable_Production[tech, t, zone]
            
            self.m.Curtailed_electricity_limit_constraint = Constraint(self.m.Techs, self.m.T, self.m.Zones, rule=Curtailed_electricity_limit_rule) #can't curtailed more electricity than is produced

            # cap renewable production at node to electrolyser capacity
            #TODO: rename this for clarity (could recast for easier understanding but probably not necessary)
            def electrolyser_capacity_limit_rule(m, t, zone):
                return (sum(m.Renewable_Production[tech, t, zone] - m.Curtailed_Renewable_Production[tech, t, zone] for tech in m.Techs)) <= m.Electrolyser_Capacity[zone]*self.h

            self.m.electrolyser_capacity_limit_constraint = Constraint(self.m.T, self.m.Zones, rule=electrolyser_capacity_limit_rule)

            # New H2 production build at nodes
            # electricity is in kWh
            # TODO: generalize to various H2 production pathways (focused on electrolysis)
                # TODO: will need to think about how renewable production matches up with H2 production
                # TODO: create variables for specification of electricity flows from each RE tech to each H2 production tech
            def H2_Production_rule(m, t, zone):
                return m.Hydrogen_Production[t, zone] == (sum(m.Renewable_Production[tech,t, zone]  - m.Curtailed_Renewable_Production[tech,t, zone] for tech in m.Techs))/m.h2_conversion_efficiency

            self.m.H2_Production_constraint = Constraint(self.m.T, self.m.Zones, rule=H2_Production_rule)

            print('Setting up power balance constraints...')
            # H2 balance considering link flows
            def H2_Balance_rule(m, t, zone):
                links_to_this_zone=self.links_to_zones[zone]
                return (m.Hydrogen_Production[t, zone]  + m.H2_Unserved[t, zone] + sum(m.Link_Flow[t, link] * m.link_flow_direction[link, zone] for link in links_to_this_zone) == m.Zone_H2_Demand_Storage[t, zone] + m.H2_Overserved[t, zone] )

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
            # TODO: generalize into generic storage types
            # TODO: allow for variable opex on discharge if desired by user
            def H2_Storage_Cost_rule(m, t, zone):
                return (m.H2_Storage_Cost[t,zone] >= m.H2_Storage_Discharge_Charge[t,zone]*m.Tank_charge_cost)

            self.m.H2_Storage_Cost_constraint = Constraint(self.m.T, self.m.Zones, rule=H2_Storage_Cost_rule)

            def Cavern_Storage_Cost_rule(m, t, zone):
                return (m.Cavern_Storage_Cost[t,zone] >= m.Cavern_Storage_Discharge_Charge[t,zone]*m.Cavern_charge_cost)

            self.m.Cavern_Storage_Cost_constraint = Constraint(self.m.T, self.m.Zones, rule=Cavern_Storage_Cost_rule)

            print('Setting up objective function...')
            #### Objective function
            ## FIXME: enable unserved H2

            def get_CRF(interest=0.10, years=30):
                return (interest * (1 + interest) ** years) /((1 + interest) ** years - 1)

            def objective_rule(m):
                return (sum(
                        m.H2_Storage_Capacity[zone] * self.build_cost.loc['Tank_Storage',zone] +  #cost of tank build
                        m.Cavern_Storage_Capacity[zone] * self.build_cost.loc['Cavern_Storage',zone] +  #cost of cavern build                                                                                                                           
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
                        sum(sum(m.H2_Storage_Cost[t,zone] for t in m.T) for zone in m.Zones) + #opex for storage (input only)
                        sum(sum(m.Cavern_Storage_Cost[t,zone] for t in m.T) for zone in m.Zones) + #opex for storage (input only)
                        sum(sum(sum(m.Renewable_Capacity[tech,zone,producer] * self.build_cost.loc[tech,zone] for tech in m.Techs) for zone in m.Zones) for producer in m.Renewable_producers)  + # renewable build
                        #TODO: handle PTC/ITC in inputs
                        -sum(sum(sum((m.Renewable_Production[tech,t,zone] - m.Curtailed_Renewable_Production[tech,t,zone])*self.PTC[tech] for zone in m.Zones) for t in m.T) for tech in m.Techs) + #PTC effects
                        -sum(sum(sum((m.Renewable_Production[tech,t,zone])*self.ITC[tech] for zone in m.Zones) for t in m.T) for tech in m.Techs) #ITC effects
                )
                        
                #log_infeasible_constraints(m)
            self.m.objective = Objective(rule=objective_rule, sense=minimize)


            # OLD NUCLEAR CONTROLS - TO BE REMOVED #
            '''
             ## nuclear H2 production TODO: remove this
            if (include_nuclear_hydrogen) & (year in nuclear_hydrogen_years):
                print('Including nuclear H2')
            
                def Nuclear_H2_Production_rule(m, t, zone):
                    return m.Nuclear_H2_Production[t, zone] == m.Nuclear_H2_Electrolyser_Capacity[zone]*float(nuclear_profiles.loc[t_dict[t]])/m.nuclear_h2_conversion_efficiency
                m.Nuclear_H2_Production_constraint = Constraint(m.T, m.Nuclear_Zones, rule=Nuclear_H2_Production_rule)
            
                
            # FROM H2 PRODUCTION RULE
                if (include_nuclear_hydrogen) & (year in nuclear_hydrogen_years) & (zone in nuclear_zones):
                    return m.Hydrogen_Production[t, zone] == (sum(m.Renewable_Production[tech,t, zone]  - m.Curtailed_Renewable_Production[tech,t, zone] for tech in m.Techs))/m.h2_conversion_efficiency + m.Nuclear_H2_Production[t, zone]
                else:
            
            # FROM OJBECTIVE FUNCTION
            #TODO: remove nuclear cost
                        sum(sum(m.Nuclear_H2_Production[t, zone] * m.nuclear_h2_conversion_efficiency * m.Nuclear_LCOE for t in m.T if (include_nuclear_hydrogen) & (self.year in nuclear_hydrogen_years)) + m.Nuclear_H2_Electrolyser_Capacity[zone] * self.build_cost.loc['Solid_Oxide_Electrolyser',zone] for zone in m.Nuclear_Zones if (include_nuclear_hydrogen) & (year in nuclear_hydrogen_years))) # nuclear H2 production - only included if include_nuclear_hydrogen==True
            '''

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
                    return
                    break
                elif con != 'y':
                    break
                else:
                    con = input('Invalid input. Please respond (y/n)')


        #collect useful parameters and write to csv for future reference
        params = dict()
        params['year_ratio'] = self.year_ratio
        params['max_imports'] = self.max_imports
        params['h'] = self.h
        params['obj'] = self.m.objective()

        pd.Series(params).to_csv(results_dir+'/params.csv')
        self.build_cost.to_csv(results_dir+'/build_cost.csv')
        PTC_data = pd.DataFrame.from_dict(self.PTC,orient='index')
        PTC_data.to_csv(results_dir+'/PTC.csv')
        ITC_data = pd.DataFrame.from_dict(self.ITC,orient='index')
        ITC_data.to_csv(results_dir+'/ITC.csv')

        #build dataframe of useful capacities
        H2_Storage_Capacity = pd.Series(self.m.H2_Storage_Capacity.extract_values(), index=self.m.H2_Storage_Capacity.extract_values().keys())
        Cavern_Storage_Capacity = pd.Series(self.m.Cavern_Storage_Capacity.extract_values(), index=self.m.Cavern_Storage_Capacity.extract_values().keys())
        Renewable_Capacity = pd.Series(self.m.Renewable_Capacity.extract_values(), index=self.m.Renewable_Capacity.extract_values().keys()).unstack()
        Zone_Capacities=Renewable_Capacity.sum(1).unstack()
        Electrolyser_Capacity = pd.Series(self.m.Electrolyser_Capacity.extract_values(), index=self.m.Electrolyser_Capacity.extract_values().keys())
        Zone_Capacities.loc['PEM_Electrolyser']=Electrolyser_Capacity
        Zone_Capacities.loc['Tank_Storage']=H2_Storage_Capacity
        Zone_Capacities.loc['Cavern_Storage']=Cavern_Storage_Capacity

        H2_Overserved = pd.Series(self.m.H2_Overserved.extract_values(), index=self.m.H2_Overserved.extract_values().keys()).unstack()
        H2_Overserved.index=list(self.t_dict.values())
        H2_Overserved.to_csv(results_dir+'/H2_overserved.csv')

        Link_Flow = pd.Series(self.m.Link_Flow.extract_values(), index=self.m.Link_Flow.extract_values().keys()).unstack()
        Link_Flow.index=list(self.t_dict.values())
        Link_Flow.to_csv(results_dir+'/Link_Flow.csv')

        Pipeline_Flow = pd.Series(self.m.Pipeline_Flow.extract_values(), index=self.m.Pipeline_Flow.extract_values().keys()).unstack()
        Pipeline_Flow.index=list(self.t_dict.values())
        Pipeline_Flow.to_csv(results_dir+'/Pipeline_Flow.csv')

        H2_Storage_Level = pd.Series(self.m.H2_Storage_Level.extract_values(), index=self.m.H2_Storage_Level.extract_values().keys()).unstack()
        H2_Storage_Level.index=list(self.t_dict.values())
        H2_Storage_Level.to_csv(results_dir+'/H2_Storage_Level.csv')

        Cavern_Storage_Level = pd.Series(self.m.Cavern_Storage_Level.extract_values(), index=self.m.Cavern_Storage_Level.extract_values().keys()).unstack()
        Cavern_Storage_Level.index=list(self.t_dict.values())
        Cavern_Storage_Level.to_csv(results_dir+'/Cavern_Storage_Level.csv')

        Renewable_Production = pd.Series(self.m.Renewable_Production.extract_values(), index=self.m.Renewable_Production.extract_values().keys()).unstack()
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


        H2_Storage_Discharge_Charge = pd.Series(self.m.H2_Storage_Discharge_Charge.extract_values(), index=self.m.H2_Storage_Discharge_Charge.extract_values().keys()).unstack()
        H2_Storage_Discharge_Charge.index=list(self.t_dict.values())
        H2_Storage_Discharge_Charge.to_csv(results_dir+'/H2_Storage_Discharge_Charge.csv')

        Cavern_Storage_Discharge_Charge = pd.Series(self.m.Cavern_Storage_Discharge_Charge.extract_values(), index=self.m.Cavern_Storage_Discharge_Charge.extract_values().keys()).unstack()
        Cavern_Storage_Discharge_Charge.index=list(self.t_dict.values())
        Cavern_Storage_Discharge_Charge.to_csv(results_dir+'/Cavern_Storage_Discharge_Charge.csv')

        H2_Unserved = pd.Series(self.m.H2_Unserved.extract_values(), index=self.m.H2_Unserved.extract_values().keys()).unstack()
        H2_Unserved.index=list(self.t_dict.values())
        H2_Unserved.to_csv(results_dir+'/H2_Unserved.csv')

        Zone_Capacities.loc['H2_Unserved_Capacity']=H2_Unserved.max()
        Zone_Capacities.to_csv(results_dir+'/Zone_Capacities.csv')

        Hydrogen_Production = pd.Series(self.m.Hydrogen_Production.extract_values(), index=self.m.Hydrogen_Production.extract_values().keys()).unstack()
        Hydrogen_Production.index=list(self.t_dict.values())
        Hydrogen_Production.to_csv(results_dir+'/H2_Production.csv')

        Curtailed_Renewable_Production = pd.Series(self.m.Curtailed_Renewable_Production.extract_values(), index=self.m.Curtailed_Renewable_Production.extract_values().keys()).unstack()
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


        # OLD NUCLEAR INFO - TO BE DELETED #
        '''
        params['nuclear_h2_conversion_efficiency'] = m.nuclear_h2_conversion_efficiency.value
        params['Nuclear_LCOE'] = m.Nuclear_LCOE.value

        if include_nuclear_hydrogen  & (year in nuclear_hydrogen_years):
            Nuclear_H2_Production = pd.Series(m.Nuclear_H2_Production.extract_values(), index=m.Nuclear_H2_Production.extract_values().keys()).unstack()
            Nuclear_Electricity_consumption=Nuclear_H2_Production*m.nuclear_h2_conversion_efficiency.value
            Nuclear_H2_Electrolyser_Capacity = pd.Series(m.Nuclear_H2_Electrolyser_Capacity.extract_values(), index=m.Nuclear_H2_Electrolyser_Capacity.extract_values().keys())
            Zone_Capacities.loc['Solid_Oxide_Electrolyser']=Nuclear_H2_Electrolyser_Capacity
            nuclear_production_lcoh=(Nuclear_Electricity_consumption*m.Nuclear_LCOE.value).sum()/Nuclear_H2_Production.sum()
            Nuclear_H2_Production.to_csv(results_dir+'/Nuclear_H2_Production.csv')
        '''


test = HYPSTAT()
test.two_step_solve()
test.write_outputs('test_case_pipelines_no_imports')
print('Done!')

'''
GUIDE TO TEST CASES:

-test_case_pipelines: version of the model with specific variable created for pipelines...should have extra output for pipeline flow, but other than that outputs should match
'''