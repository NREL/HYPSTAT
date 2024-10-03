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

from model_functions_241003 import *

#TODO rename variables


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

class HYPSTAT:
        
    def __init__(self, yaml_file_path='HYPSTAT_scenario.yaml'):
        self.yaml_file_path = yaml_file_path
        #self.load_inputs()

    def load_inputs(self,optimize_pipelines=False,Pipeline_Exists=None):
        # READ INPUTS FROM YAML FILE
        yaml_path = Path(self.yaml_file_path)
        with yaml_path.open('r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        
        #   collect model control values from YAML file
        self.year = yaml_data.get('analysis_year',None)
        #self.firstYear = yaml_data.get('firstYear', None)
        #self.lastYear = yaml_data.get('lastYear', None)
        #self.timeWindow = yaml_data.get('timeWindow', None)
        self.max_imports_ratio = yaml_data.get('max_imports_ratio', None)
        self.min_exports_ratio = yaml_data.get('min_exports_ratio', None)
        self.import_nodes = yaml_data.get('import_nodes', None)
        self.export_nodes = yaml_data.get('export_nodes', None)
        self.overserved_cost = yaml_data.get('overserved_cost', None)
        self.unserved_cost = yaml_data.get('unserved_cost', None) #[$/kg]
        self.start = yaml_data.get('start', None)
        self.end = yaml_data.get('end', None)
        self.coarse_resolution = yaml_data.get('coarse_resolution', None)
        self.fine_resolution = yaml_data.get('fine_resolution', None)
        
        #   build hourly time period set based on model year, for input processing
        self.time_periods = pd.date_range(str(self.year)+'-01-01 00:00:00', periods=8760, freq="H") 

        #   read in file paths for inputs
        #TODO: rename in YAML file
        self.demand_path = yaml_data.get('demandFile', [])
        self.financials_path = yaml_data.get('FinancialFile', [])
        self.gen_cost_path = yaml_data.get('REcostFile', [])
        self.gen_cost_profiles_path = yaml_data.get('REcostProfileFile', [])
        self.prod_cost_path = yaml_data.get('ProductioncostFile', [])
        self.transport_cost_path = yaml_data.get('H2DeliveryFile', [])
        self.stor_cost_path = yaml_data.get('H2StorageFile', [])
        self.gen_profiles_path = yaml_data.get('SupplycurveFolder', [])
        self.network_path = yaml_data.get('NetworksFiles', [])
        #self.Pipeline_opex_override_path = yaml_data.get('PipelineCostOverideFile',[])
        self.gen_incentives_path = yaml_data.get('IncentiveFiles', [])
        self.stor_capacity_path = yaml_data.get('StorageCapacityFiles', [])
        self.stor_constraints_path = yaml_data.get('StorageLimitsFiles', [])

        self.nodes = yaml_data.get('nodes',[])
        self.gen_techs = yaml_data.get('techs') #TODO: rename in YAML file
        self.stor_techs = yaml_data.get('stor_techs')
            
        # BUILD TECHNOLOGY AND COST INPUTS FOR MODEL FROM INPUT FILES     
        self.raw_gen_cost = pd.concat([pd.read_csv(path,index_col=['Year','Node','Tech']) for path in self.gen_cost_path])#, ignore_index=True)
        self.gen_cost = expand_input_df(self.raw_gen_cost,'Node',self.nodes)
        self.raw_stor_cost = pd.concat([pd.read_csv(path,index_col=['Year','Node','Tech']) for path in self.stor_cost_path])#, ignore_index=True)
        self.stor_cost = expand_input_df(self.raw_stor_cost,'Node',self.nodes)
        self.raw_prod_cost = pd.read_csv(self.prod_cost_path[0],index_col=['Year','Node','Tech'])
        self.prod_cost = expand_input_df(self.raw_prod_cost,'Node',self.nodes)

        if self.gen_techs is None:
            print('Generation technologies are not explicitly specified; inferring from generation cost file!')
            self.gen_techs = list(self.gen_cost.index.get_level_values('Tech').unique())
        
        if self.stor_techs is None:
            print('Storage technologies are not explicilty specified; inferring from storage cost file!')
            self.stor_techs = list(self.stor_cost.index.get_level_values('Tech').unique())

        #   Get profiles and capacities for electricity gen, demand, and links
        self.gen_profiles, self.gen_capacity_limits = get_gen_profiles(year=self.year, techs=self.gen_techs, path=self.gen_profiles_path[0], drop_capacity_below=False) #[RE profiles: CF (kWh/kW)/hr for each hour; capacities: MW] TODO: make capacities kW?, note about drop_capacity_below
        
        #TODO: update so demand references time periods and not gen profiles
        self.demand = get_demand(self.gen_profiles, year=self.year, freq_in='D',freq_out='h', file_path=self.demand_path[0]) #Must be daily
        
        self.links, self.link_flow_direction, self.links_to_nodes = get_links(self.network_path[0])
        
        #   Create matrix for annuitized costs for CAPEX + Fixed OPEX for each technology
        self.financials = pd.read_csv(self.financials_path[0],index_col=['Year','Tech'])     

        self.build_cost = get_build_cost_matrix(self.financials,self.gen_cost,self.prod_cost,self.stor_cost,self.year,self.nodes)

        #   Costs and params for H2 production TODO avoid hard-coding "PEM Electrolyzer"; come back and rename "Electrolyzer"
        self.prod_efficiency = self.prod_cost.xs((self.year,'PEM Electrolyzer'),level=('Year','Tech'))['efficiency(kWh/kg)']
        # TODO: add in consideration of H2 variable opex, the "compressor charge" is not currently used
        #self.compressor_charge = self.prod_cost.xs((self.year,'PEM Electrolyzer'),level=('Year','Tech'))['Variable OPEX ($/kg)']

        #   Develop cost tables for storage and delivery
        self.stor_charge_cost = self.stor_cost.xs(self.year,level='Year')['Variable OPEX ($/kg-charged)']

        self.stor_capacity_limits = pd.read_csv(self.stor_capacity_path[0],index_col='Node')
        self.stor_constraints = pd.read_csv(self.stor_constraints_path[0],index_col='Storage Tech')

        self.transport_cost = pd.read_csv(self.transport_cost_path[0],index_col=['Year','Tech'])

        distance_unit = 100 #km; used to input transportation cost, e.g., costs are input per distance-unit kms. Default is 100

        #self.transport_cost = (self.transport_cost[self.transport_cost['Year'] == self.year]).set_index('Tech')
        self.transport_cost = self.transport_cost.xs(self.year,level='Year')
        self.links['Pipeline Opex ($/kg)'] = self.transport_cost.loc['Pipeline','Variable OPEX ($/kg/{}-km)'.format(distance_unit)] * \
                                                  self.links['Pipeline distance [km]'] / distance_unit
        self.links['Truck LCOT ($/kg)'] = self.transport_cost.loc['Truck','LCOT ($/kg/{}-km)'.format(distance_unit)] * \
                                                  self.links['Truck distance [km]'] / distance_unit
        self.pipeline_fixed_capex = self.transport_cost.loc['Pipeline','Fixed CAPEX ($/km)']
        self.pipeline_capacity_capex = self.transport_cost.loc['Pipeline','Capacity CAPEX ($/(kg/hr)/km)']  
        self.pipeline_fixed_opex_frac = self.transport_cost.loc['Pipeline','Fixed OPEX (fraction of CAPEX/yr)']      

        # BUILD ELECTRICITY GEN OPEX TABLE
        #   The electricity gen opex table is indexed for each node and technology and is indexed for time intervals to allow for variable pricing (e.g., wholesale grid).
        #   A constant variable opex (in currency/kWh) is automatically loaded from the technology financials file. 
        #   If provided, profile files for electricity gen opex will override the constant value in the financial files--i.e., the variable OPEX value in the technology financials file is NOT used.
                    
        multi_index = pd.MultiIndex.from_product([self.nodes,self.gen_techs])
        self.gen_opex = pd.DataFrame(data=0, index=self.time_periods, columns=multi_index)

        # fill electricity gen opex table with constant variable opex values from REcostFiles
        for node in self.nodes:
            for tech in self.gen_techs:
                self.gen_opex.loc[:,(node,tech)] = self.gen_cost.loc[(self.year,node,tech),'Variable OPEX ($/kWh)']

        # override electricity gen opex for technologies with time profiles (e.g., grid representing hourly wholesale pricing)
        #   Price inputs must be hourly with 8760 entries; these are lined up with the model's time index.
                
        for node in self.nodes:
            for tech in self.gen_techs:
                if tech not in self.gen_cost_profiles_path.keys():
                    continue
                self.gen_cost_profiles = pd.read_csv(self.gen_cost_profiles_path[tech])
                self.gen_cost_profiles.fillna(0,inplace=True)
                self.gen_cost_profiles.index = self.time_periods
                if False: #eventually for locational pricing, TODO: resolve this
                    self.gen_opex.loc[:,(node,tech)] = self.gen_cost_profiles[node]/1000 # [$/kWh]
                else:
                    self.gen_opex.loc[:,(node,tech)] = self.gen_cost_profiles['Price ($/kWh)'] #TODO: figure out way to input or specify column name

        # READ ELECTRICITY GEN INCENTIVES
        self.gen_incentives = pd.read_csv(self.gen_incentives_path[0],index_col=['Year','Tech'])

        # IMPORTS/EXPORTS AND MODEL CONTROLS
        self.max_imports = self.demand.sum().sum() * self.max_imports_ratio
        self.min_exports = self.demand.sum().sum() * self.min_exports_ratio #NOTE: ratio or amount could be changed

        # UPDATE TIME BOUNDS FOR RELEVANT PARAMETERS (for runs <1 full year)
        #self.gen_profiles = self.gen_profiles.loc[self.start + str(self.year): self.end + str(self.year)]
        self.gen_profiles = self.gen_profiles.loc['{} {}'.format(self.start,self.year): '{} {}'.format(self.end,self.year):]
        self.year_ratio = len(self.gen_profiles.resample('d').first()) / 365
        self.max_imports *= self.year_ratio
        self.min_exports *= self.year_ratio
        self.build_cost *= self.year_ratio

        # SET UP PIPELINE INPUTS BASED ON WHETHER OR NOT PIPELINE LOCATIONS ARE OPTIMIZED
        if optimize_pipelines:
            #first iteration
            self.h=self.coarse_resolution # [hrs/period] this specifies the number of hours in each time interval. It allows the conversion of kg/h when multiple hours are in each timestep.
            print('Optimizing pipelines')
            print()
        else:
            #second iteration, hourly run
            self.h=self.fine_resolution # [hrs/period]
            print('Performing full optimization')
            print()

        if Pipeline_Exists is not None:
            threshold = 0.1
            for link in Pipeline_Exists.index:
                if Pipeline_Exists[link] < threshold:
                    #close to 0, pipeline doesn't exist
                    self.links.loc[link,'Pipeline max capacity [kg/hr]'] = 0
                    self.links.loc[link,'Pipeline allowed'] = 'N'
        else:
            self.Pipeline_Exists = None

        # RESAMPLE RELEVANT DATA BASED ON TIME RESOLUTION
        if self.h != 1:
            self.gen_profiles = self.gen_profiles.resample('{}h'.format(self.h)).sum() #[period CF, i.e., (kWh/kW)/period]
            self.demand = self.demand.resample('{}h'.format(self.h)).sum() # [kg/period]
            self.gen_opex = self.gen_opex.resample('{}h'.format(self.h)).mean() # [$/kWh, average]

    
    def load_model(self,optimize_pipelines=False):
        self.m = ConcreteModel()

        #Define time and node index sets
        self.t_dict = {ind + 1: k for ind, k in enumerate(self.gen_profiles.index)}

        ### INDEX SETS ###
        self.m.T = Set(initialize=(sorted(list(self.t_dict.keys()))), ordered=True) # sort t_dict

        self.m.Nodes = Set(initialize=list(self.nodes), ordered=True)
        self.m.Links = Set(initialize=list(self.links.index), ordered=True)
        self.m.Gen_Techs = Set(initialize=list(self.gen_techs), ordered=True)
        self.m.Stor_Techs = Set(initialize=list(self.stor_techs),ordered=True)
        self.m.Gen_Tranches = Set(initialize=list(self.gen_capacity_limits.index), ordered=True) # This set defines the index for different resource qualities in a single location

        ### PARAMETERS ###
        self.m.Cost_of_unserved_H2 = Param(initialize=self.unserved_cost) # [EUR/kg] We assume that a kg of unserved hydrogen has an economic cost of 10eX.

        self.m.Storage_charge_cost = Param(self.m.Stor_Techs,self.m.Nodes,initialize=lambda m,st,z: self.stor_charge_cost.loc[(z,st)])
        self.m.Storage_charge_limit = Param(self.m.Stor_Techs, initialize=lambda m,st: self.stor_constraints.loc[st,'charge limit']*self.h,within=Any) #[% of wk cap per period or INF]
        self.m.Storage_discharge_limit = Param(self.m.Stor_Techs, initialize=lambda m,st: self.stor_constraints.loc[st,'discharge limit']*self.h,within=Any) #[% of working cap per period or INF]
        self.m.Total_storage_capacity = Param(self.m.Stor_Techs, initialize=lambda m,st: self.stor_constraints.loc[st,'capacity'],within=Any) #[kg or INF]
        self.m.Truck_flow_allowed = Param(self.m.Links,initialize={str(lk): 1 if self.links.loc[lk,'Truck allowed']=='Y' else 0 for lk in self.m.Links}) #0 if not allowed for each link, 1 if allowed (multiplier for capacity limits)
        self.m.Pipeline_flow_allowed = Param(self.m.Links,initialize={str(lk): 1 if self.links.loc[lk,'Pipeline allowed']=='Y' else 0 for lk in self.m.Links})

        ### CAPACITY VARIABLES ###
        self.m.Gen_Capacity = Var(self.m.Gen_Techs,self.m.Nodes,self.m.Gen_Tranches, domain=NonNegativeReals, initialize=0)  #[kW]
        self.m.Electrolyser_Capacity = Var(self.m.Nodes, domain=NonNegativeReals) #[kW] 
        self.m.Storage_Capacity = Var(self.m.Stor_Techs,self.m.Nodes,domain=NonNegativeReals) #[kg] wk capacity
        self.m.Pipeline_Capacity = Var(self.m.Links,domain=NonNegativeReals) # [kg/period] pipeline capacity for each node
        if optimize_pipelines:
            self.m.Pipeline_Exists = Var(self.m.Links,domain=Binary) #whether or not a pipeline exists
        
        ### PRODUCTION AND FLOW VARIABLES ###

        # Electricity
        self.m.Electricity_Potential  = Var(self.m.Gen_Techs, self.m.T, self.m.Nodes)  #[kWh] potential to generate electricity from RE (or grid) 
        self.m.Electricity_Used = Var(self.m.Gen_Techs, self.m.T, self.m.Nodes) #[kWh] electricity used for hydrogen production
        self.m.Electricity_Curtailed = Var(self.m.Gen_Techs,self.m.T, self.m.Nodes, domain=NonNegativeReals) #[kWh] electricity production

        # Hydrogen
        self.m.H2_Production = Var(self.m.T, self.m.Nodes, domain=NonNegativeReals)   #[kg/period]
        self.m.H2_Demand_Met = Var(self.m.T, self.m.Nodes, domain=NonNegativeReals) #[kg/period]
        self.m.H2_Unserved = Var(self.m.T, self.m.Nodes, domain=NonNegativeReals) #[kg/period],bounds=(0,0) )   #kg/per interval
        self.m.H2_Overserved = Var(self.m.T, self.m.Nodes, domain=NonNegativeReals) #[kg/period] ,bounds=(0,0) )  
        self.m.H2_Imports = Var(self.m.T, self.m.Nodes, domain=NonNegativeReals) #[kg/period] imported hydrogen, kg/interval
        self.m.H2_Exports = Var(self.m.T, self.m.Nodes, domain=NonNegativeReals) #[kg/period] exported hydrogen, kg/interval

        # Storage
        self.m.Storage_Charge = Var(self.m.Stor_Techs,self.m.T,self.m.Nodes) #[kg]
        self.m.Storage_Level = Var(self.m.Stor_Techs,self.m.T,self.m.Nodes,domain=NonNegativeReals) #[kg]
        self.m.Storage_Charge_OPEX = Var(self.m.Stor_Techs,self.m.T,self.m.Nodes, domain=NonNegativeReals) #[EUR/period]
        
        # Transmission
        self.m.Link_Flow = Var(self.m.T, self.m.Links)  #[kg]
        self.m.Pipeline_Flow = Var(self.m.T, self.m.Links) #[kg]
        self.m.Truck_Flow = Var(self.m.T,self.m.Links) #[kg]
        self.m.Pipeline_OPEX = Var(self.m.T, self.m.Links, domain=NonNegativeReals) #[$/period]
        self.m.Truck_Cost = Var(self.m.T, self.m.Links, domain=NonNegativeReals) #[$/period] 
        
        
    def load_constraints(self,optimize_pipelines=False):
        # Load pipeline flow incidence matrix
        def link_rule(m, link, node):
            if node not in self.link_flow_direction.columns:
                #Node is not connected by any links
                return 0
            else:
                return float(self.link_flow_direction.loc[link, node])

        self.m.link_flow_direction = Param(self.m.Links, self.m.Nodes, initialize=link_rule)

        ### DEMAND CONSTRAINTS ###
        print('Setting up demand constraints...')

        # Option 1: Flexible demand over multiple time periods
        def flexible_demand_rule(m, t, node):
            #NOTE: this flexible demand rule is not rigorously tested
            period_hours = 24 #length of time over which demand must be met
            flex_steps = period_hours/self.h
            if t%flex_steps==0:
                return (sum(m.H2_Demand_Met[t_step,node] for t_step in range(t,t+flex_steps)) == sum(self.demand.loc[self.t_dict[t_step],node] for t_step in range(t,t+flex_steps)))
            else:
                return Constraint.Skip

        # Option 2: Demand must be strictly met in each period
        def strict_demand_rule(m, t, node):
            return (m.H2_Demand_Met[t,node] == self.demand.loc[self.t_dict[t],node]) #[kg/period]

        #self.m.demand_constraint = Constraint(m.T, m.Nodes, rule=flexible_demand_rule)
        self.m.demand_constraint = Constraint(self.m.T, self.m.Nodes, rule=strict_demand_rule)

        ### IMPORT CONSTRAINTS ###
        print('Setting up import constraints...')

        # Limit quantity of total imports
        def total_imports_rule(m):
            return sum(sum(m.H2_Imports[t,node] for t in m.T) for node in m.Nodes) <= self.max_imports #[kg]
        
        self.m.total_imports_constraint = Constraint(rule=total_imports_rule)

        # Restrict which nodes are allowed to import
        def node_imports_rule(m, t, node):
            if node in self.import_nodes:
                return Constraint.Skip
            else:
                return m.H2_Imports[t,node] == 0 #[kg/period]
        
        self.m.node_imports_constraint = Constraint(self.m.T,self.m.Nodes,rule=node_imports_rule)

        ### EXPORT CONSTRAINTS ###
        print('Setting up export constraints...')

        # Ensure that model meets a minimum quota of exports
        def total_exports_rule(m):
            return (sum(sum(m.H2_Exports[t,node] for t in m.T) for node in m.Nodes)) >= self.min_exports #[kg]

        self.m.total_exports_constraint = Constraint(rule=total_exports_rule)

        # Restrict which nodes are allowed to export
        def node_exports_rule(m, t, node):
            if node in self.export_nodes:
                return Constraint.Skip
            else:
                return m.H2_Exports[t,node] == 0 #[kg/period]
        
        self.m.node_exports_constraint = Constraint(self.m.T,self.m.Nodes,rule=node_exports_rule)

        ### STORAGE CONSTRAINTS ###
        print('Setting up storage constraints...')

        # Constraint storage level based on inflow/outflow of storage
        def Storage_multi_period_rule(m, stor_tech, t, node):
            if t == m.T.first():
                return m.Storage_Level[stor_tech,t,node] == m.Storage_Level[stor_tech,m.T.last(),node] + m.Storage_Charge[stor_tech,t,node] #[kg]
            else:
                return (m.Storage_Level[stor_tech, t, node] == m.Storage_Level[stor_tech, t-1, node]  + m.Storage_Charge[stor_tech, t, node]) #[kg]

        self.m.Storage_multi_period_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Nodes, rule=Storage_multi_period_rule)

        # Constrain storage charge speed
        def Storage_charge_rule(m, stor_tech, t, node):
            if m.Storage_charge_limit[stor_tech]==np.inf or np.isnan(m.Storage_charge_limit[stor_tech]):
                return Constraint.Skip
                #Could also use just a really large number but probably leave as skip constraint
            else:
                return (m.Storage_Charge[stor_tech, t, node] <= m.Storage_charge_limit[stor_tech]*m.Storage_Capacity[stor_tech, node]) #[kg]
        
        self.m.Storage_charge_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Nodes, rule=Storage_charge_rule)

        # Constrain storage discharge speed
        def Storage_discharge_rule(m, stor_tech, t, node):
            if m.Storage_discharge_limit[stor_tech]==np.inf or np.isnan(m.Storage_discharge_limit[stor_tech]):
                return Constraint.Skip
            else:
                return (m.Storage_Charge[stor_tech, t, node] >= -m.Storage_discharge_limit[stor_tech]*m.Storage_Capacity[stor_tech, node]) #[kg]

        self.m.Storage_discharge_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Nodes, rule=Storage_discharge_rule)

        # Constrain storage level to be within capacity
        def Storage_level_rule(m, stor_tech, t, node):
            return (m.Storage_Level[stor_tech, t, node] <= m.Storage_Capacity[stor_tech, node]) #[kg]

        self.m.Storage_level_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Nodes, rule=Storage_level_rule)

        # Constrain storage capacity within each node (e.g., for geologic limitations)
        def Storage_capacity_cap_rule(m, stor_tech, node):
            cap = self.stor_capacity_limits.loc[stor_tech,node]
            if cap==np.inf or np.isnan(cap):
                return Constraint.Skip
            else:
                return (m.Storage_Capacity[stor_tech, node] <= cap) #[kg]

        self.m.Storage_capacity_cap_constraint = Constraint(self.m.Stor_Techs, self.m.Nodes, rule=Storage_capacity_cap_rule)

        # Constrain total storage capacity among all storage of a given type
        def Total_storage_capacity_rule(m, stor_tech):
            if m.Total_storage_capacity[stor_tech]==np.inf or np.isnan(m.Total_storage_capacity[stor_tech]):
                return Constraint.Skip
            else:
                return (sum(m.Storage_Capacity[stor_tech, node] for node in m.Nodes) <= m.Total_storage_capacity[stor_tech]) #[kg]

        self.m.Total_storage_capacity_constraint = Constraint(self.m.Stor_Techs, rule=Total_storage_capacity_rule)

        ### LINK FLOW CONSTRAINTS ###
        print('Setting up flow constraints...')
        # Pipeline flow (update this later to be part of the variable bound)

        # Constrain pipeline sizes based on binary variables OR based on inputs from the previous iteration
        if optimize_pipelines:
            def pipeline_exists_rule(m, link):
                return m.Pipeline_Capacity[link] <= self.links.loc[link,'Pipeline max capacity [kg/hr]']*self.h*m.Pipeline_flow_allowed[link]*m.Pipeline_Exists[link] #[kg/period]
            
            self.m.pipeline_exists_constraint = Constraint(self.m.Links,rule=pipeline_exists_rule)
        
        else:
            #not optimizing pipelines, limit sizes according to links

            def pipeline_size_rule(m, link):
                return m.Pipeline_Capacity[link] <= self.links.loc[link,'Pipeline max capacity [kg/hr]']*self.h*m.Pipeline_flow_allowed[link] #[kg/period]
            
            self.m.pipeline_size_constraint = Constraint(self.m.Links, rule=pipeline_size_rule)

        # Set total link flow for mass balance based on pipeline and truck flow
        
        def total_link_flow_rule(m, t, link):
            return m.Link_Flow[t, link] == m.Pipeline_Flow[t, link] + m.Truck_Flow[t, link]  #[kg]

        self.m.total_link_flow_constraint = Constraint(self.m.T, self.m.Links, rule=total_link_flow_rule)
        
        # Limit truck flows according to size limit
        def forward_max_truck_rule(m, t, link):
            return m.Truck_Flow[t,link] <= self.links.loc[link,'Truck max capacity [kg/hr]']*self.h*m.Truck_flow_allowed[link] #[kg/period]
        
        self.m.forward_max_truck_constraint = Constraint(self.m.T, self.m.Links, rule=forward_max_truck_rule)

        def reverse_max_truck_rule(m, t, link):
            return m.Truck_Flow[t,link] >= -self.links.loc[link,'Truck max capacity [kg/hr]']*self.h*m.Truck_flow_allowed[link] #[kg/period]

        self.m.reverse_max_truck_constraint = Constraint(self.m.T, self.m.Links, rule=reverse_max_truck_rule)

        # Limit pipeline flows according to pipeline capacities
        def forward_max_pipeline_rule(m, t, link):
            return m.Pipeline_Flow[t, link] <= m.Pipeline_Capacity[link] #[kg/period]
            
        self.m.forward_max_pipeline_constraint = Constraint(self.m.T, self.m.Links, rule=forward_max_pipeline_rule)

        def reverse_max_pipeline_rule(m, t, link):
            return m.Pipeline_Flow[t, link] >= -1*m.Pipeline_Capacity[link] #[kg/period]

        self.m.reverse_max_pipeline_constraint = Constraint(self.m.T, self.m.Links, rule=reverse_max_pipeline_rule)

        ### PRODUCTION CONSTRAINTS ###
        print('Setting up electricity and hydrogen production constraints...')
        
        # Define the electricity production potential at each node in kwh
        def Electricity_Potential_rule(m, tech, t, node):
            return m.Electricity_Potential[tech, t, node] == sum(self.gen_profiles.loc[self.t_dict[t],(tech,node,tranche)] * m.Gen_Capacity[tech,node,tranche]  for tranche in get_producers(self.gen_capacity_limits,node=node,tech = tech).index ) #[kWh, from kW * (kWh/kW)/period for period]

        self.m.Electricity_Potential_constraint = Constraint(self.m.Gen_Techs, self.m.T, self.m.Nodes, rule=Electricity_Potential_rule)

        # Constrain the capacity of electricity generators
        #TODO: revisit this set of constraints
        def Gen_capacity_limit_rule(m, tech, node, tranche):
            capacity = get_producers(self.gen_capacity_limits,node=node,tech=tech)
            if tranche in capacity.index:
                if not capacity.loc[tranche]==capacity.loc[tranche]:
                    capacity.loc[tranche] = 0
                return  (m.Gen_Capacity[tech,node,tranche] <= capacity.loc[tranche] * 1000) # [kW, capacity must be in MW] #TODO: specify units in inputs to avoid this 1000
            else:
                return Constraint.Skip

        self.m.Gen_capacity_limit_constraint = Constraint(self.m.Gen_Techs, self.m.Nodes, self.m.Gen_Tranches, rule=Gen_capacity_limit_rule)

        # Constrain curtailed electricty to less than production
        def Curtailed_electricity_limit_rule(m, tech, t, node):
            return m.Electricity_Curtailed[tech, t, node] <= m.Electricity_Potential[tech, t, node] #[kWh/period]
        
        self.m.Curtailed_electricity_limit_constraint = Constraint(self.m.Gen_Techs, self.m.T, self.m.Nodes, rule=Curtailed_electricity_limit_rule) #can't curtailed more electricity than is produced

        # Set amount of electricity used based on curtailed electricity
        def Electricity_Used_rule(m, tech, t, node):
            return (m.Electricity_Used[tech, t, node] == m.Electricity_Potential[tech, t, node] - m.Electricity_Curtailed[tech, t, node]) #[kWh/period]

        self.m.Electricity_Used_constraint = Constraint(self.m.Gen_Techs, self.m.T, self.m.Nodes, rule=Electricity_Used_rule)

        # Constrain electricity used based on electrolyzer capacity
        def Electrolyser_Capacity_Limit_rule(m, t, node):
            return (sum(m.Electricity_Used[tech, t, node] for tech in m.Gen_Techs)) <= m.Electrolyser_Capacity[node]*self.h #[kWh/period]

        self.m.electrolyser_capacity_limit_constraint = Constraint(self.m.T, self.m.Nodes, rule=Electrolyser_Capacity_Limit_rule)

        # New H2 production build at nodes
        # electricity is in kWh
        
        # Set H2 production
        def H2_Production_rule(m, t, node):
            return m.H2_Production[t, node] == (sum(m.Electricity_Used[tech, t, node] for tech in m.Gen_Techs))/self.prod_efficiency.loc[node] #[kg/period]

        self.m.H2_Production_constraint = Constraint(self.m.T, self.m.Nodes, rule=H2_Production_rule)

        ### POWER BALANCE ###
        print('Setting up power balance constraints...')

        # Set the hydrogen balance
        def H2_Balance_rule(m, t, node):
            links_to_this_node=self.links_to_nodes[node]
            return (m.H2_Production[t, node] + m.H2_Unserved[t, node] + sum(m.Link_Flow[t, link] * m.link_flow_direction[link, node] for link in links_to_this_node) + m.H2_Imports[t,node] == \
                    sum(m.Storage_Charge[stor_tech, t, node] for stor_tech in m.Stor_Techs) + m.H2_Demand_Met[t, node] + m.H2_Overserved[t, node] + m.H2_Exports[t, node]) #[kg/period]

        self.m.H2_Balance_constraint = Constraint(self.m.T, self.m.Nodes, rule=H2_Balance_rule)

        ### TRANSMISSION AND STORAGE COSTS ###
        print('Setting up transmission cost constraints...')
        
        # Set and truck costs
        def Forward_Pipeline_OPEX_rule(m, t, link):
            return (m.Pipeline_OPEX[t,link] >= m.Pipeline_Flow[t,link]*self.links.loc[link,'Pipeline Opex ($/kg)']) #[$/period]

        self.m.Forward_Pipeline_OPEX_constraint = Constraint(self.m.T, self.m.Links, rule=Forward_Pipeline_OPEX_rule)

        def Reverse_Pipeline_OPEX_rule(m, t, link):
            return (m.Pipeline_OPEX[t,link] >= -m.Pipeline_Flow[t,link]*self.links.loc[link,'Pipeline Opex ($/kg)']) #[$/period]

        self.m.Reverse_Pipeline_OPEX_constraint = Constraint(self.m.T, self.m.Links, rule=Reverse_Pipeline_OPEX_rule)

        def Forward_Truck_Cost_rule(m, t, link):
            return (m.Truck_Cost[t,link] >= m.Truck_Flow[t,link]*self.links.loc[link,'Truck LCOT ($/kg)']) #[$/period] 
        
        self.m.Forward_Truck_Cost_constraint = Constraint(self.m.T, self.m.Links, rule=Forward_Truck_Cost_rule)

        def Reverse_Truck_Cost_rule(m, t, link):
            return (m.Truck_Cost[t,link] >= -m.Truck_Flow[t,link]*self.links.loc[link,'Truck LCOT ($/kg)']) #[$/period]
        
        self.m.Reverse_Truck_Cost_constraint = Constraint(self.m.T, self.m.Links, rule=Reverse_Truck_Cost_rule)

        print('Setting up storage cost constraints...')

        #Set storage charge costs
        def Storage_Charge_OPEX_rule(m, stor_tech, t, node):
            return (m.Storage_Charge_OPEX[stor_tech,t,node] >= m.Storage_Charge[stor_tech,t,node]*m.Storage_charge_cost[stor_tech,node]) #[$/period]

        self.m.Storage_Charge_OPEX_constraint = Constraint(self.m.Stor_Techs, self.m.T, self.m.Nodes, rule=Storage_Charge_OPEX_rule)

        print('Setting up objective function...')
        #### Objective function
        
        def objective_rule(m):
            
            Gen_Stor_CAPEX = sum(
                sum(m.Storage_Capacity[stor_tech, node] * self.build_cost.loc['{}'.format(stor_tech),node] for stor_tech in m.Stor_Techs) + #storage technology
                m.Electrolyser_Capacity[node] * self.build_cost.loc['PEM Electrolyzer',node] + #electrolyzers TODO: make this not hard coded
                sum(sum(m.Gen_Capacity[tech,node,tranche] * self.build_cost.loc[tech,node] for tech in m.Gen_Techs) for tranche in m.Gen_Tranches) #generators
            for node in m.Nodes)
            
            Pipeline_raw_CAPEX = sum(
                ((m.Pipeline_Capacity[link]/self.h)*self.pipeline_capacity_capex + self.pipeline_fixed_capex*(m.Pipeline_Exists[link] if optimize_pipelines else 1))*self.year_ratio*self.links.loc[link,'Pipeline distance [km]']
            for link in m.Links)

            pipeline_wacc = self.financials.loc[(self.year,'Pipelines'),'WACC']
            pipeline_recovery_time = self.financials.loc[(self.year,'Pipelines'),'Recovery_time (years)']
            Pipeline_CAPEX = get_annuity(Pipeline_raw_CAPEX,pipeline_wacc,pipeline_recovery_time) + Pipeline_raw_CAPEX*self.pipeline_fixed_opex_frac

            Transmission_OPEX = sum(
                sum(m.Pipeline_OPEX[t,link] for t in m.T) + #pipeline opex
                sum(m.Truck_Cost[t,link] for t in m.T) #truck costs
            for link in m.Links)
            
            Gen_Stor_OPEX = sum(
                sum(sum(m.Storage_Charge_OPEX[stor_tech,t,node] for t in m.T) for stor_tech in m.Stor_Techs) + #storage opex
                sum(sum((m.Electricity_Used[tech, t, node])*self.gen_opex.loc[self.t_dict[t],(node,tech)] for t in m.T) for tech in m.Gen_Techs) #gen opex
            for node in m.Nodes)

            Incentives = (
                -sum(sum(sum((m.Electricity_Used[tech, t, node])*self.gen_incentives.loc[(self.year,tech),'PTC ($/kWh)'] for node in m.Nodes) for t in m.T) for tech in m.Gen_Techs) + #PTC effects
                -sum(sum(sum((m.Electricity_Potential[tech,t,node])*self.gen_incentives.loc[(self.year,tech),'ITC ($/kWh)'] for node in m.Nodes) for t in m.T) for tech in m.Gen_Techs) #ITC effects
             ) 

            Penalties = sum(
                sum(m.H2_Unserved[t, node] * m.Cost_of_unserved_H2 for t in m.T) + #unserved hydrogen
                sum(m.H2_Overserved[t, node] * self.overserved_cost for t in m.T) #overserved hydrogen
            for node in m.Nodes)

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
        incentive_data = self.gen_incentives.xs(self.year,level='Year')
        incentive_data.to_csv(results_dir+'/incentives.csv')
      
        
        Storage_Capacity = pd.Series(self.m.Storage_Capacity.extract_values(), index=self.m.Storage_Capacity.extract_values().keys()).unstack()
        Renewable_Capacity = pd.Series(self.m.Gen_Capacity.extract_values(), index=self.m.Gen_Capacity.extract_values().keys()).unstack()
        Node_Capacities=Renewable_Capacity.sum(1).unstack()
        Electrolyser_Capacity = pd.Series(self.m.Electrolyser_Capacity.extract_values(), index=self.m.Electrolyser_Capacity.extract_values().keys())
        Node_Capacities.loc['PEM_Electrolyser']=Electrolyser_Capacity
        Node_Capacities = pd.concat([Node_Capacities,Storage_Capacity],axis=0)

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

        Node_Capacities.loc['H2_Unserved_Capacity']=H2_Unserved.max()
        Node_Capacities.to_csv(results_dir+'/Zone_Capacities.csv')

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

        if self.Pipeline_Exists is not None:
            self.Pipeline_Exists.to_csv(results_dir+'/Pipeline_Exists.csv')

if __name__=='__main__':
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