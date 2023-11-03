# Author: Steven Percy
# Date: 08/24/2022

#Toggle between importing for HPC or for local machine. 1 of these must be commented out!
from Model_Functions_v2 import *
#from Model_Functions_v2_for_HPC import *

import os
from pyomo.environ import *

print('### Running model ###')
print()
for iteration in (0,1):

    from HYPSTAT_data_inputs_231010 import *

    ## Get data:
    if iteration==0:
        #first iteration
        h=24 # this specifies the number of hours in each time interval.  It allows the conversion of kg/h when multiple hours are in each timestep.  if resampling data please change this value 
        optimize_pipelines = True
        print('Optimizing pipelines')
        print()
    else:
        #second iteration, hourly run
        h=2
        optimize_pipelines = False
        print()
        print('Performing full optimization')
        print()

    if iteration==1:
        #second iteration, need to rewrite links
        threshold = 0.1
        for link in Pipeline_Exists.index:
            if Pipeline_Exists[link] < threshold:
                #close to 0, pipeline doesn't exist
                links.loc[link,'Capacity (kg/hr)'] = 0
        print(links)

    if h != 1: #TODO: think about time resolution, for now leave as minimum 1 hour
        all_renewable_profiles = all_renewable_profiles.resample('{}h'.format(h)).sum()
        demand = demand.resample('{}h'.format(h)).sum()
    
    #TODO: some form of inputs that describe what is allowed for mapping of RE generation types to H2 production types

    ###################################################
    # Model build #
    ###################################################

    #Create model
    m = ConcreteModel()

    #Define time and node index sets
    t_dict = {ind + 1: k for ind, k in enumerate(all_renewable_profiles.index)}

    #setup model index sets
    m.T = Set(initialize=(sorted(list(t_dict.keys()))), ordered=True) # sort t_dict
    m.Nuclear_Zones = Set(initialize=list(nuclear_zones), ordered=True) #TODO: delete nuclear zones (set up as an input as part of different tech types)
    m.Zones = Set(initialize=list(all_zones), ordered=True)
    m.Links = Set(initialize=list(links.index), ordered=True)
    techs = ['Terrestrial_Wind','Offshore_Wind','Solar'] #TODO: convert to inputs (booleans for different tech types or flexible?)
    m.Techs = Set(initialize=list(techs), ordered=True) #TODO: rename for more specific to RE generators
    
    # This set defined the index for the renewable profiles.
    m.Renewable_producers = Set(initialize=list(capacities.index), ordered=True) #TODO: rename to be more clear (e.g., tranches)

    '''
    TODO: SETS THAT ARE MISSING
    -H2 production techs
    -storage techs
    -tank-based transmission techs

    -Pipelines? Maybe extra tranches
    '''


    #cost_parameters: #TODO: maybe just convert to input variables (question for Steven?)
    m.Cost_of_Unserved_H2 = Param(initialize=10e3) # We assume that a kg of unserved hydrogen has an economic cost of 10eX.
    m.h2_conversion_efficiency = Param(initialize=h2_conversion_efficiency) # kW/kg.h2
    m.nuclear_h2_conversion_efficiency = Param(initialize=nuclear_h2_conversion_efficiency) # kW/kg.h2
    m.Nuclear_LCOE = Param(initialize=Nuclear_LCOE) #$/Kwh
    m.Cavern_charge_limit = Param(initialize=(0.03/24)*h) #% of capacity per hour, from 3%/day
    m.Cavern_discharge_limit = Param(initialize=(0.1/24)*h) #% of capacity per hour, from 10%/day
    m.Tank_charge_cost = Param(initialize=0.06) #$/kg, opex for loading H2 storage tank
    m.Cavern_charge_cost = Param(initialize=0.06) #$/kg, opex for loading salt cavern storage
    if Total_cavern_capacity != 'inf':
        m.Total_cavern_capacity = Param(initialize = Total_cavern_capacity) # kg


    #Time varying variables

    #TODO: will need to add variable for specific flows of electricity to specific production types
    m.Zone_H2_Demand_Storage = Var(m.T, m.Zones)#, domain=NonNegativeReals)  #kg/per interval i.e. kg/h #TODO: (?) eliminate this variable and consolodate into mass balance
    m.H2_Storage_Discharge_Charge = Var(m.T, m.Zones) # can be positive or negative  (kg/per interval i.e. kg/h)
    m.Cavern_Storage_Discharge_Charge = Var(m.T, m.Zones) # can be positive or negative  (kg/per interval i.e. kg/h) #TODO: consolodate into single variable with additional index for storage tech
    m.H2_Storage_Level = Var(m.T, m.Zones, domain=NonNegativeReals) #(kg) 
    m.Cavern_Storage_Level = Var(m.T, m.Zones, domain=NonNegativeReals) #(kg) #TODO: consolodate into single variable with additional index for storage tech
    m.H2_Storage_Capacity = Var(m.Zones, domain=NonNegativeReals)  #(kg)
    m.Cavern_Storage_Capacity = Var(m.Zones, domain=NonNegativeReals) #(kg) #TODO: consolodate into single variable with additional index for storage tech
    m.Link_Flow = Var(m.T, m.Links)  # (kg/per interval i.e. kg/h) #TODO: create a variable for pipeline flow (maybe remove Link_Flow)
    m.Truck_Flow = Var(m.T,m.Links)
    m.Renewable_Production  = Var(m.Techs, m.T, m.Zones)  #kWh 
    m.Renewable_Capacity = Var(m.Techs,m.Zones,m.Renewable_producers, domain=NonNegativeReals)  #kw
    m.Hydrogen_Production = Var(m.T, m.Zones, domain=NonNegativeReals)   #kg/per interval #TODO: split into different types of production
    m.H2_Unserved = Var(m.T, m.Zones, domain=NonNegativeReals)#,bounds=(0,0) )   #kg/per interval
    m.H2_Overserved = Var(m.T, m.Zones, domain=NonNegativeReals)#,bounds=(0,0) )  
    m.Electrolyser_Capacity = Var(m.Zones, domain=NonNegativeReals) #TODO: split into different types of production (maybe rename)
    m.Curtailed_Renewable_Production = Var(m.Techs,m.T, m.Zones, domain=NonNegativeReals) 
    m.H2_Imports = Var(m.T, m.Zones, domain=NonNegativeReals) # imported hydrogen, kg/interval

    m.Demand_Met = Var(m.T, m.Zones, domain=NonNegativeReals)# kg per interval #TODO: rename H2_demand_met or similar
    m.Pipeline_Cost = Var(m.T, m.Links, domain=NonNegativeReals) #$/kg #TODO: rename all of these to specify OPEX
    m.Truck_Cost = Var(m.T, m.Links, domain=NonNegativeReals) #$/kg #TODO: rename all of these to specify OPEX, TODO: generalize to all tank-based transmission
    m.H2_Storage_Cost = Var(m.T, m.Zones, domain=NonNegativeReals) #$/kg for charging storage #TODO: rename all of these to specify OPEX
    m.Cavern_Storage_Cost = Var(m.T, m.Zones, domain=NonNegativeReals) #$/kg for charging storage #TODO: rename all of these to specify OPEX, TODO: consolodate storage into single variable with storage tech index set

    m.Pipeline_Capacity = Var(m.Links, domain = NonNegativeReals) # kg/period pipeline capacity for each zone
    if optimize_pipelines:
        m.Pipeline_Exists = Var(m.Links,domain=Binary) #whether or not a pipeline exists
    
    if include_nuclear_hydrogen  & (year in nuclear_hydrogen_years): #TODO: remove nuclear technology
        nuclear_profiles = all_renewable_profiles['Nuclear']
    #hourly capacity factor 
    m.Nuclear_Production  = Var(m.T, m.Nuclear_Zones)  #kWh 
    #m.Nuclear_Capacity = Var(m.Techs,m.Nuclear_Zones, domain=NonNegativeReals,bounds=(0, maximum_nuclear_capacity)) # Bound to the maximum nuclear capacity
    m.Nuclear_H2_Electrolyser_Capacity = Var(m.Nuclear_Zones, domain=NonNegativeReals, bounds=(0, maximum_nuclear_capacity)) # Bound to the maximum nuclear capacity
    m.Nuclear_H2_Production= Var(m.T, m.Nuclear_Zones, domain=NonNegativeReals) #kg

    # pipeline flow incidence matrix
    def link_rule(m, link, zone):
        return float(link_flow_direction.loc[link, zone])

    m.link_flow_direction = Param(m.Links, m.Zones, initialize=link_rule)

    print('Setting up storage constraints...')
    ## Storage_demand; adding in imports here
    ## A negative value of H2_Storage_Discharge_Charge is equiv to production; 
    # TODO: choose either fixed demand or Demand_Met variable and consolodate to 1 storage and demand rule
    def storage_demand_rule(m, t, zone):
            return ( m.Zone_H2_Demand_Storage[t,zone] == demand.loc[t_dict[t], zone] + m.H2_Storage_Discharge_Charge[t,zone] + m.Cavern_Storage_Discharge_Charge[t,zone] - m.H2_Imports[t,zone])

    def storage_demand_rule_2(m, t, zone):
            return ( m.Zone_H2_Demand_Storage[t,zone] == m.Demand_Met[t,zone] + m.H2_Storage_Discharge_Charge[t,zone] + m.Cavern_Storage_Discharge_Charge[t,zone] - m.H2_Imports[t,zone])

    m.storage_demand_constraint = Constraint(m.T, m.Zones, rule=storage_demand_rule_2)

    # Meet demand over course of day

    def demand_rule(m, t, zone):
        if t%24==1:
            return (sum(m.Demand_Met[t_step,zone] for t_step in range(t,t+24)) == sum(demand.loc[t_dict[t_step],zone] for t_step in range(t,t+24)))
        else:
            return Constraint.Skip

    def demand_rule_2(m, t, zone):
        return (m.Demand_Met[t,zone] == demand.loc[t_dict[t],zone])

    #m.demand_constraint = Constraint(m.T, m.Zones, rule=demand_rule)
    m.demand_constraint = Constraint(m.T, m.Zones, rule=demand_rule_2)

    # Limit imports

    def total_imports_rule(m):
        return sum(sum(m.H2_Imports[t,zone] for t in m.T) for zone in m.Zones) <= max_imports
    
    m.total_imports_constraint = Constraint(rule=total_imports_rule)

    # Tabling imports for now; #TODO: think about exports
    def zone_imports_rule(m, t, zone):
        if zone in import_zones:
            return Constraint.Skip
        else:
            return m.H2_Imports[t,zone] == 0
    
    m.zone_imports_constraint = Constraint(m.T,m.Zones,rule=zone_imports_rule)

    # Storage Charge Discharge, multi-period; TODO: figure out why mass balance wasn't quite closing and fix
    # TODO: clarify point within the hour that we mean for storage, transmission, etc.
    # TODO: consolodate into single storage rule with multiple storage techs
    def H2_storage_multi_period_rule(m, t, zone):
        if t == m.T.first():
            return m.H2_Storage_Level[t, zone]==m.H2_Storage_Level[m.T.last(), zone] + m.Cavern_Storage_Discharge_Charge[t,zone]
        else:
            return (m.H2_Storage_Level[t, zone] == m.H2_Storage_Level[t - 1,zone]  + m.H2_Storage_Discharge_Charge[t, zone])

    m.H2_storage_multi_period_constraint = Constraint(m.T, m.Zones, rule=H2_storage_multi_period_rule)

    def Cavern_storage_multi_period_rule(m, t, zone):
        if t == m.T.first():
            return m.Cavern_Storage_Level[t, zone]==m.Cavern_Storage_Level[m.T.last(), zone] + m.Cavern_Storage_Discharge_Charge[t,zone]
        else:
            return (m.Cavern_Storage_Level[t, zone] == m.Cavern_Storage_Level[t - 1,zone]  + m.Cavern_Storage_Discharge_Charge[t, zone])

    m.Cavern_storage_multi_period_constraint = Constraint(m.T, m.Zones, rule=Cavern_storage_multi_period_rule)

    # Cavern charge discharge speed
    # TODO: consolidate into single storage
    def Cavern_charge_rule(m, t, zone):
        return (m.Cavern_Storage_Discharge_Charge[t, zone] <= m.Cavern_charge_limit*m.Cavern_Storage_Capacity[zone])

    m.Cavern_charge_constraint = Constraint(m.T, m.Zones, rule=Cavern_charge_rule)

    def Cavern_discharge_rule(m, t, zone):
        return (m.Cavern_Storage_Discharge_Charge[t, zone] >= -m.Cavern_discharge_limit*m.Cavern_Storage_Capacity[zone])

    m.Cavern_discharge_constraint = Constraint(m.T, m.Zones, rule=Cavern_discharge_rule)

    # Storage Level
    # TODO: consolidate into single storage
    # TODO: note in documentation that storage capacities need to be working capacities
    def H2_storage_level_rule(m, t, zone):
        return (m.H2_Storage_Level[t, zone] <= m.H2_Storage_Capacity[zone])

    m.H2_storage_level_constraint = Constraint(m.T, m.Zones, rule=H2_storage_level_rule)

    def Cavern_storage_level_rule(m, t, zone):
        return (m.Cavern_Storage_Level[t, zone] <= m.Cavern_Storage_Capacity[zone])

    m.Cavern_storage_level_constraint = Constraint(m.T, m.Zones, rule=Cavern_storage_level_rule)

    #TODO: consolidate all of this into some sort of input table with limits for each storage type and zone
    #TODO: consider if we want to still have a total storage constraint
    def Tank_capacity_rule(m,zone):
        cap = tank_capacities[zone]
        if cap=='inf':
            return Constraint.Skip
        else:
            return (m.H2_Storage_Capacity[zone] <= cap)

    m.Tank_capacity_constraint = Constraint(m.Zones, rule=Tank_capacity_rule)

    def Cavern_capacity_rule(m, zone):
        cap = cavern_capacities[zone]
        if cap=='inf':
            return Constraint.Skip
        else:
            return (m.Cavern_Storage_Capacity[zone] <= cap)

    m.Cavern_capacity_constraint = Constraint(m.Zones, rule=Cavern_capacity_rule)

    def Total_cavern_capacity_rule(m):
        return (sum(m.Cavern_Storage_Capacity[zone] for zone in m.Zones) <= m.Total_cavern_capacity)

    if Total_cavern_capacity != 'inf':
        m.Total_cavern_capacity_constraint = Constraint(rule=Total_cavern_capacity_rule)

    
    print('Setting up flow constraints...')
    # Pipeline flow (update this later to be part of the variable bound)

    #TODO: have upper bound on pipeline size as an input
    if optimize_pipelines:
        def pipeline_exists_rule(m, link):
            bigM = 300000*h #kg/period, upper bound on pipeline capacity (~7000 TPD)
            return m.Pipeline_Capacity[link] <= bigM*m.Pipeline_Exists[link]
        
        m.pipeline_exists_constraint = Constraint(m.Links,rule=pipeline_exists_rule)
    
    else:
        #not optimizing pipelines, limit sizes according to links

        #TODO: think about how pipeline capacity limits (i.e., which pipelines exist) is passed in the second iteration

        def pipeline_size_rule(m, link):
            return m.Pipeline_Capacity[link] <= links.loc[link,'Capacity (kg/hr)']*h
        
        m.pipeline_size_constraint = Constraint(m.Links, rule=pipeline_size_rule)
    
    # TODO: generalize into tank-based transmission
    # TODO: think about if we want to incorporate capex/fixed opex or keep as levelized cost (via opex)
    def forward_max_truck_rule(m, t, link):
        return m.Truck_Flow[t,link] <= truck_size_limit*h #kg/period
    
    m.forward_max_truck_constraint = Constraint(m.T, m.Links, rule=forward_max_truck_rule)

    def reverse_max_truck_rule(m, t, link):
        return m.Truck_Flow[t,link] >= -truck_size_limit*h #kg/period
    
    m.reverse_max_truck_constraint = Constraint(m.T, m.Links, rule=reverse_max_truck_rule)

    def link_forward_flow_rule(m, t, link): #TODO: recast specifically with pipelines
        #return m.Link_Flow[t, link] <= links.loc[link,'Capacity (kg/hr)'] * h
        return m.Link_Flow[t, link] - m.Truck_Flow[t,link] <= m.Pipeline_Capacity[link]
        
    m.link_forward_flow_constraint = Constraint(m.T, m.Links, rule=link_forward_flow_rule)

    def link_reverse_flow_rule(m, t, link):
        #return m.Link_Flow[t, link] >= -1*links.loc[link ,'Capacity (kg/hr)'] * h
        return m.Link_Flow[t, link] - m.Truck_Flow[t,link] >= -1*m.Pipeline_Capacity[link]

    m.link_reverse_flow_constraint = Constraint(m.T, m.Links, rule=link_reverse_flow_rule)

    print('Setting up renewable zone constraints...')
    # Renewable production - this defines the renewable production at each node  in kwh
    def Renewable_Production_rule(m, tech, t, zone):
        return m.Renewable_Production[tech, t, zone] == sum( all_renewable_profiles.loc[t_dict[t],(tech,zone,producer)] * m.Renewable_Capacity[tech,zone,producer]  for producer in get_producers(capacities,zone=zone,tech = tech).index )

    m.Renewable_Production_constraint = Constraint(m.Techs, m.T, m.Zones, rule=Renewable_Production_rule)

    # Renewable build limit  - sets the capacity of each REZ to be less than a value 
    def Renewable_build_limit_rule(m, tech, zone, producer):
        # 
        capacity = get_producers(capacities,zone=zone, tech = tech)
        
        if producer in capacity.index:
            if not capacity.loc[producer]==capacity.loc[producer]:

                capacity.loc[producer] = 0
            #print(capacity.loc[producer])
            return  (m.Renewable_Capacity[tech,zone,producer] <= capacity.loc[producer] * 1000) #TODO: specify units in inputs to avoid this 1000
        else:
            return Constraint.Skip

    m.Renewable_build_limit_constraint = Constraint(m.Techs, m.Zones, m.Renewable_producers, rule=Renewable_build_limit_rule)

    # TODO: recast curtailment and electrolyzer capacity based on generalized electricity/production flows
    def Curtailed_electricity_limit_rule(m, tech, t, zone):
        return m.Curtailed_Renewable_Production[tech, t, zone] <= m.Renewable_Production[tech, t, zone]
    
    m.Curtailed_electricity_limit_constraint = Constraint(m.Techs, m.T, m.Zones, rule=Curtailed_electricity_limit_rule) #can't curtailed more electricity than is produced

    # cap renewable production at node to electrolyser capacity
    #TODO: rename this for clarity (could recast for easier understanding but probably not necessary)
    def electrolyser_capacity_limit_rule(m, t, zone):
        return (sum(m.Renewable_Production[tech, t, zone] - m.Curtailed_Renewable_Production[tech, t, zone] for tech in m.Techs)) <= m.Electrolyser_Capacity[zone]*h

    m.electrolyser_capacity_limit_constraint = Constraint(m.T, m.Zones, rule=electrolyser_capacity_limit_rule)

    ## nuclear H2 production TODO: remove this
    if (include_nuclear_hydrogen) & (year in nuclear_hydrogen_years):
        print('Including nuclear H2')
    
        def Nuclear_H2_Production_rule(m, t, zone):
            return m.Nuclear_H2_Production[t, zone] == m.Nuclear_H2_Electrolyser_Capacity[zone]*float(nuclear_profiles.loc[t_dict[t]])/m.nuclear_h2_conversion_efficiency
        m.Nuclear_H2_Production_constraint = Constraint(m.T, m.Nuclear_Zones, rule=Nuclear_H2_Production_rule)

    # New H2 production build at nodes
    # electricity is in kWh
    # TODO: generalize to various H2 production pathways (focused on electrolysis)
        # TODO: will need to think about how renewable production matches up with H2 production
        # TODO: create variables for specification of electricity flows from each RE tech to each H2 production tech
    def H2_Production_rule(m, t, zone):
        if (include_nuclear_hydrogen) & (year in nuclear_hydrogen_years) & (zone in nuclear_zones):
            return m.Hydrogen_Production[t, zone] == (sum(m.Renewable_Production[tech,t, zone]  - m.Curtailed_Renewable_Production[tech,t, zone] for tech in m.Techs))/m.h2_conversion_efficiency + m.Nuclear_H2_Production[t, zone]
        else:
            return m.Hydrogen_Production[t, zone] == (sum(m.Renewable_Production[tech,t, zone]  - m.Curtailed_Renewable_Production[tech,t, zone] for tech in m.Techs))/m.h2_conversion_efficiency

    m.H2_Production_constraint = Constraint(m.T, m.Zones, rule=H2_Production_rule)

    print('Setting up power balance constraints...')
    # H2 balance considering link flows
    def H2_Balance_rule(m, t, zone):
        links_to_this_zone=links_to_zones[zone]
        return (m.Hydrogen_Production[t, zone]  + m.H2_Unserved[t, zone] + sum(m.Link_Flow[t, link] * m.link_flow_direction[link, zone] for link in links_to_this_zone) == m.Zone_H2_Demand_Storage[t, zone] + m.H2_Overserved[t, zone] )

    m.H2_Balance_constraint = Constraint(m.T, m.Zones, rule=H2_Balance_rule)

    print('Setting up transmission cost constraints...')
    #TODO: work out in inputs somewhere how to add multipliers for links or set link costs in some specific way (or disallow links)
    def Forward_Pipeline_Cost_rule(m, t, link):
        return (m.Pipeline_Cost[t,link] >= (m.Link_Flow[t,link] - m.Truck_Flow[t,link])*links.loc[link,'Transmission Opex ($/kg)'])

    m.Forward_Pipeline_Cost_constraint = Constraint(m.T, m.Links, rule=Forward_Pipeline_Cost_rule)

    def Reverse_Pipeline_Cost_rule(m, t, link):
        return (m.Pipeline_Cost[t,link] >= -(m.Link_Flow[t,link] - m.Truck_Flow[t,link])*links.loc[link,'Transmission Opex ($/kg)'])

    m.Reverse_Pipeline_Cost_constraint = Constraint(m.T, m.Links, rule=Reverse_Pipeline_Cost_rule)

    def Forward_Truck_Cost_rule(m, t, link): #TODO: generalize to all tank transport
        return (m.Truck_Cost[t,link] >= m.Truck_Flow[t,link]*truck_cost*(link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]/100))
    
    m.Forward_Truck_Cost_constraint = Constraint(m.T, m.Links, rule=Forward_Truck_Cost_rule)

    def Reverse_Truck_Cost_rule(m, t, link):
        return (m.Truck_Cost[t,link] >= -m.Truck_Flow[t,link]*truck_cost*(link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]/100))
    
    m.Reverse_Truck_Cost_constraint = Constraint(m.T, m.Links, rule=Reverse_Truck_Cost_rule)

    print('Setting up storage cost constraints...')
    # TODO: generalize into generic storage types
    # TODO: allow for variable opex on discharge if desired by user
    def H2_Storage_Cost_rule(m, t, zone):
        return (m.H2_Storage_Cost[t,zone] >= m.H2_Storage_Discharge_Charge[t,zone]*m.Tank_charge_cost)

    m.H2_Storage_Cost_constraint = Constraint(m.T, m.Zones, rule=H2_Storage_Cost_rule)

    def Cavern_Storage_Cost_rule(m, t, zone):
        return (m.Cavern_Storage_Cost[t,zone] >= m.Cavern_Storage_Discharge_Charge[t,zone]*m.Cavern_charge_cost)

    m.Cavern_Storage_Cost_constraint = Constraint(m.T, m.Zones, rule=Cavern_Storage_Cost_rule)

    print('Setting up objective function...')
    #### Objective function
    ## FIXME: enable unserved H2

    def get_CRF(interest=0.10, years=30):
        return (interest * (1 + interest) ** years) /((1 + interest) ** years - 1)

    def objective_rule(m):
        return (sum(
                m.H2_Storage_Capacity[zone] * build_cost.loc['Tank_Storage',zone] +  #cost of tank build
                m.Cavern_Storage_Capacity[zone] * build_cost.loc['Cavern_Storage',zone] +  #cost of cavern build                                                                                                                           
                m.Electrolyser_Capacity[zone] * build_cost.loc['PEM_Electrolyser',zone] + #cost of electrolyser TODO: think about how to reference costs with inputs, generalize for H2 production
                sum(m.H2_Unserved[t, zone] * m.Cost_of_Unserved_H2 for t in m.T) + #penilty for unserved H2 TODO: think about inputs for overserved/underserved penalties
                sum(m.H2_Overserved[t, zone] * overserved_cost for t in m.T) for zone in m.Zones) + #penilty for overserved H2
                sum(sum(m.Pipeline_Cost[t,link] for t in m.T) for link in m.Links) + #opex for pipelines
                sum(sum(m.Truck_Cost[t,link] for t in m.T) for link in m.Links) + #levelized cost for trucks
                #TODO: consider variable opex for generation techs (e.g., to use LCOE)
                #TODO: consider variable opex for hydrogen production techs
                #TODO: consider CAPEX for tank-based based
                #TODO: convert pipeline costs into inputs
                sum((((m.Pipeline_Capacity[link]/h)*18.86 + 2122612*(m.Pipeline_Exists[link] if optimize_pipelines else 1))*year_ratio*link_distances.loc[link.split(' to ')[0],link.split(' to ')[1]]) for link in m.Links)*get_CRF() + #TODO: take CRF out of this and make sure it is inputs, like for all other techs
                sum(sum(m.H2_Storage_Cost[t,zone] for t in m.T) for zone in m.Zones) + #opex for storage (input only)
                sum(sum(m.Cavern_Storage_Cost[t,zone] for t in m.T) for zone in m.Zones) + #opex for storage (input only)
                sum(sum(sum(m.Renewable_Capacity[tech,zone,producer] * build_cost.loc[tech,zone] for tech in m.Techs) for zone in m.Zones) for producer in m.Renewable_producers)  + # renewable build
                #TODO: handle PTC/ITC in inputs
                -sum(sum(sum((m.Renewable_Production[tech,t,zone] - m.Curtailed_Renewable_Production[tech,t,zone])*PTC[tech] for zone in m.Zones) for t in m.T) for tech in m.Techs) + #PTC effects
                -sum(sum(sum((m.Renewable_Production[tech,t,zone])*ITC[tech] for zone in m.Zones) for t in m.T) for tech in m.Techs) + #ITC effects
                #TODO: remove nuclear cost
                sum(sum(m.Nuclear_H2_Production[t, zone] * m.nuclear_h2_conversion_efficiency * m.Nuclear_LCOE for t in m.T if (include_nuclear_hydrogen) & (year in nuclear_hydrogen_years)) + m.Nuclear_H2_Electrolyser_Capacity[zone] * build_cost.loc['Solid_Oxide_Electrolyser',zone] for zone in m.Nuclear_Zones if (include_nuclear_hydrogen) & (year in nuclear_hydrogen_years))) # nuclear H2 production - only included if include_nuclear_hydrogen==True
        #log_infeasible_constraints(m)
    m.objective = Objective(rule=objective_rule, sense=minimize)

    stream_solver=True
    print('Solving')
    #opt = SolverFactory("gurobi", solver_io="python")
    opt = SolverFactory('glpk')
    results_final = opt.solve(m, tee=stream_solver) #,options={'NonConvex':2}

    if iteration==0:
        Pipeline_Exists = pd.Series(m.Pipeline_Exists.extract_values(), index=m.Pipeline_Exists.extract_values().keys())
        print('Done with pipeline optimization')
    else:
        print('Done with hourly optimization')



# Plot Model Results
results_dir = 'test_case_v2_outputs'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

#collect useful parameters and write to csv for future reference
params = dict()
params['nuclear_h2_conversion_efficiency'] = m.nuclear_h2_conversion_efficiency.value
params['Nuclear_LCOE'] = m.Nuclear_LCOE.value
params['year_ratio'] = year_ratio
params['max_imports'] = max_imports
params['h'] = h

pd.Series(params).to_csv(results_dir+'/params.csv')
build_cost.to_csv(results_dir+'/build_cost.csv')
PTC = pd.DataFrame.from_dict(PTC,orient='index')
PTC.to_csv(results_dir+'/PTC.csv')
ITC = pd.DataFrame.from_dict(ITC,orient='index')
ITC.to_csv(results_dir+'/ITC.csv')

H2_Storage_Capacity = pd.Series(m.H2_Storage_Capacity.extract_values(), index=m.H2_Storage_Capacity.extract_values().keys())
Cavern_Storage_Capacity = pd.Series(m.Cavern_Storage_Capacity.extract_values(), index=m.Cavern_Storage_Capacity.extract_values().keys())
Renewable_Capacity = pd.Series(m.Renewable_Capacity.extract_values(), index=m.Renewable_Capacity.extract_values().keys()).unstack()
#print(Renewable_Capacity.to_frame().T)
Zone_Capacities=Renewable_Capacity.sum(1).unstack()
Electrolyser_Capacity = pd.Series(m.Electrolyser_Capacity.extract_values(), index=m.Electrolyser_Capacity.extract_values().keys())
Zone_Capacities.loc['PEM_Electrolyser']=Electrolyser_Capacity
Zone_Capacities.loc['Tank_Storage']=H2_Storage_Capacity
Zone_Capacities.loc['Cavern_Storage']=Cavern_Storage_Capacity
#print(Zone_Capacities)

if include_nuclear_hydrogen  & (year in nuclear_hydrogen_years):
    Nuclear_H2_Production = pd.Series(m.Nuclear_H2_Production.extract_values(), index=m.Nuclear_H2_Production.extract_values().keys()).unstack()
    Nuclear_Electricity_consumption=Nuclear_H2_Production*m.nuclear_h2_conversion_efficiency.value
    Nuclear_H2_Electrolyser_Capacity = pd.Series(m.Nuclear_H2_Electrolyser_Capacity.extract_values(), index=m.Nuclear_H2_Electrolyser_Capacity.extract_values().keys())
    Zone_Capacities.loc['Solid_Oxide_Electrolyser']=Nuclear_H2_Electrolyser_Capacity
    nuclear_production_lcoh=(Nuclear_Electricity_consumption*m.Nuclear_LCOE.value).sum()/Nuclear_H2_Production.sum()
    Nuclear_H2_Production.to_csv(results_dir+'/Nuclear_H2_Production.csv')

#print(Electrolyser_Capacity.to_frame().T)

#print(H2_Storage_Capacity.to_frame().T)

H2_Overserved = pd.Series(m.H2_Overserved.extract_values(), index=m.H2_Overserved.extract_values().keys()).unstack()
H2_Overserved.index=list(t_dict.values())
H2_Overserved.to_csv(results_dir+'/H2_overserved.csv')

Link_Flow = pd.Series(m.Link_Flow.extract_values(), index=m.Link_Flow.extract_values().keys()).unstack()
Link_Flow.index=list(t_dict.values())
Link_Flow.to_csv(results_dir+'/Link_Flow.csv')

H2_Storage_Level = pd.Series(m.H2_Storage_Level.extract_values(), index=m.H2_Storage_Level.extract_values().keys()).unstack()
H2_Storage_Level.index=list(t_dict.values())
H2_Storage_Level.to_csv(results_dir+'/H2_Storage_Level.csv')

Cavern_Storage_Level = pd.Series(m.Cavern_Storage_Level.extract_values(), index=m.Cavern_Storage_Level.extract_values().keys()).unstack()
Cavern_Storage_Level.index=list(t_dict.values())
Cavern_Storage_Level.to_csv(results_dir+'/Cavern_Storage_Level.csv')

Renewable_Production = pd.Series(m.Renewable_Production.extract_values(), index=m.Renewable_Production.extract_values().keys()).unstack()
Renewable_Production = pd.DataFrame(Renewable_Production)
Renewable_Production_raw = copy.copy(Renewable_Production)
Renewable_Production_raw.index = pd.MultiIndex.from_tuples([(i[0],t_dict[i[1]]) for i in Renewable_Production_raw.index],names=Renewable_Production_raw.index.names)
Renewable_Production_raw.to_csv(results_dir+'/Renewable_Production_raw.csv')
Renewable_Production_Tech = Renewable_Production.groupby(level=0).sum()
Renewable_Production = Renewable_Production.groupby(level=1).sum()
Renewable_Production.index=list(t_dict.values())
#Renewable_Production.columns = list(techs)
Renewable_Production.to_csv(results_dir+'/Renewable_Production.csv')
Renewable_Production_Tech.to_csv(results_dir+'/Renewable_Production_Tech_Total.csv')

Renewable_Capacity = pd.Series(m.Renewable_Capacity.extract_values(), index=m.Renewable_Capacity.extract_values().keys()).unstack()
Renewable_Capacity.to_csv(results_dir+'/Renewable_Capacity.csv')


H2_Storage_Discharge_Charge = pd.Series(m.H2_Storage_Discharge_Charge.extract_values(), index=m.H2_Storage_Discharge_Charge.extract_values().keys()).unstack()
H2_Storage_Discharge_Charge.index=list(t_dict.values())
H2_Storage_Discharge_Charge.to_csv(results_dir+'/H2_Storage_Discharge_Charge.csv')

Cavern_Storage_Discharge_Charge = pd.Series(m.Cavern_Storage_Discharge_Charge.extract_values(), index=m.Cavern_Storage_Discharge_Charge.extract_values().keys()).unstack()
Cavern_Storage_Discharge_Charge.index=list(t_dict.values())
Cavern_Storage_Discharge_Charge.to_csv(results_dir+'/Cavern_Storage_Discharge_Charge.csv')

H2_Unserved = pd.Series(m.H2_Unserved.extract_values(), index=m.H2_Unserved.extract_values().keys()).unstack()
H2_Unserved.index=list(t_dict.values())
H2_Unserved.to_csv(results_dir+'/H2_Unserved.csv')

Zone_Capacities.loc['H2_Unserved_Capacity']=H2_Unserved.max()
Zone_Capacities.to_csv(results_dir+'/Zone_Capacities.csv')

Hydrogen_Production = pd.Series(m.Hydrogen_Production.extract_values(), index=m.Hydrogen_Production.extract_values().keys()).unstack()
Hydrogen_Production.index=list(t_dict.values())
Hydrogen_Production.to_csv(results_dir+'/H2_Production.csv')

Curtailed_Renewable_Production = pd.Series(m.Curtailed_Renewable_Production.extract_values(), index=m.Curtailed_Renewable_Production.extract_values().keys()).unstack()
Curtailed_Renewable_Production = pd.DataFrame(Curtailed_Renewable_Production)
Curtailed_Renewable_Production_raw = copy.copy(Curtailed_Renewable_Production)
Curtailed_Renewable_Production_raw.index = pd.MultiIndex.from_tuples([(i[0],t_dict[i[1]]) for i in Curtailed_Renewable_Production_raw.index],names=Curtailed_Renewable_Production_raw.index.names)
Curtailed_Renewable_Production_raw.to_csv(results_dir+'/Curtailed_Renewable_Production_raw.csv')
Curtailed_Renewable_Production_Tech = Curtailed_Renewable_Production.groupby(level = 0).sum()
Curtailed_Renewable_Production = Curtailed_Renewable_Production.groupby(level = 1).sum()
Curtailed_Renewable_Production.index=list(t_dict.values())
Curtailed_Renewable_Production.to_csv(results_dir+'/Curtailed_Renewable_Production.csv')
Curtailed_Renewable_Production_Tech.to_csv(results_dir+'/Curtailed_Renewable_Production_Tech_Total.csv')

Pipeline_Capacity = pd.Series(m.Pipeline_Capacity.extract_values(), index=m.Pipeline_Capacity.extract_values().keys())    
Pipeline_Capacity.to_csv(results_dir+'/Pipeline_Capacity.csv')
#print(Pipeline_Capacity)

# Fix the html file 
Demand_Met = pd.Series(m.Demand_Met.extract_values(),index=m.Demand_Met.extract_values().keys()).unstack()
Demand_Met.index=list(t_dict.values())
Demand_Met.to_csv(results_dir+'/Demand_Met.csv')

H2_Imports = pd.Series(m.H2_Imports.extract_values(),index=m.H2_Imports.extract_values().keys()).unstack()
H2_Imports.index = list(t_dict.values())
H2_Imports.to_csv(results_dir+'/H2_Imports.csv')

Truck_Flow = pd.Series(m.Truck_Flow.extract_values(),index=m.Truck_Flow.extract_values().keys()).unstack()
Truck_Flow.index = list(t_dict.values())
Truck_Flow.to_csv(results_dir+'/Truck_Flow.csv') 

Truck_Cost = pd.Series(m.Truck_Cost.extract_values(),index=m.Truck_Cost.extract_values().keys()).unstack()
Truck_Cost.index = list(t_dict.values())
Truck_Cost.to_csv(results_dir+'/Truck_Cost.csv')    

if True:# optimize_pipelines:
    #Pipeline_Exists = pd.Series(m.Pipeline_Exists.extract_values(), index=m.Pipeline_Exists.extract_values().keys())    
    Pipeline_Exists.to_csv(results_dir+'/Pipeline_Exists.csv')
    print(Pipeline_Exists)

print()
print('Done!')

