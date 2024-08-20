import pandas as pd
import numpy as np
import copy
import yaml
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def get_CRF(interest=0.1, years=30):
    return (interest * (1 + interest) ** years) /((1 + interest) ** years - 1)

def multiindex_todatetime(df,index_levels,inplace=False):
    #converts the specified levels (by numeric index, not name) of df's multiindex to datetime indices
    new_index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(df.index.get_level_values(level=lv)) if lv in index_levels else df.index.get_level_values(level=lv) for lv in range(len(df.index.names))],
        names=df.index.names
    )
    
    if inplace:
        df.index = new_index
        return
    else:
        new_df = copy.deepcopy(df)
        new_df.index = new_index
        return new_df

class Scenario:

    def __init__(self,results_dir,scen_name,year,yaml_path,can_resample=True,scen_num=None):
        self.year = year
        self.scen_name = scen_name
        self.yaml_path = yaml_path
        self.scen_num = scen_num

        #read in model parameters
        self.params = pd.read_csv(results_dir+'/params.csv',index_col=0).squeeze('columns')
        self.build_cost = pd.read_csv(results_dir+'/build_cost.csv',index_col=0)

        #read in model results
        self.H2_Overserved = pd.read_csv(results_dir+'/H2_overserved.csv',index_col=0)
        self.Link_Flow = pd.read_csv(results_dir+'/Link_Flow.csv',index_col=0)
        self.Pipeline_Flow = pd.read_csv(results_dir+'/Pipeline_Flow.csv',index_col=0)
        self.Storage_Level = pd.read_csv(results_dir+'/Storage_Level.csv',index_col=[0,1])
        self.Renewable_Production = pd.read_csv(results_dir+'/Renewable_Production.csv',index_col=0)
        self.Storage_Discharge_Charge = pd.read_csv(results_dir+'/Storage_Discharge_Charge.csv',index_col=[0,1])
        self.H2_Unserved = pd.read_csv(results_dir+'/H2_Unserved.csv',index_col=0)
        self.Zone_Capacities = pd.read_csv(results_dir+'/Zone_Capacities.csv',index_col=0)
        self.H2_Production = pd.read_csv(results_dir+'/Hydrogen_Production.csv',index_col=0)
        self.Demand_Met = pd.read_csv(results_dir+'/Demand_Met.csv',index_col=0)
        self.Renewable_Capacity = pd.read_csv(results_dir+'/Renewable_Capacity.csv',index_col=[0,1])

        self.Curtailed_Renewable_Production = pd.read_csv(results_dir+'/Curtailed_Renewable_Production.csv',index_col=0)
        self.H2_Imports = pd.read_csv(results_dir+'/H2_Imports.csv',index_col=0)

        self.Pipeline_Capacity = pd.read_csv(results_dir+'/Pipeline_Capacity.csv',index_col=0)
        self.Truck_Flow = pd.read_csv(results_dir+'/Truck_Flow.csv',index_col=0)

        self.PTC = pd.read_csv(results_dir+'/PTC.csv',index_col=0)['0']
        self.ITC = pd.read_csv(results_dir+'/ITC.csv',index_col=0)['0']
        
        self.Renewable_Production_Tech = pd.read_csv(results_dir+'/Renewable_Production_Tech_Total.csv',index_col=0)
        
        self.Curtailed_Renewable_Production_Tech = pd.read_csv(results_dir+'/Curtailed_Renewable_Production_Tech_Total.csv',index_col=0)
        
        self.Renewable_Production_raw = pd.read_csv(results_dir+'/Renewable_Production_raw.csv',index_col=[0,1])
        self.Curtailed_Renewable_Production_raw = pd.read_csv(results_dir+'/Curtailed_Renewable_Production_raw.csv',index_col=[0,1])

        if can_resample:
            self.to_datetime()

        #calculate actual used electricity
        self.Electricity_Used_raw = self.Renewable_Production_raw - self.Curtailed_Renewable_Production_raw
        self.Electricity_Used = self.Electricity_Used_raw.groupby(level=[1]).sum()

        self.mass_balance_check()
        self.get_input_params()
    
    def __str__(self):
        return '{scen} in {y}'.format(scen=self.scen_name,y=self.year)
    
    def get_input_params(self):
        yaml_path = Path(self.yaml_path)
        with yaml_path.open('r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
         
        self.FinancialFiles_paths = yaml_data.get('FinancialFile', [])
        self.financials = pd.read_csv(self.FinancialFiles_paths[0],index_col=[0,1])
        self.H2DeliveryFile_paths = yaml_data.get('H2DeliveryFile', [])
        self.transport_inputs = pd.read_csv(self.H2DeliveryFile_paths[0],index_col=[0,1])

        self.H2StorageFile_paths = yaml_data.get('H2StorageFile', [])
        self.storage_inputs = pd.read_csv(self.H2StorageFile_paths[0],index_col=[0,2])
        self.ProductioncostFiles_paths = yaml_data.get('ProductioncostFile', [])
        self.prod_inputs = pd.read_csv(self.ProductioncostFiles_paths[0],index_col=[0,1]) #NOTE: this can change between runs; will eventually need to make sure we pull the right one. But for now only reading OPEX which is the same for all runs  
        self.REopexFiles_paths = yaml_data.get('REcostProfileFile', [])
        self.grid_prices = pd.read_csv(self.REopexFiles_paths['Grid'],index_col = 0)

        self.grid_prices.index = pd.to_datetime(self.grid_prices.index)
        self.grid_prices = self.grid_prices.resample('{}h'.format(int(self.params['h']))).mean()
        self.grid_prices.insert(1,'Price [$/kWh]',self.grid_prices['Price ($/MWh)']/1000) 

        self.Networks_paths = yaml_data.get('NetworksFiles', [])
        self.pipeline_cost_alpha = yaml_data.get('pipeline_cost_alpha',None) #[$/(kg/hr)-km]
        self.pipeline_cost_beta = yaml_data.get('pipeline_cost_beta',None) #[$/km]
        self.pipeline_cost_coeff = yaml_data.get('pipeline_cost_coeff',None)
        self.pipeline_cost_power = yaml_data.get('pipeline_cost_power',None) 

    def mass_balance_check(self):
        self.mb = self.H2_Production.sum().sum() + self.H2_Imports.sum().sum() + self.H2_Unserved.sum().sum() - self.Demand_Met.sum().sum() - self.H2_Overserved.sum().sum()
        self.rel_mb = abs(self.mb)/self.Demand_Met.sum().sum()
        threshold = 0.001
        if self.rel_mb > threshold:
            print('Case {} in {} mass balance does not close: relative error of {:.5f}%'.format(self.scen_name,self.year,self.rel_mb*100))

    def to_datetime(self):
        for df in [self.H2_Overserved,self.Link_Flow,self.Renewable_Production,self.H2_Unserved,self.H2_Production,
                   self.Demand_Met,self.Curtailed_Renewable_Production,self.Truck_Flow,self.H2_Imports,self.Pipeline_Flow]:
            df.index = pd.to_datetime(df.index)
        
        for df in [self.Storage_Discharge_Charge, self.Storage_Level, self.Curtailed_Renewable_Production_raw,self.Renewable_Production_raw]:
            multiindex_todatetime(df,[1],inplace=True)
        

    
    def get_capacities_w_units(self):
        capacities = copy.copy(self.Zone_Capacities)
        units = {
            'Offshore_Wind':'kW',
            'Solar':'kW',
            'Onshore_Wind':'kW',
            'PEM_Electrolyzer': 'kW',
            'Tank_Storage':'kg',
            'Geologic_Storage':'kg',
            'H2_Unserved_Capacity':'kg'
        }
        for tech in capacities.index:
            if units[tech] == 'kW':
                capacities.loc[tech] = ['{} MW'.format(round((x if not np.isnan(x) else 0)/1000)) for x in capacities.loc[tech].values]
            elif units[tech] == 'kg':
                capacities.loc[tech] = ['{} kT'.format(round((x if not np.isnan(x) else 0)/1e6,2)) for x in capacities.loc[tech].values]

        return capacities
    
    def get_transport_costs(self):
        #calculate pipeline capex
        self.pipeline_costs = copy.copy(self.Pipeline_Capacity)
        self.pipeline_costs.columns = ['Capacity [kg/pd]']
        self.pipeline_costs.insert(len(self.pipeline_costs.columns),'Capacity [kg/hr]',self.pipeline_costs['Capacity [kg/pd]']/self.params['h'])

        def calc_pipe_costs(caps):
            return self.pipeline_cost_coeff*(caps**self.pipeline_cost_power)
        
        link_distances=pd.read_csv(self.Networks_paths[0], index_col=0)

        self.pipeline_costs.insert(len(self.pipeline_costs.columns),'CAPEX [$/km]',calc_pipe_costs(self.pipeline_costs['Capacity [kg/hr]']))

        self.pipeline_costs.insert(len(self.pipeline_costs.columns),'CAPEX [$]',self.pipeline_costs['CAPEX [$/km]']*link_distances['Pipeline distance [km]'])
        self.pipeline_costs.fillna(0,inplace=True)
        self.pipeline_costs.insert(len(self.pipeline_costs.columns),'Fixed OPEX [$/yr]',self.pipeline_costs['CAPEX [$]']*0.01)

        self.pipeline_costs.insert(len(self.pipeline_costs.columns),'Annuitized CAPEX [$/yr]',
                                   self.pipeline_costs['CAPEX [$]']*get_CRF(self.financials.loc[('Pipelines',self.year),'WACC'],self.financials.loc[('Pipelines',self.year),'Recovery_time (years)']))
        
        self.pipeline_costs.insert(len(self.pipeline_costs.columns),'Var OPEX [$/kg]',self.transport_inputs.loc[(self.year,'Pipeline'),'Variable OPEX ($/kg/100-km)']*link_distances['Pipeline distance [km]']/100)
        self.pipeline_costs.fillna(0,inplace=True)

        self.pipeline_costs.insert(len(self.pipeline_costs.columns),'Total pipeline flow [kg/yr]',self.Pipeline_Flow.abs().sum(axis=0))
        self.pipeline_costs.insert(len(self.pipeline_costs.columns),'Total var OPEX [$/yr]',self.pipeline_costs['Var OPEX [$/kg]']*self.pipeline_costs['Total pipeline flow [kg/yr]'])

        self.truck_costs = pd.DataFrame(self.Truck_Flow.abs().sum(axis=0),columns=['Annual truck flow [kg/yr]'])
        self.truck_costs.insert(len(self.truck_costs.columns),'Link truck cost [$/kg]',self.transport_inputs.loc[(self.year,'Truck'),'LCOT ($/kg/100-km)']*link_distances['Truck distance [km]']/100)
        self.truck_costs.fillna(0,inplace=True)

        self.truck_costs.insert(len(self.truck_costs.columns),'Annual truck cost [$/yr]',self.truck_costs['Annual truck flow [kg/yr]']*self.truck_costs['Link truck cost [$/kg]'])
        
        self.pipeline_annual_cost = self.pipeline_costs['Annuitized CAPEX [$/yr]'] + self.pipeline_costs['Fixed OPEX [$/yr]'] + self.pipeline_costs['Total var OPEX [$/yr]']
        self.truck_annual_cost = self.truck_costs['Annual truck cost [$/yr]']

    def get_storage_costs(self):
        storage_techs = self.Storage_Discharge_Charge.index.get_level_values(0).unique()
        count=len(storage_techs)
        self.storage_cost = (self.build_cost*self.Zone_Capacities).loc[storage_techs].T
        self.storage_cost.columns = pd.MultiIndex.from_arrays([storage_techs,['Annual CAPEX+fixed OPEX [$/yr]']*count])
        
        #opex only on charging
        total_charge = self.Storage_Discharge_Charge.clip(lower=0).groupby(level=0).sum().T
        total_charge.columns = pd.MultiIndex.from_arrays([total_charge.columns,['Annual charge [kg/yr]']*count])
        self.storage_cost = pd.concat([self.storage_cost,total_charge],axis=1)
        self.storage_cost.sort_index(axis='columns',inplace=True)
       
        for tech in storage_techs:
            self.storage_cost.insert(0,(tech,'Annual var OPEX [$/yr]'),self.storage_cost[(tech,'Annual charge [kg/yr]')]*self.storage_inputs.loc[(self.year,tech),'OPEX ($/kg)'])
        
        self.storage_cost.sort_index(axis='columns',inplace=True)
        self.storage_annual_costs = pd.concat([self.storage_cost[(tech,cost_field)] for tech in storage_techs for cost_field in ['Annual var OPEX [$/yr]','Annual CAPEX+fixed OPEX [$/yr]']],
                                             axis=1).groupby(level=0,axis='columns').sum()
    
    def get_prod_costs(self):
        storage_techs = self.Storage_Discharge_Charge.index.get_level_values(0).unique()
        self.prod_costs = self.build_cost*self.Zone_Capacities
        #remove non-prod cost items
        self.prod_costs.drop(labels=storage_techs,axis='index',inplace=True)
        self.prod_costs.drop(labels=['H2_Unserved_Capacity'],axis='index',inplace=True)
        self.prod_costs = self.prod_costs.T #$/yr CAPEX + fixed OPEX        
        self.PTC_total_saving = (self.Renewable_Production_Tech-self.Curtailed_Renewable_Production_Tech).mul(self.PTC,axis='index')
        self.ITC_total_saving = self.Renewable_Production_Tech.mul(self.ITC,axis='index')
        
        #calculate grid opex
        self.grid_costs = self.Electricity_Used_raw.xs('Grid').mul(self.grid_prices['Price [$/kWh]'],axis=0).sum(axis=0)
        self.zone_lcoe = self.grid_costs/self.Electricity_Used_raw.xs('Grid').sum(axis=0)
        self.grid_lcoe = self.grid_costs.sum()/self.Electricity_Used_raw.xs('Grid').sum().sum()

        #calculate lcoe
        self.elec_annual_cost = (self.build_cost.loc[self.Electricity_Used_raw.index.get_level_values(0).unique()] * \
                            self.Zone_Capacities.loc[self.Electricity_Used_raw.index.get_level_values(0).unique()])
        self.elec_annual_cost_w_ira = self.elec_annual_cost - self.PTC_total_saving - self.ITC_total_saving

        self.re_lcoe_tech = self.elec_annual_cost/self.Electricity_Used_raw.groupby(level=0).sum()

        self.re_annual_cost = self.elec_annual_cost.drop(['Grid'],axis='index').sum(axis='index')
        self.nw_re_lcoe_tech = self.elec_annual_cost.sum()/self.Electricity_Used_raw.groupby(level=0).sum().sum()

        self.re_lcoe_tech_w_ira = self.elec_annual_cost_w_ira/self.Electricity_Used_raw.groupby(level=0).sum()
        
        self.nw_re_lcoe_tech_w_ira = self.elec_annual_cost_w_ira.sum()/self.Electricity_Used_raw.groupby(level=0).sum().sum()

        self.re_annual_used = self.Electricity_Used_raw.groupby(level=0).sum().drop(['Grid'],axis='index').sum(axis='index')     
        
        #remove super low capacity numbers so that we don't get artificial LCOEs
        total_re_capacity = self.Zone_Capacities.loc[(self.Zone_Capacities.index.isin(self.re_lcoe_tech.index)) & \
                                       (self.Zone_Capacities.index != 'Grid')].sum(axis='index')
        threshold = 0.1
        self.re_annual_used.loc[total_re_capacity<threshold] = 0

        self.re_lcoe = self.re_annual_cost/self.re_annual_used

        self.nw_re_lcoe = self.re_annual_cost.sum()/self.re_annual_used.sum()
        
        #calculate compression opex
      
        #self.comp_opex = self.H2_Production.sum(axis=0)*self.prod_inputs.loc[(self.year,'PEM Electrolyzer'),'Variable OPEX ($/kg)']

    def get_LCOH_table(self):
        self.get_prod_costs()
        self.get_storage_costs()
        self.get_transport_costs()

        self.prod_lcoh = self.prod_costs.divide(self.H2_Production.sum(axis='index'),axis='index')
        
        #remove rows with very little production, could have artificial H2 costs
        threshold = 0.1
        self.prod_lcoh.loc[self.H2_Production.sum(axis='index')<threshold] = np.nan

        self.prod_lcoh.insert(len(self.prod_lcoh.columns),'Total',self.prod_lcoh.sum(axis=1))
        
        self.all_annual_costs = pd.Series()
        self.all_annual_costs = pd.concat([self.all_annual_costs,self.prod_costs.sum(axis=0)])
        self.all_annual_costs['Grid'] += self.grid_costs.sum()
        #self.all_annual_costs['Compression'] = self.comp_opex.sum()
        self.all_annual_costs = pd.concat([self.all_annual_costs,self.storage_annual_costs.sum(axis=0)])
        self.all_annual_costs['Pipelines'] = self.pipeline_annual_cost.sum()
        self.all_annual_costs['Trucks'] = self.truck_annual_cost.sum()
        self.all_annual_costs['Renewable_PTC'] = -self.PTC_total_saving.sum().sum()
        self.all_annual_costs['Renewable_ITC'] = -self.ITC_total_saving.sum().sum()
        self.total_annual_cost = self.all_annual_costs.sum()
        self.all_annual_costs['Total'] = self.total_annual_cost
        
        try:
            for tech in self.all_annual_costs:
                if tech=='Geologic' or tech=='Tank' or tech == 'Pipelines' or tech == 'Trucks':
                    denom = self.Demand_Met.sum().sum()
                else:
                    denom = self.Demand_Met.sum().sum()-self.H2_Imports.sum().sum()
        except:
            denom = self.Demand_Met.sum().sum()
            
        self.lcoh = self.all_annual_costs/denom

    def get_RE_CF_table(self): # change ___ for tech
        self.re_CF = self.Renewable_Production_raw.groupby(level=0).sum().T/(self.Zone_Capacities.T[['Offshore_Wind','Solar','Onshore_Wind']]*8760)
        self.nw_re_CF = self.Renewable_Production_raw.groupby(level=0).sum().T.sum()/(self.Zone_Capacities.T[['Offshore_Wind','Solar','Onshore_Wind']].sum()*8760)
   

    
    def get_CF_table(self): # change this PEM_Electrolyser
        print('zone capacity',self.Zone_Capacities)
        self.elec_CF = self.Electricity_Used.sum(axis=0)/(self.Zone_Capacities.loc['PEM_Electrolyser']*8760)
        threshold = 0.01
        for zone in self.elec_CF.index:
            if self.Zone_Capacities.loc['PEM_Electrolyser',zone] < threshold:
                self.elec_CF.loc[zone] = np.nan
        self.nw_elec_CF = self.Electricity_Used.sum().sum()/(self.Zone_Capacities.loc['PEM_Electrolyser'].sum()*8760)
        #print(self.elec_CF)

    def get_capacity_ratios(self):
        techs = self.Renewable_Production_raw.index.get_level_values(0).unique().drop('Grid')
        
        self.capacity_ratios = (self.Zone_Capacities.loc[techs]/self.Zone_Capacities.loc['PEM_Electrolyser']).T


LCOHs = pd.DataFrame() # levelized cost of H2 table
CFs = pd.DataFrame()   # capacity factor table
LCOEs = pd.DataFrame() # levelized cost of energy table
rw_LCOEs = dict()      # region-wide levelized cost of energy
rw_CFs = dict()        # region-wide capacity factor 
cap_ratio = pd.DataFrame() #capacity ratio

runs=['19']
#/Users/yli6/Desktop/NREL/Project/HYPSTAT/HYPSTAT_github/HYPSTAT/Case_Study/Outputs/test1/params.csv
year = 2050
for run_num in runs:
    scen = Scenario(
    results_dir='../HYPSTAT/Case_Study/Outputs/test{rn}'.format(rn=run_num),
    scen_num=run_num,
    scen_name='Run {}'.format(run_num),
    year=year,
    yaml_path='../HYPSTAT/Case_Study/Case_Study_Scenario.yaml',
    can_resample=True)

    scen.get_LCOH_table()
    scen.get_CF_table()
    scen.get_capacity_ratios()
    scen.get_RE_CF_table()

    LCOHs = pd.concat([LCOHs,scen.lcoh],axis='columns')
    CFs = pd.concat([CFs,scen.elec_CF],axis='columns')
    LCOEs = pd.concat([LCOEs,scen.re_lcoe],axis='columns')
    cap_ratio = pd.concat([cap_ratio,scen.capacity_ratios],axis='columns')
    rw_CFs[run_num] = scen.nw_elec_CF
    rw_LCOEs[run_num] = scen.nw_re_lcoe

LCOHs.columns = runs
CFs.columns = runs
LCOEs.columns = runs

print(cap_ratio)
print(CFs)
print(LCOHs)
LCOEs.loc['nw'] = pd.Series(rw_LCOEs)
cf_df = pd.concat([CFs.min(axis=0),CFs.max(axis=0)],axis=1).T
cf_df.index = ['min','max']
cf_df.loc['nw'] = pd.Series(rw_CFs)
print(LCOEs)
print(cf_df)

