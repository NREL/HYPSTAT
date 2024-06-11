# Author: Steven Percy
# Date: 08/24/2022

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import copy

def get_renewable_profiles(drop_capacity_below=False,year=2030,techs = ['Terrestrial_Wind','Offshore_Wind','Solar'],path='Temporal/NREL_data/Temporal_Supply_Files_with_Clustering_Limited_final_221118'):
    """This function loads the profile data and returns it as one date multi-index data frame


    Args:
        drop_capacity_below (bool, optional): set to filter profiles that have a small capacity. Defaults to False.
        year (int, optional): the year to load data. Defaults to 2030.
        techs (list, optional): list of technologies. Defaults to ['Terrestrial_Wind','Offshore_Wind','Solar'].
        path (str, optional): path where data is stored. Defaults to 'NREL_data\Temporal Supply Files with Clustering'.

    Returns:
        dataframe: multi index data frame
    """    
    
    #get list of all files from folder
    import glob
    files=(glob.glob(path+ "/*"))
    all_renewable_profiles=[]
    #loop through techs
    all_capacities=[]
    for tech in techs:

        tech=tech.replace('_',' ')
        tech_profiles=[]
        tech_files=[f for f in files if (( tech in f ) and (str(year) in f))]
        #print(tech_files)
        zones_from_files=[]
        # create data from for each tech
        for f in tech_files:
            zone=(f.split('_Zone')[1].split('_')[0])
            profile=pd.read_csv(f,index_col=0)
            if len(profile)>0:
                zones_from_files.append(zone)
                profile.columns=[int(i) for i in profile.columns.astype(float)]
                if drop_capacity_below:
                    profile=profile.loc[:,profile.columns>drop_capacity_below]
                tech_profiles.append(profile)

        #concat tech dataframe
        tech_profiles=pd.concat(tech_profiles,axis=1,keys=zones_from_files)
        #return tech_profiles
        capacities=get_capacity_frame(tech_profiles)
        all_capacities.append(capacities)
        tech_profiles=index_profiles(tech_profiles)
        all_renewable_profiles.append(tech_profiles)
    
    #concat all data with column index set to the technology
    all_renewable_profiles=pd.concat(all_renewable_profiles,axis=1,keys=techs)
    all_capacities=pd.concat(all_capacities,axis=1,keys=techs)
    #capacities=get_capacity_frame(all_renewable_profiles)
    # This code removes the capacity from the dataframe and replaces it with 
    # an index that matches the capacities data frame
    all_renewable_profiles.index=pd.to_datetime(str(year),yearfirst=True) + pd.to_timedelta(all_renewable_profiles.index, unit='h')
    return all_renewable_profiles,all_capacities

def index_profiles(all_renewable_profiles):
    '''
    This function replaces the capacities with indexes 1 to x 
    '''
    all_renewable_profiles=all_renewable_profiles.T.reset_index()
    new_index=all_renewable_profiles.groupby('level_0').cumcount()
    all_renewable_profiles.level_1=new_index
    all_renewable_profiles=all_renewable_profiles.set_index(['level_0','level_1']).T
    return all_renewable_profiles

def get_capacity_frame(all_renewable_profiles):
    capacities=pd.DataFrame([list(i) for i in (all_renewable_profiles.columns)],columns=['zone','capacity'])
    capacities=capacities[['zone','capacity']]
    capacities.index=capacities.groupby('zone').cumcount()
    capacities = capacities.pivot(columns='zone')
    capacities=capacities.droplevel(0,axis=1)
    return capacities

def get_links(path,scenario,year,unconstrained=False):
    """This function returns the links and flow direciton matrix for a link file csv

    Args:
        file (str, optional): file name stored in 'NREL_data/Network_configurations/. Defaults to 'sample_links.csv'.

    Returns:
        _type_: _description_
    """ 
    
    if unconstrained:
            links=pd.read_csv('../Test case data/unlimited_links.csv').set_index('Link')
    else:
        #links=pd.read_csv('Temporal/NREL_data/Network_configurations/'+str(year)+'_'+scenario+'_links.csv').set_index('Link')
        links=pd.read_csv(path).set_index('Link')
    # unique() function is used to find the unique elements of an array. 
    all_zones=links[['End zone','Start zone']].stack().unique()
    # ~ for not, non-empty links is then recored and tracked
    links=links.loc[~links['Delivery Method'].isna()]
    flow_direction=get_link_flow_direction(links.index,seperator = ' to ')
    flow_direction=flow_direction.reindex(all_zones,axis=1).fillna(0)
    links_to_zones={}
    links_to_zones=pd.concat([links['End zone'],links['Start zone']]).reset_index().set_index(0)
    links_to_zones=dict(links_to_zones.Link.groupby(links_to_zones.index).apply(set))
     
    return flow_direction, links, all_zones,links_to_zones

def get_link_flow_direction(links,seperator = ' to '):
    '''
    This function returns the flow direction matrix, used in the get_links function
    '''
    link_flow_direction = pd.DataFrame(index=links, columns=[], data=0)

    for link in links:
        # Get forward and reverse zones
        fn, tn = link.split(seperator)
        link_flow_direction.loc[link, fn] = -1
        link_flow_direction.loc[link, tn] = 1
    return link_flow_direction.fillna(0)

def get_demand(all_renewable_profiles,year=2030, daily_demand=True,freq='h', scenario='Best_Guess'):
    """this function returns the demand for a scenario and year

    Args:
        year (int, optional): year of analysis. Defaults to 2030.
        daily_demand (bool, optional): divide demand by number of days in month. Defaults to True.
        scenario (str, optional): scenario key. Defaults to 'Best_Guess'.
        freq=frequency of data 
    Returns:
        _type_: _description_
    """  
    #demand=pd.read_csv('Temporal/NREL_data/Demand_profiles/Profiles_220907/'+scenario+'_demand.tsv', sep='\t')
    demand=pd.read_csv('../Test case data/Demand profiles/'+scenario+'_demand.tsv', sep='\t')
    demand=demand.loc[demand.Year==year]
    #return demand
    demand['total_demand']=demand[['Fueling-Station Demand [kg]','Non-Fueling-Station Demand [kg]']].sum(1)
    #return demand
    demand=demand.set_index(['Period','Network ID'])['total_demand'].unstack()#.sort_index()
    demand.index=demand.index.str.replace('M','').astype(int)
    demand=demand.sort_index()
    
    
    if daily_demand:
        demand.index='1-'+demand.index.astype(str)+'-'+str(year)
        demand.index=pd.to_datetime(demand.index, dayfirst=True)
        periods_in_month=all_renewable_profiles.resample(freq).first().resample('MS').count().iloc[:,0]
        #print(periods_in_month)
        demand=pd.concat([demand[d]/periods_in_month for d in demand.columns],axis=1,keys=demand.columns)
        demand=demand.reindex(all_renewable_profiles.resample(freq).first().index).ffill()
        
    return demand

def get_producers(capacities,zone='A',tech = 'Terrestrial_Wind'):
    '''
    This function allows for the creation of a list of the max capacity of all profile for a node
    It handles the case where a zone can't have a particular technology by returning an empty series
    '''
    if zone in capacities[tech].columns:
        max_capacity=capacities[(tech,zone)]
    else:
        #return empty series if node does not exist in 'capacities'
        max_capacity=pd.Series(dtype='float64')
    return max_capacity.dropna()

def get_producers_tech(tech_capacities,zone='A'):
    '''
    This function allows for the creation of a list of the max capacity of specific technology for a node
    It handles the case where a zone can't have a particular technology by returning an empty series
    '''
    if zone in tech_capacities.columns:
        max_capacity=tech_capacities[(zone)]
    else:
        #return empty series if node does not exist in 'capacities'
        max_capacity=pd.Series(dtype='float64')
    return max_capacity.dropna()


def drop_dup(df3,keep='first',axis=0):
    '''This drops duplicated indicies'''
    if axis==1:
        return df3[:,~df3.columns.duplicated(keep=keep)]
    elif axis==0:
        return df3[~df3.index.duplicated(keep=keep)]

def get_build_cost_matrix(interest=0.10,payback_years=30, payback_years_tank=30,payback_years_electrolyzer =40, year=2030,file='Build_Cost_Inputs_elec_cost_900.csv',all_zones=['A','BN', 'BS', 'CN', 'CS', 'E','EN','F','GE','GW','HI','J','K'],includ_interconnection_cost = True):
    '''
    This function returns a matrix with the annualised build cost for each node
    '''
    #technology_build=pd.read_csv('/Users/yli6/Desktop/NREL/Project/NYSERDA Project/NYSERDA Task 1-3/Scenario Tool/Scenario_Tool_Github/NYSERDA_Scenario_Tool/Temporal/NREL_data/Cost_data/'+file)
    technology_build=pd.read_csv('../Test case data/'+file)
    technology_build=technology_build.set_index(['Year','Tech'])
    technology_build=drop_dup(technology_build)
    build_cost=pd.DataFrame()
        
    for tech in technology_build.index.get_level_values(1).unique():
        for zone in all_zones:
            if (year,tech) in technology_build.index:
                if tech == 'PEM Electrolyser':
                    build_cost.loc[tech,zone]=get_annuity(technology_build.loc[(year,tech),'CAPEX($/kW)'], interest, payback_years_electrolyzer) + technology_build.loc[(year,tech),'OPEX($/kW-yr)']
                elif tech == 'Solid Oxide Electrolyser':
                    build_cost.loc[tech,zone]=get_annuity(technology_build.loc[(year,tech),'CAPEX($/kW)'], interest, payback_years_electrolyzer) + technology_build.loc[(year,tech),'OPEX($/kW-yr)']
                else:
                    if includ_interconnection_cost == True:
                        build_cost.loc[tech,zone]=get_annuity((technology_build.loc[(year,tech),'CAPEX($/kW)']+technology_build.loc[(year,tech),'InterconnectionCost($/kW)']), interest, payback_years) + technology_build.loc[(year,tech),'OPEX($/kW-yr)'] 
                    else:
                        build_cost.loc[tech,zone]=get_annuity(technology_build.loc[(year,tech),'CAPEX($/kW)'], interest, payback_years) + technology_build.loc[(year,tech),'OPEX($/kW-yr)'] 
        
    build_cost.loc['Tank Storage']=get_annuity(900, interest, payback_years_tank)
    build_cost.loc['Cavern Storage']=get_annuity(35, interest,payback_years_tank)
    build_cost.index=build_cost.index.str.replace(' ','_')
    
    return build_cost

def get_annuity(capex, interest, years):
    '''
    The function provides the annual repayment formula to calculate the annual payment amount for your loan. 
    '''
    #if interest>1:
    #    interest = interest / 100  # conversion from %
    an = capex  * (interest * (1 + interest) ** years) /((1 + interest) ** years - 1)
    
    return an

def plot(df,mode='scatter', bar_mode=None,line_mode='lines',yTitle='',xTitle='',horizontal_legend=True,template='simple_white',colour_set='nipy_spectral',line_shape=None,title=None,
stackgroup=None,xlim=None,ylim=None, VEPC_image=False, fade=False,
legend_shift=None,n=False,x_tick_interval=False,y_tick_interval=False,
colour_shift=0,Fuel_colours=True,bargap=None,barline=0,bin_size=None,
 boxmean=True,size=None,dropna=False,hist_remove_zeros=True,showlegend=True,
 notched=True,line_hist_mean=True,custom_colours=False):
        '''This is a flexible plotting function use mode to define plot type'''
        
        fig = go.Figure()
        
        try:
            df=df.to_frame()
        except:
            a=1
        df.columns=df.columns.astype(str)
        columns=df.columns
        if colour_set:
            col=get_colours_for_labels_n(columns,colour_set=colour_set,n=n,colour_shift=colour_shift,Fuel_colours=Fuel_colours,fade=fade,custom_colours=custom_colours)
        
            
        if mode=='bar':
            
             for trace in list(df.columns):
                fig.add_trace(
                    go.Bar(x=df.index, y=df[trace], name=trace.replace('_',' '),marker=dict(color=col[trace],line=dict(width=barline, color=col[trace])))
                )
                fig.update_layout(barmode=bar_mode)
        elif mode=='hist':
            
            fig = go.Figure()
            for trace in df.columns:
                if bin_size:
                    xbins=dict(
                                      start=int(df.min().min())-1,
                                      end=int(df.max().max())+1,
                                      size=bin_size)
                else:
                    xbins=None
                fig.add_trace(go.Histogram(x=df[trace], xbins=xbins,marker=dict(color=col[trace]),name=trace.replace('_',' ')))
            # Overlay both histograms
            if len(df.columns)>1:
                fig.update_layout(barmode='overlay')
                # Reduce opacity to see both histograms
                fig.update_traces(opacity=0.6)
        elif mode=='box':
            
            fig = go.Figure()
            for trace in df.columns:
                
                fig.add_trace(go.Box(x=df[trace].dropna(),boxmean=boxmean, notched=notched,fillcolor=col[trace],line=dict(width=1,color='black'),marker=dict(size=1,color='red'),name=trace.replace('_',' '),showlegend=False))
            # Overlay both histograms
            
                
        elif mode=='line_hist':
            fig = go.Figure()
            counts=[]
            for trace in df.columns:
                if bin_size:
                    bin_size=bin_size
                else:
                    bin_size=(df.min().min()-df.max().max())/20
                bins=np.arange(int(df.min().min()),int(df.max().max()),(bin_size))
                count, index = np.histogram(df[trace].dropna(), bins=bins)

                #index2=np.where(count==0, np.nan, index[1:])
                if hist_remove_zeros:
                    count=np.where(count==0, np.nan, count)
                #count[np.where(index2[1:] == np.nanmin(index2[1:]))[0]]=0
                #count[np.where(index2[1:] == np.nanmax(index2[1:]))[0]+1]=0
                #print(index)
                #print(count)
                fig.add_trace(go.Scatter(x=index, showlegend=showlegend, y = count,name=trace.replace('_',' '),
                          line=dict(color=col[trace], width = 2, shape='hvh'),mode='lines'))
                if line_hist_mean:
                        
                        mean=df[trace].mean()
                        
                        fig.add_trace(
                        go.Scatter(x=[mean,mean], y=[0, pd.Series(count).max()*1.1], 
                                   mode='lines',stackgroup=stackgroup, 
                                   showlegend=False,line=dict(color=col[trace], dash='dot',width=1))
                        )
                counts.append(count)
            counts=pd.DataFrame(data=counts,index=df.columns, columns=index[:-1])

        elif mode=='scatter':
            for trace in list(df.columns):
                    if colour_set:
                        colour=col[trace]
                    else:
                        colour=None
                    if dropna:
                        data=df[trace].dropna()
                    else:
                        data=df[trace]
                    fig.add_trace(
                        go.Scatter(x=data.index, y=data, name=trace.replace('_',' ') , mode=line_mode,line_shape=line_shape,stackgroup=stackgroup, marker=dict(color=colour,size=None))
                    )
                    
        fig.update_xaxes(title=xTitle)
        fig.update_yaxes(title=yTitle)
        #fig.show()
        #return fig
        fig.update_layout(template=template,title=title,legend=dict(y=legend_shift) ,font=dict(color='black'),bargap=bargap,autosize=True)

        fig.update_xaxes(range=xlim)
        fig.update_yaxes(range=ylim)
        
        if x_tick_interval:
            fig.update_layout(
                        xaxis = dict(
                            tickmode = 'linear',
                            tick0 = 0,
                            dtick = x_tick_interval
                        )
                    )
        if y_tick_interval:
            fig.update_layout(
                        yaxis = dict(
                            tickmode = 'linear',
                            tick0 = 0,
                            dtick = y_tick_interval
                        )
                    )
        if VEPC_image:
            fig.add_layout_image(
                dict(
                    source="https://static.wixstatic.com/media/cb01c4_8b6a2bf455364d6980434eb57d584a2e~mv2_d_3213_1275_s_2.jpg",
                    xref="paper", yref="paper",
                    x=1, y=1,
                    sizex=0.18, sizey=0.18,opacity=0.75,
                    xanchor="right", yanchor="top"
                )
            )
        if horizontal_legend==True:
            fig.update_layout(legend_orientation="h")
        elif horizontal_legend=='right':
            fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ))
        #if set_black_lines==True:
        #    fig.update_xaxes(showline=True, linecolor='black',linewidth=1) 
        #    fig.update_yaxes(showline=True,  linecolor='black',linewidth=1,tickcolor='black')
        if mode=='line_hist':
            return fig,counts
        else:
            return fig

def get_colours_for_labels_n(labels=['<$0','$0-20', '$20-40', '$40-60','$60-80','$80-100','$100-150','$150-300','$300-1000','$1000-5000','>$5000'],colour_set='nipy_spectral',notebook=False,fade=False,n=False,colour_shift=0,Fuel_colours=True,raw_colour=False,custom_colours=False):
    '''
    Returns a dictionary with unique colours for each lablel
    '''
    import matplotlib
    labels=list(labels)
    
    if colour_shift>0:
        [labels.insert(i, i) for i in range(colour_shift) ]
    if n:
        if len(labels)<n:
            labels.extend(list(range(n-len(labels))))
    print(labels)
    cmap = matplotlib.cm.get_cmap(colour_set,lut=len(labels))
    i=0
    colours={}
    if fade==False:
        fade=256
    for l in labels:
        print(cmap)
        if raw_colour:
            colours[l]=cmap(i)
        else:
            colours[l]='rgb('+str(round(cmap(i)[0]*256))+','+str(round(cmap(i)[1]*256))+','+str(round(cmap(i)[2]*256))+','+str(round(cmap(i)[3]*fade))+')'
        i=i+1
    if Fuel_colours:
        colours['Large Scale Solar']='gold'
        colours['Rooftop Solar']='darkorange'
        colours['Wind']='mediumseagreen'
        colours['Black Coal']='black'
        colours['Brown Coal']='sienna'
        colours['Natural Gas']='grey'
        colours['Natural Gas (OCGT)']='DimGray'
        colours['Natural Gas (Steam)']='slategrey'
        colours['Natural Gas (CCGT)']='Silver'
        colours['Hydro']='SteelBlue'
        colours['Other Fuels']='orange'
        colours['Water']=colours['Hydro']
        colours['Battery']='orange'
        colours['Coal Seam Methane']='red'
        
        ## ARUP Colours:
        if custom_colours:
            for c in custom_colours.keys():

                colours[c]=custom_colours[c]


        

    return colours

