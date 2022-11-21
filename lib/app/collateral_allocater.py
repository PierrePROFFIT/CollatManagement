## Importing libraries
import streamlit as st
import pandas as pd
from deta import Deta
import numpy as np
from math import floor, ceil
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

def app():
        """
        Main function that is called when running the app
        """
        st.title("Collateral Manager")
        #Key to read/write in Deta base
        key="a0ujshlg_M37v7zCfNRvrPj9RiaD7ykTESJFdEbm4"
        #Establishing connection with Deta
        deta = Deta(key)

        st.subheader("--Upload portfolio--")
        date_selected = st.date_input("Date for which to perform Collateral Managment optimization")
        load_default_ptf = st.checkbox("Load Default Portfolio instead")
        if load_default_ptf:
                filename_input_portfolio="lib/data/Portfolio.xlsx"
        else:
                filename_input_portfolio = st.file_uploader("Upload Portfolio CSV or XLSX file", type=([".csv",".xlsx"]))
        if filename_input_portfolio:
                if load_default_ptf ==False:
                        #Load Portfolio (either csv file or xlsx)
                        if "csv" in filename_input_portfolio.name:
                                data_input_portfolio = pd.read_csv(filename_input_portfolio,sep=",").fillna("None")
                                st.write("Top rows of uploaded data")
                                st.write(data_input_portfolio.head())
                        elif "xlsx" in filename_input_portfolio.name:
                                data_input_portfolio = pd.read_excel(filename_input_portfolio,engine="openpyxl").fillna("None")
                                st.write("Top rows of uploaded data")
                                st.write(data_input_portfolio.head())
                else:
                        data_input_portfolio = pd.read_excel(filename_input_portfolio,engine="openpyxl").fillna("None")
                        st.write("Top rows of uploaded data")
                        st.write(data_input_portfolio.head())
        
                st.subheader("--Specify constraints for each Counterparty--")

                #Fetch latest constraints value
                deta_db_constraints = deta.Base("constraints_"+str(date_selected))
                deta_db_constraints_list = deta_db_constraints.fetch().items
                if len(deta_db_constraints_list)!=0:
                        deta_db_constraints_df = pd.DataFrame(deta_db_constraints_list)
                        st.write("Found agents already defined in database for the selected date")
                        st.write(deta_db_constraints_df.set_index("counterparty"))
                        deta_db_constraints_dict = deta_db_constraints_df.to_dict("index")
                else:
                        deta_db_constraints_dict = {int(k):v for k,v in json.load(open( "lib/data/default_constraints.json")).items()}

                with st.form(key='my_form'):
                        ud_constraints={}
                        nb_agents = 3
                        cols = st.columns(int(nb_agents))
                        for i, col in enumerate(cols):
                                with col:
                                        st.write("Counterparty "+str(i+1))
                                        ud_constraints[i]={}
                                        ud_constraints[i]["counterparty"]="Counterparty "+str(i+1)

                                        if i in deta_db_constraints_dict.keys():
                                                #Non eligible counterpaties
                                                ud_constraints[i]["non_eligible_type"] = st.multiselect(label='Type (to exclude)',key='non_eligible_type'+str(i),options=["Government Securities","Corporate Equity Securities","Corporate Debt Securities"],default=deta_db_constraints_dict[i]["non_eligible_type"])
                                                try:
                                                        ud_constraints[i]["non_eligible_issue_country"] = st.multiselect(label='Issue Country (to exclude)',key='non_eligible_issue_country'+str(i),options=data_input_portfolio["Issuer Country"].unique(),default=deta_db_constraints_dict[i]["non_eligible_issue_country"])
                                                except:
                                                        ud_constraints[i]["non_eligible_issue_country"] = st.multiselect(label='Issue Country (to exclude)',key='non_eligible_issue_country'+str(i),options=data_input_portfolio["Issuer Country"].unique())
                                                try:
                                                        ud_constraints[i]['non_eligible_issuer_sector'] = st.multiselect(label='Issuer Sector (to exclude)',key='non_eligible_issuer_sector'+str(i),options=data_input_portfolio["Issuer Sector"].unique(),default=deta_db_constraints_dict[i]["non_eligible_issuer_sector"])
                                                except:
                                                        ud_constraints[i]['non_eligible_issuer_sector'] = st.multiselect(label='Issuer Sector (to exclude)',key='non_eligible_issuer_sector'+str(i),options=data_input_portfolio["Issuer Sector"].unique())
                                                ud_constraints[i]["non_eligible_ratings_agency_2"] = st.multiselect(label='Ratings Agency 2 (to exclude)',key='non_eligible_ratings_agency_2'+str(i),options=['AAA', 'AA', 'A', 'BBB','BB', 'B', 'CCC', 'CC','C'],default=deta_db_constraints_dict[i]["non_eligible_ratings_agency_2"])
                                                ud_constraints[i]["non_eligible_adtv"] = st.selectbox(label='ADTV>3*average 3 month ADTV (to exclude)',key='non_eligible_adtv'+str(i),options=["Applicable","Not Applicable"],index=["Applicable","Not Applicable"].index(deta_db_constraints_dict[i]["non_eligible_adtv"]))
                                                #Concentration rule
                                                ud_constraints[i]["concentration_rule_issuer_country"] = st.number_input(label='Concentration Issuer Country (Max.)',key='concentration_rule_issuer_country'+str(i),format="%g",value=deta_db_constraints_dict[i]["concentration_rule_issuer_country"])
                                                ud_constraints[i]["concentration_rule_issuer_sector"] = st.number_input(label='Concentration Issuer Sector (Max.)',key='concentration_rule_issuer_sector'+str(i),format="%g",value=deta_db_constraints_dict[i]["concentration_rule_issuer_sector"])
                                                #Haircut rules
                                                ud_constraints[i]["haircut_rule_corpo_eq_AAA_BBB"] = st.number_input(label='Haircut Corpo Eq AAA to BBB',key='haircut_rule_corpo_eq_AAA_BBB'+str(i),format="%g",value=deta_db_constraints_dict[i]["haircut_rule_corpo_eq_AAA_BBB"])
                                                ud_constraints[i]["haircut_rule_corpo_eq_BB_CC"] = st.number_input(label='Haircut Corpo Eq BB to CC',key='haircut_rule_corpo_eq_BB_CC'+str(i),format="%g",value=deta_db_constraints_dict[i]["haircut_rule_corpo_eq_BB_CC"])
                                                ud_constraints[i]["haircut_rule_corpo_debt_AAA_BBB"] = st.number_input(label='Haircut Corpo Debt AAA to BBB',key='haircut_rule_corpo_debt_AAA_BBB'+str(i),format="%g",value=deta_db_constraints_dict[i]["haircut_rule_corpo_debt_AAA_BBB"])
                                                ud_constraints[i]["haircut_rule_corpo_debt_BB_CC"] = st.number_input(label='Haircut Corpo Debt BB to CC',key='haircut_rule_corpo_debt_BB_CC'+str(i),format="%g",value=deta_db_constraints_dict[i]["haircut_rule_corpo_debt_BB_CC"])
                                                ud_constraints[i]["haircut_rule_govies_AAA_BBB"] = st.number_input(label='Haircut Govies AAA to BBB',key='haircut_rule_govies_AAA_BBB'+str(i),format="%g",value=deta_db_constraints_dict[i]["haircut_rule_govies_AAA_BBB"])
                                                ud_constraints[i]["haircut_rule_govies_BB_CC"] = st.number_input(label='Haircut Govies BB to CC',key='haircut_rule_govies_BB_CC'+str(i),format="%g",value=deta_db_constraints_dict[i]["haircut_rule_govies_BB_CC"])
                                                ud_constraints[i]["haircut_rule_cross_currency"] = st.number_input(label='Haircut Cross Currency',key='haircut_rule_cross_currency'+str(i),format="%g",value=deta_db_constraints_dict[i]["haircut_rule_cross_currency"])
                                                #Objective exposure
                                                ud_constraints[i]["exposure"] = st.number_input(label='Exposure',key='exposure'+str(i),min_value=0,value=deta_db_constraints_dict[i]["exposure"])
                                        
                        submitted = st.form_submit_button('Submit')

                if submitted:
                        #Put/Save constraints in distant Deta database with name constraints_#date_selected#
                        put_df(ud_constraints,"constraints",date_selected,"counterparty",deta)
                        
                        st.subheader("--Output--")
                        #Perform optimization
                        optimization_ouput = perform_optimization(ud_constraints,data_input_portfolio)

                        #Put/Save portfolio in distant Deta database with name portfolio_#date_selected#
                        put_df(optimization_ouput.to_dict('index'),"portfolio",date_selected,"Code ISIN",deta)
                
                        st.subheader("--Output Visualization--")
                        #Display visualization
                        visualize_output(optimization_ouput)

def perform_optimization(constraint,df_portfolio):
        """
        Function that performs optimization given a set of constraints and a portfolio (pandas dataframe)
        """
        best_df = []
        all_permutations = [[3,2,1],[3,1,2],[1,2,3],[1,3,2],[2,1,3],[2,3,1]]
        for attribution_order_if_eq_value in all_permutations:
                #Load ratings mapping
                df_ratings = pd.read_excel("lib/data/ratings_project.xlsx")
                #Map ratings to numerical values
                df_portfolio["Ratings Agency 1 (Num)"] = df_portfolio["Ratings Agency 1"].map({'AAA':1, 'AA':2, 'A':3, 'BBB':4,'BB':5, 'B':6, 'CCC':7, 'CC':8,'C':9})
                df_portfolio["Ratings Agency 2 (Num)"] = df_portfolio["Ratings Agency 2"].map({'AAA':1, 'AA':2, 'A':3, 'BBB':4,'BB':5, 'B':6, 'CCC':7, 'CC':8,'C':9})
                #Compute for each row the maximum (ie worst rating) between numerical ratings of both agencies
                df_portfolio["Ratings Agency Worst of 2 (Num)"] = df_portfolio[["Ratings Agency 1 (Num)","Ratings Agency 2 (Num)"]].max(axis=1)
                df_portfolio["Ratings Agency Worst of 2"] = df_portfolio["Ratings Agency Worst of 2 (Num)"].map({1:'AAA', 2:'AA', 3:'A', 4:'BBB',5:'BB', 6:'B', 7:'CCC', 8:'CC',9:'C'})
                #Merge df_portfolio with df_ratings to add the Scarce Rating field
                df_portfolio_2 = pd.merge(df_portfolio, df_ratings,  how='left', left_on=['Type','Ratings Agency Worst of 2'], right_on = ['Type','Ratings Agency Worst of 2'])
                df_portfolio_2.rename(columns  = {'Scarce Rating': 'new_ratings'}, inplace = True)
                #Add exchnage rates to df_portfolio
                df_portfolio_2["Exchange Rate"] = df_portfolio_2.apply(get_exchange_rate,axis=1)
                #Convert prices to EUR
                df_portfolio_2['Price EUR'] = df_portfolio_2['Price'].astype(float)*df_portfolio_2["Exchange Rate"].astype(float)
                #Compute Market Capitalization (in EUR)
                df_portfolio_2['Market Capitalization EUR'] = df_portfolio_2['Price EUR'].astype(float)*df_portfolio_2['Quantity'].astype(float)
                #Check if ADTV>3*Average ADTV
                df_portfolio_2['ADTV_test'] = np.where(df_portfolio_2['ADTV'].astype(float)>3*df_portfolio_2['Average 3 Months ADTV'].astype(float),0,1)
                #Ensure that type of Quantity field is integer
                df_portfolio_2["Quantity"] = df_portfolio_2["Quantity"].astype(int)
                #Loop through constraint keys
                for j in constraint.keys():
                        #Create Eligible Counterparty field for each Counterparty that has value 1 if security satisfies eligibility conditions and 0 otherwise
                        df_portfolio_2["Eligible Counterparty "+str(j+1)] = df_portfolio_2.index.isin(apply_filter(df_portfolio_2,constraint[j])).astype(int)
                        #Create a Haircut field for each Counterparty that corresponds to a factor that integrates haircuts
                        df_portfolio_2['Haircut '+str(j+1)] = ((1-df_portfolio_2.apply(lambda x:apply_haircut(x,constraint[j]), axis=1))*(1-(df_portfolio_2['Currency']!='EUR')*constraint[j]["haircut_rule_cross_currency"])).round(4)
                        #Create a Value field for each counterparty that corresponds to the haircuted price of the securities for the eligible securities
                        df_portfolio_2['Value '+str(j+1)] = df_portfolio_2['Haircut '+str(j+1)]*df_portfolio_2["Eligible Counterparty "+str(j+1)]*df_portfolio_2["Market Capitalization EUR"]
                
                ##Start optimization

                #Sort portfolio dataframe by new_ratings values (decreasing order)
                df_portfolio_2.sort_values(by="new_ratings",inplace=True,axis=0,ascending=False)
                #Crete columns corresponding to the quantity and counterparty-wise-value allocated to each counterparty for each securities in portfolio
                df_portfolio_2[["allocation "+str(i+1) for i in range(len(constraint.keys()))]] = 0
                df_portfolio_2[["allocation qtt"+str(i+1) for i in range(len(constraint.keys()))]] = 0
                #Loop through each rows of the portfolio (ie each securities), starting with those that have the worst new_ratings
                for ind,row in df_portfolio_2.iterrows():
                        #Create a list where eligible counterparty ids will be stored
                        eligible=[]
                        #Loop through constraint keys (ie through counterparties)
                        for j in constraint.keys():
                                #Checking if counterparty eligible
                                eligibility_filter = df_portfolio_2.loc[ind,"Eligible Counterparty "+str(j+1)] == 1
                                #Create filters (boolean variables) on exposure, concentration by sector and concentration by issuer
                                exposure_filter = df_portfolio_2["allocation "+str(j+1)].sum()<constraint[j]["exposure"]
                                concentration_rule_issuer_country_filter = df_portfolio_2[df_portfolio_2["Issuer Country"]==df_portfolio_2.loc[ind,"Issuer Country"]]["allocation "+str(j+1)].sum()<constraint[j]["concentration_rule_issuer_country"]*constraint[j]["exposure"]
                                if df_portfolio_2.loc[ind,"Issuer Sector"]!="None":
                                        concentration_rule_issuer_sector_filter = df_portfolio_2[df_portfolio_2["Issuer Sector"]==df_portfolio_2.loc[ind,"Issuer Sector"]]["allocation "+str(j+1)].sum()<constraint[j]["concentration_rule_issuer_sector"]*constraint[j]["exposure"]
                                else:
                                        #Govies have no Sector associated thus, sector filter is set to True for those securitites
                                        concentration_rule_issuer_sector_filter = True
                                if eligibility_filter & exposure_filter & concentration_rule_issuer_country_filter & concentration_rule_issuer_sector_filter:
                                        #Compute the maximum possible allocation on the security to match concentration & exposure constraints
                                        max_add_exposure = constraint[j]["exposure"]-df_portfolio_2["allocation "+str(j+1)].sum()
                                        max_add_issuer_country = constraint[j]["exposure"]*constraint[j]["concentration_rule_issuer_country"] - df_portfolio_2[df_portfolio_2["Issuer Country"]==df_portfolio_2.loc[ind,"Issuer Country"]]["allocation "+str(j+1)].sum()
                                        if df_portfolio_2.loc[ind,"Issuer Sector"]!="None":
                                                max_add_issuer_sector = constraint[j]["exposure"]*constraint[j]["concentration_rule_issuer_sector"] - df_portfolio_2[df_portfolio_2["Issuer Sector"]==df_portfolio_2.loc[ind,"Issuer Sector"]]["allocation "+str(j+1)].sum()
                                        else:
                                                #For securities that have no Issuer Sector, we set max_add_issuer_sector to max_add_issuer_country (as we take the min afterwards it will have no impact)
                                                max_add_issuer_sector=max_add_issuer_country
                                        #Compute the global maximum possible allocation on the security
                                        max_add_total = min(max_add_exposure,max_add_issuer_country,max_add_issuer_sector)
                                        #Compute the maximum possible allocation in terms of concentration rules
                                        max_add_concentration = min(max_add_issuer_country,max_add_issuer_sector)
                                        #Append eligible counterparty id and maximum possible allocations
                                        eligible.append((j+1,max_add_total,max_add_concentration))
                        
                        if len(eligible)>0:
                                #Create dict in which actual possible allocation and allocation qtt will be stored
                                possible_allocation = {}
                                #Loop through eligible counterparties
                                for k in range(len(eligible)):
                                        #Retrieve components previously append to eligible list
                                        max_add_concentration = eligible[k][2]
                                        max_add_total = eligible[k][1]
                                        j = eligible[k][0]
                                        #Create a sub-dictionnary for each eligible counterparties
                                        possible_allocation[j] = {}
                                        #If counterparty j values the security less than the maximum possible allocation in terms of exposure & concentration, the possible allocation is the whole quantity on the line
                                        if df_portfolio_2.loc[ind,"Value "+str(j)]<=max_add_total:
                                                possible_allocation[j]["allocation"] = df_portfolio_2.loc[ind,"Value "+str(j)]
                                                possible_allocation[j]["allocation qtt"] = int(df_portfolio_2.loc[ind,"Quantity"])
                                        #Else, the possible allocation qtt is the qtt that matches the maximum possible allocation
                                        else:
                                                qtt=df_portfolio_2.loc[ind,"Quantity"]
                                                price=df_portfolio_2.loc[ind,"Price EUR"]
                                                haircut=df_portfolio_2.loc[ind, 'Haircut '+str(j)]
                                                #If max_add_concentration==max_add_total then the most constraining constraint is one of the concentration rule thus we take the floor to stay below the constraint threshold
                                                if max_add_concentration==max_add_total:
                                                        possible_allocation[j]["allocation qtt"] = floor(max_add_total/(price*haircut))
                                                #Else the most constraining constraint is the one on exposure thus we take ceil value so that the constraint is satured (possibly slightly over-satured)
                                                else:
                                                        possible_allocation[j]["allocation qtt"] = ceil(max_add_total/(price*haircut))
                                                #We deduct allocation (ie value of the qtt allocated to counterparty j (it factors in the haircut))
                                                possible_allocation[j]["allocation"] = possible_allocation[j]["allocation qtt"]*price*haircut
                                
                                #Find correct order to apply when allocating
                                #The list keys end up containing the ordered ids of the counterparties for which the current security is available
                                row_values = row[['Value '+str(v) for v in attribution_order_if_eq_value]]
                                row_values.index=attribution_order_if_eq_value
                                attribution_order_if_non_eq_value = list(row_values.sort_values(ascending=False).index)
                                keys = [k for k in possible_allocation.keys()]
                                keys = [k for k in attribution_order_if_non_eq_value if k in keys]
                                
                                #Allocate the maximum possible allocation to the first counterparty in keys
                                df_portfolio_2.loc[ind,"allocation "+str(keys[0])] = possible_allocation[keys[0]]["allocation"]
                                df_portfolio_2.loc[ind,"allocation qtt"+str(keys[0])] = possible_allocation[keys[0]]["allocation qtt"]
                                #If we didn't allocate the full available qtt to the first counterparty in keys we allocate what's left to the second counterparty in keys (if there's one)
                                if (len(keys)>=2) & (df_portfolio_2.loc[ind,"allocation qtt"+str(keys[0])] != df_portfolio_2.loc[ind,"Quantity"]):
                                        df_portfolio_2.loc[ind,"allocation qtt"+str(keys[1])] = min(df_portfolio_2.loc[ind,"Quantity"]-df_portfolio_2.loc[ind,"allocation qtt"+str(keys[0])],possible_allocation[keys[1]]["allocation qtt"])
                                        df_portfolio_2.loc[ind,"allocation "+str(keys[1])] = df_portfolio_2.loc[ind,"allocation qtt"+str(keys[1])]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(keys[1])]
                                        #If we didn't allocate the full available qtt to the first and second counterparty in keys we allocate what's left to the third counterparty in keys (if there's one)
                                        if (len(keys)==3) & (df_portfolio_2.loc[ind,"allocation qtt"+str(keys[0])] + df_portfolio_2.loc[ind,"allocation qtt"+str(keys[1])] != df_portfolio_2.loc[ind,"Quantity"]):
                                                df_portfolio_2.loc[ind,"allocation qtt"+str(keys[2])] = min(df_portfolio_2.loc[ind,"Quantity"]-df_portfolio_2.loc[ind,"allocation qtt"+str(keys[0])]-df_portfolio_2.loc[ind,"allocation qtt"+str(keys[1])],possible_allocation[keys[2]]["allocation qtt"])
                                                df_portfolio_2.loc[ind,"allocation "+str(keys[2])] = df_portfolio_2.loc[ind,"allocation qtt"+str(keys[2])]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(keys[2])]

                #Compute the Quantity Kept in Book and its market value
                df_portfolio_2["Quantity Kept in Book"] = df_portfolio_2["Quantity"]-df_portfolio_2["allocation qtt1"]-df_portfolio_2["allocation qtt2"]-df_portfolio_2["allocation qtt3"]
                df_portfolio_2["Value Kept in Book"] = df_portfolio_2["Quantity Kept in Book"]*df_portfolio_2["Price EUR"]
                #Compute the average rating of the securities kept in book
                score_kept_in_book = (df_portfolio_2["Value Kept in Book"]*df_portfolio_2["new_ratings"]).sum()/df_portfolio_2["Value Kept in Book"].sum()
                #Construct a dataframe that contains amount of total allocation by counterparty and required exposure
                output_total_allocation = pd.DataFrame(df_portfolio_2[["allocation 1","allocation 2","allocation 3"]].sum(),columns=["Total Allocation"])
                output_total_allocation['Required Exposure'] = [float(constraint[j]["exposure"]) for j in constraint.keys()]
                
                #Store in best_... variables the results associated with the permutation that yields the best score_kept_in_book or, if none of the permutations end up matching the required exposures, the permutation that yields the best tot_exposure
                if len(best_df)==0:
                        best_permutation = attribution_order_if_eq_value
                        best_df=df_portfolio_2
                        best_df_score = score_kept_in_book
                        best_df_tot_exposure = output_total_allocation[['Required Exposure','Total Allocation']].min(axis=1).sum()/output_total_allocation['Required Exposure'].sum()
                else:
                        tot_exposure = output_total_allocation[['Required Exposure','Total Allocation']].min(axis=1).sum()/output_total_allocation['Required Exposure'].sum()
                        if tot_exposure > best_df_tot_exposure:
                                best_df_tot_exposure = tot_exposure
                                best_df=df_portfolio_2
                                best_df_score = score_kept_in_book
                                best_permutation = attribution_order_if_eq_value
                        elif tot_exposure==best_df_tot_exposure:
                                if score_kept_in_book<best_df_score:
                                        best_df_tot_exposure = tot_exposure
                                        best_df=df_portfolio_2
                                        best_df_score = score_kept_in_book
                                        best_permutation = attribution_order_if_eq_value

        st.write("Score : Average Rating of the securities kept in Book (weighted by Market Cap. EUR) "+str(best_df_score))
        st.write("Best allocation order when equal valuation "+str(best_permutation))
        output_total_allocation = pd.DataFrame(best_df[["allocation 1","allocation 2","allocation 3"]].sum(),columns=["Total Allocation"])
        output_total_allocation['Required Exposure'] = [float(constraint[j]["exposure"]) for j in constraint.keys()]
        
        st.write("Total Allocation VS Required Exposure")
        if not np.all(output_total_allocation['Required Exposure']<=output_total_allocation['Total Allocation']):
                st.write("Did not succeed in reaching required exposure")
        st.write(output_total_allocation)

        st.write("Augmented portfolio with Allocations")
        st.write(best_df)

        return best_df

def put_df(dict_data,name,date,key_field,deta):
        """
        Function that enables to write in remote Deta database
        """
        deta_db = deta.Base(name+"_"+str(date))
        for k in dict_data.keys():
                if key_field not in dict_data[k].keys():
                        st.write("Make sure there's a column "+str(key_field)+ " in your file as this column will be used as key")
                else:
                        deta_db.put(dict_data[k],dict_data[k][key_field])

def apply_filter(data,constraint):
        """
        Function that returns filtered set of index
        """
        filter_type = ~data["Type"].isin(constraint["non_eligible_type"])
        filter_issuer_country = ~data["Issuer Country"].isin(constraint["non_eligible_issue_country"])
        filter_issuer_sector = ~data["Issuer Sector"].isin(constraint["non_eligible_issuer_sector"]) 
        filter_ratings = ~data['Ratings Agency 2'].isin(constraint["non_eligible_ratings_agency_2"])
        if constraint["non_eligible_adtv"]=="Applicable":
                filter_adtv = ~(data["ADTV_test"] == 0)
                idx=data[filter_type & filter_issuer_country & filter_issuer_sector & filter_ratings & filter_adtv].index
        else:
                idx=data[filter_type & filter_issuer_country & filter_issuer_sector & filter_ratings].index
        return idx
                       
def apply_haircut(data,constraint):
        """
        Function that returns value of the haircut to apply depending on Type & Ratings Agency Worst of 2 (Num)
        """
        if data['Type'] == 'Corporate Equity Securities' and data['Ratings Agency Worst of 2 (Num)']<=4:
                return  constraint["haircut_rule_corpo_eq_AAA_BBB"] 
        elif data['Type'] == 'Corporate Equity Securities' and data['Ratings Agency Worst of 2 (Num)']<=8:
                return constraint["haircut_rule_corpo_eq_BB_CC"] 
        elif data['Type'] == 'Corporate Debt Securities' and data['Ratings Agency Worst of 2 (Num)']<=4:
                return constraint["haircut_rule_corpo_debt_AAA_BBB"] 
        elif data['Type'] == 'Corporate Debt Securities' and data['Ratings Agency Worst of 2 (Num)']<=8:
                return constraint["haircut_rule_corpo_debt_BB_CC"] 
        elif data['Type'] == 'Government Securities' and data['Ratings Agency Worst of 2 (Num)']<=4:
                return constraint["haircut_rule_govies_AAA_BBB"]  
        elif data['Type'] == 'Government Securities' and data['Ratings Agency Worst of 2 (Num)']<=8:
                return constraint["haircut_rule_govies_BB_CC"] 
        else:
                return 0

def visualize_output(data):
        """
        Function that displays charts
        """
        fig = make_subplots(rows=1, cols=3,specs=[[{"type": "pie"}, {"type": "pie"},{"type": "pie"}]],subplot_titles=("Counterparty 1", "Counterparty 2","Counterparty 3"))
        for i in [1,2,3]:
                data2 = data.groupby("Issuer Sector")[["allocation "+str(i)]].sum()
                fig.add_trace(go.Pie(labels=data2.index, values = data2["allocation "+str(i)]),row=1,col=i)
        fig.update_layout(title="Allocation by Issuer Sector")
        st.plotly_chart(fig)

        fig = make_subplots(rows=1, cols=3,specs=[[{"type": "pie"}, {"type": "pie"},{"type": "pie"}]],subplot_titles=("Counterparty 1", "Counterparty 2","Counterparty 3"))
        for i in [1,2,3]:
                data2 = data.groupby("Issuer Country")[["allocation "+str(i)]].sum()
                fig.add_trace(go.Pie(labels=data2.index, values = data2["allocation "+str(i)]),row=1,col=i)
        fig.update_layout(title="Allocation by Issuer Country")
        st.plotly_chart(fig)
        st.subheader("----------Go To : Retrieve, Visualize & Download for a wider Visualization & to Download output----------")

def get_exchange_rate(data):
        """
        Function that returns exchange rate against EUR for Currency field
        """
        if data["Currency"]=="EUR":
                return 1
        else:
                return yf.Ticker(data["Currency"]+'EUR=X').history()['Close'].iloc[-1]
