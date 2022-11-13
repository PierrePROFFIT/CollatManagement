## Importing libraries
import streamlit as st
import pandas as pd
from deta import Deta
import numpy as np
from math import floor, ceil
import yfinance as yf

#streamlit run c:/Users/pierr/OneDrive/Documents/CollatManagement/main.py
# attention option qui disappear de base
def app():
        st.title("Collateral Manager")
        #Key to read/write in Deta base
        key="a0ujshlg_M37v7zCfNRvrPj9RiaD7ykTESJFdEbm4"
        #Establishing connection
        deta = Deta(key)

        st.subheader("Upload portfolio in distant database")
        date_selected = st.date_input("Date for which to perform Collateral Managment optimization")
        filename_input_portfolio = st.file_uploader("Upload Portfolio CSV or XLSX file", type=([".csv",".xlsx"]))
        if filename_input_portfolio:
                if "csv" in filename_input_portfolio.name:
                        data_input_portfolio = pd.read_csv(filename_input_portfolio,sep=",").fillna("None")
                        st.write("Top rows of uploaded data")
                        st.write(data_input_portfolio.head())
                elif "xlsx" in filename_input_portfolio.name:
                        data_input_portfolio = pd.read_excel(filename_input_portfolio,engine="openpyxl").fillna("None")
                        st.write("Top rows of uploaded data")
                        st.write(data_input_portfolio.head())

                put_df(data_input_portfolio.to_dict('index'),"portfolio",date_selected,"Code ISIN",deta)
                
                st.subheader("Specify constraints for each Counterparty")

                deta_db_constraints = deta.Base("constraints_"+str(date_selected))
                deta_db_constraints_list = deta_db_constraints.fetch().items
                if len(deta_db_constraints_list)!=0:
                        deta_db_constraints_df = pd.DataFrame(deta_db_constraints_list)
                        st.write("Found agents already defined in database for the selected date")
                        st.write(deta_db_constraints_df.set_index("counterparty"))
                        deta_db_constraints_dict = deta_db_constraints_df.to_dict("index")
                        
                        #nb_agents_default = deta_db_constraints_df.shape[0]
                else:
                        pass
                        #nb_agents_default = 3

                #nb_agents = st.number_input("Number of agents",min_value=1,value=nb_agents_default)

                with st.form(key='my_form'):
                        ud_constraints={}
                        nb_agents = 3
                        cols = st.columns(int(nb_agents))
                        
                        for i, col in enumerate(cols):
                                with col:
                                        st.write("Counterparty "+str(i+1))
                                        ud_constraints[i]={}
                                        ud_constraints[i]["counterparty"]="Counterparty "+str(i+1)

                                        if (len(deta_db_constraints_list)!=0) and (i in deta_db_constraints_dict.keys()):
                                                ud_constraints[i]["non_eligible_type"] = st.multiselect(label='Type (to exclude)',key='non_eligible_type'+str(i),options=["Government Securities","Corporate Equity Securities","Corporate Debt Securities"],default=deta_db_constraints_dict[i]["non_eligible_type"])
                                                ud_constraints[i]["non_eligible_issue_country"] = st.multiselect(label='Issue Country (to exclude)',key='non_eligible_issue_country'+str(i),options=data_input_portfolio["Issuer Country"].unique(),default=deta_db_constraints_dict[i]["non_eligible_issue_country"])
                                                ud_constraints[i]['non_eligible_issuer_sector'] = st.multiselect(label='Issuer Sector (to exclude)',key='non_eligible_issuer_sector'+str(i),options=data_input_portfolio["Issuer Sector"].unique(),default=deta_db_constraints_dict[i]["non_eligible_issuer_sector"])
                                                ud_constraints[i]["non_eligible_ratings_agency_2"] = st.multiselect(label='Ratings Agency 2 (to exclude)',key='non_eligible_ratings_agency_2'+str(i),options=['AAA', 'AA', 'A', 'BBB','BB', 'B', 'CCC', 'CC','C'],default=deta_db_constraints_dict[i]["non_eligible_ratings_agency_2"])
                                                ud_constraints[i]["non_eligible_adtv"] = st.selectbox(label='ADTV>3*average 3 month ADTV',key='non_eligible_adtv'+str(i),options=["Applicable","Not Applicable"],index=["Applicable","Not Applicable"].index(deta_db_constraints_dict[i]["non_eligible_adtv"]))
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
                                                #Objective
                                                ud_constraints[i]["exposure"] = st.number_input(label='Exposure',key='exposure'+str(i),min_value=0,value=deta_db_constraints_dict[i]["exposure"])
                                        else:
                                                ud_constraints[i]["non_eligible_type"] = st.multiselect(label='Type',key='non_eligible_type'+str(i),options=["Government Securities","Corporate Equity Securities","Corporate Debt Securities"])
                                                ud_constraints[i]["non_eligible_issue_country"] = st.multiselect(label='Issue Country',key='non_eligible_issue_country'+str(i),options=data_input_portfolio["Issuer Country"].unique())
                                                ud_constraints[i]['non_eligible_issuer_sector'] = st.multiselect(label='Issuer Sector',key='non_eligible_issuer_sector'+str(i),options=data_input_portfolio["Issuer Sector"].unique())
                                                ud_constraints[i]["non_eligible_ratings_agency_2"] = st.multiselect(label='Ratings Agency 2 allowed',key='non_eligible_ratings_agency_2'+str(i),options=['AAA', 'AA', 'A', 'BBB','BB', 'B', 'CCC', 'CC','C'])
                                                ud_constraints[i]["non_eligible_adtv"] = st.selectbox(label='ADTV>3*average 3 month ADTV',key='non_eligible_adtv'+str(i),options=["Applicable","Not Applicable"])
                                                #Concentration rule
                                                ud_constraints[i]["concentration_rule_issuer_country"] = st.number_input(label='Concentration Issuer Country (Max.)',key='concentration_rule_issuer_country'+str(i),format="%g",value=0.15)
                                                ud_constraints[i]["concentration_rule_issuer_sector"] = st.number_input(label='Concentration Issuer Sector (Max.)',key='concentration_rule_issuer_sector'+str(i),format="%g",value=0.15)
                                                #Haircut rules
                                                ud_constraints[i]["haircut_rule_corpo_eq_AAA_BBB"] = st.number_input(label='Haircut Corpo Eq AAA to BBB',key='haircut_rule_corpo_eq_AAA_BBB'+str(i),format="%g")
                                                ud_constraints[i]["haircut_rule_corpo_eq_BB_CC"] = st.number_input(label='Haircut Corpo Eq BB to CC',key='haircut_rule_corpo_eq_BB_CC'+str(i),format="%g")
                                                ud_constraints[i]["haircut_rule_corpo_debt_AAA_BBB"] = st.number_input(label='Haircut Corpo Debt AAA to BBB',key='haircut_rule_corpo_debt_AAA_BBB'+str(i),format="%g")
                                                ud_constraints[i]["haircut_rule_corpo_debt_BB_CC"] = st.number_input(label='Haircut Corpo Debt BB to CC',key='haircut_rule_corpo_debt_BB_CC'+str(i),format="%g")
                                                ud_constraints[i]["haircut_rule_govies_AAA_BBB"] = st.number_input(label='Haircut Govies AAA to BBB',key='haircut_rule_govies_AAA_BBB'+str(i),format="%g")
                                                ud_constraints[i]["haircut_rule_govies_BB_CC"] = st.number_input(label='Haircut Govies BB to CC',key='haircut_rule_govies_BB_CC'+str(i),format="%g")
                                                ud_constraints[i]["haircut_rule_cross_currency"] = st.number_input(label='Haircut Cross Currency',key='haircut_rule_cross_currency'+str(i),format="%g")
                                                
                                                #Objective
                                                ud_constraints[i]["exposure"] = st.number_input(label='Exposure',key='exposure'+str(i),min_value=0)

                        submitted = st.form_submit_button('Submit')
                if submitted:
                        put_df(ud_constraints,"constraints",date_selected,"counterparty",deta)
                        rsltt = perform_optimization(ud_constraints,data_input_portfolio)
                        st.download_button('Download CSV',convert_df(rsltt), "testtt.csv",'text/csv')


def put_df(dict_data,name,date,key_field,deta):
        deta_db = deta.Base(name+"_"+str(date))
        for k in dict_data.keys():
                if key_field not in dict_data[k].keys():
                        st.write("Make sure there's a column "+str(key_field)+ " in your file as this column will be used as key")
                else:
                        deta_db.put(dict_data[k],dict_data[k][key_field])
                        
def res_deta(db):
    res = db.fetch()
    all_items = res.items

    # fetch until last is 'None'
    while res.last:
        res = db.fetch(last=res.last)
        all_items += res.items
    return res

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def perform_optimization(constraint,df_portfolio):
        #st.write(constraint)
        df_ratings = pd.read_excel("lib/data/ratings_project.xlsx")
        df_portfolio_2 = pd.merge(df_portfolio, df_ratings,  how='left', left_on=['Type','Ratings Agency 2'], right_on = ['Type','Ratings Agency worst of 2'])
        df_portfolio_2.rename(columns  = {'Scarce Rating': 'new_ratings'}, inplace = True)
        data_ratings_2 =  pd.DataFrame({'grades': ['AAA', 'AA', 'A', 'BBB','BB', 'B', 'CCC', 'CC','C'], 'ranks': [1,2,3,4,5,6,7,8,9]})  
        df_portfolio_2 = pd.merge(df_portfolio_2, data_ratings_2,  how='left', left_on=['Ratings Agency 2'], right_on = ['grades'])
        df_portfolio_2["Exchange Rate"] = df_portfolio_2.apply(get_exchange_rate,axis=1)
        df_portfolio_2['Price EUR'] = df_portfolio_2['Price'].astype(float)*df_portfolio_2["Exchange Rate"].astype(float)
        df_portfolio_2['Market Capitalization'] = df_portfolio_2['Price EUR'].astype(float)*df_portfolio_2['Quantity'].astype(float)
        df_portfolio_2['ADTV_test'] = np.where(df_portfolio_2['ADTV'].astype(float)>3*df_portfolio_2['Average 3 Months ADTV'].astype(float),0,1)
        df_portfolio_2["Quantity"].astype(int)
        for j in constraint.keys():
                df_portfolio_2["Eligible Counterparty "+str(j+1)] = df_portfolio_2.index.isin(apply_filter(df_portfolio_2,constraint[j])).astype(int)
                df_portfolio_2['Haircut '+str(j+1)] = ((1-df_portfolio_2.apply(lambda x:apply_haircut(x,constraint[j]), axis=1))*(1-(df_portfolio_2['Currency']!='EUR')*constraint[j]["haircut_rule_cross_currency"])).round(4)
                df_portfolio_2['Value '+str(j+1)] = df_portfolio_2['Haircut '+str(j+1)]*df_portfolio_2["Eligible Counterparty "+str(j+1)]*df_portfolio_2["Market Capitalization"]
        df_portfolio_2.sort_values(by="new_ratings",inplace=True,axis=0,ascending=False)
        st.write(df_portfolio_2)

        df_portfolio_2[["allocation "+str(i+1) for i in range(len(constraint.keys()))]] = 0
        df_portfolio_2[["allocation qtt"+str(i+1) for i in range(len(constraint.keys()))]] = 0
        df_portfolio_2["el_list"] = 0
        df_portfolio_2["alloc poss"] =0
        for ind,row in df_portfolio_2.iterrows():
                st.write(ind)
                eligible=[]
                for j in constraint.keys():
                        #Checking if counterparty eligible
                        eligibility_filter = df_portfolio_2.loc[ind,"Eligible Counterparty "+str(j+1)] == 1
                        exposure_filter = df_portfolio_2["allocation "+str(j+1)].sum()<constraint[j]["exposure"]
                        concentration_rule_issuer_country_filter = df_portfolio_2[df_portfolio_2["Issuer Country"]==df_portfolio_2.loc[ind,"Issuer Country"]]["allocation "+str(j+1)].sum()<constraint[j]["concentration_rule_issuer_country"]*constraint[j]["exposure"]
                        concentration_rule_issuer_sector_filter = df_portfolio_2[df_portfolio_2["Issuer Sector"]==df_portfolio_2.loc[ind,"Issuer Sector"]]["allocation "+str(j+1)].sum()<constraint[j]["concentration_rule_issuer_sector"]*constraint[j]["exposure"]
                        st.write(exposure_filter,concentration_rule_issuer_country_filter,concentration_rule_issuer_sector_filter)
                        if eligibility_filter & exposure_filter & concentration_rule_issuer_country_filter & concentration_rule_issuer_sector_filter:
                                max_add_exposure = constraint[j]["exposure"]-df_portfolio_2["allocation "+str(j+1)].sum()
                                max_add_issuer_country = constraint[j]["exposure"]*constraint[j]["concentration_rule_issuer_country"] - df_portfolio_2[df_portfolio_2["Issuer Country"]==df_portfolio_2.loc[ind,"Issuer Country"]]["allocation "+str(j+1)].sum()
                                max_add_issuer_sector = constraint[j]["exposure"]*constraint[j]["concentration_rule_issuer_sector"] - df_portfolio_2[df_portfolio_2["Issuer Sector"]==df_portfolio_2.loc[ind,"Issuer Sector"]]["allocation "+str(j+1)].sum()
                                st.write(max_add_exposure,max_add_issuer_country,max_add_issuer_sector)
                                max_add_total = min(max_add_exposure,max_add_issuer_country,max_add_issuer_sector)
                                eligible.append((j+1,max_add_total))
                
                st.write(eligible)
                if len(eligible)>0:
                        possible_allocation = {}
                        for k in range(len(eligible)):
                                max_add_total = eligible[k][1]
                                j = eligible[k][0]
                                possible_allocation[j] = {}
                                if df_portfolio_2.loc[ind,"Value "+str(j)]<=max_add_total:
                                        possible_allocation[j]["allocation"] = df_portfolio_2.loc[ind,"Value "+str(j)]
                                        possible_allocation[j]["allocation qtt"] = int(df_portfolio_2.loc[ind,"Quantity"])
                                else:
                                        qtt=df_portfolio_2.loc[ind,"Quantity"]
                                        price=df_portfolio_2.loc[ind,"Price EUR"]
                                        haircut=df_portfolio_2.loc[ind, 'Haircut '+str(j)]
                                        st.write(qtt,price,haircut,max_add_total)
                                        possible_allocation[j]["allocation qtt"] = floor(max_add_total/(price*haircut))
                                        possible_allocation[j]["allocation"] = possible_allocation[j]["allocation qtt"]*price*haircut
                        st.write(possible_allocation)
                        if len(eligible)==1:
                                print(1)
                                for k in possible_allocation.keys():
                                        df_portfolio_2.loc[ind,"allocation "+str(k)] = possible_allocation[k]["allocation"]
                                        df_portfolio_2.loc[ind,"allocation qtt"+str(k)] = possible_allocation[k]["allocation qtt"]

                        if len(eligible)==2:
                                print(2)
                                keys = [k for k in possible_allocation.keys()]
                                values = [df_portfolio_2.loc[ind,"Value "+str(k)] for k in keys]
                                for combin in [[0,1],[1,0]]:
                                        high = keys[combin[0]]
                                        low = keys[combin[1]]
                                        if df_portfolio_2.loc[ind,"Value "+str(high)] > df_portfolio_2.loc[ind,"Value "+str(low)]:
                                                df_portfolio_2.loc[ind,"allocation "+str(high)] = possible_allocation[high]["allocation"]
                                                df_portfolio_2.loc[ind,"allocation qtt"+str(high)] = possible_allocation[high]["allocation qtt"]
                                                if df_portfolio_2.loc[ind,"allocation qtt"+str(high)] != df_portfolio_2.loc[ind,"Quantity"]:
                                                        df_portfolio_2.loc[ind,"allocation qtt"+str(low)] = min(df_portfolio_2.loc[ind,"Quantity"]-df_portfolio_2.loc[ind,"allocation qtt"+str(high)],possible_allocation[low]["allocation qtt"])
                                                        df_portfolio_2.loc[ind,"allocation "+str(low)] = df_portfolio_2.loc[ind,"allocation qtt"+str(low)]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(low)]
                                if df_portfolio_2.loc[ind,"Value "+str(keys[0])] == df_portfolio_2.loc[ind,"Value "+str(keys[1])]:
                                        df_portfolio_2.loc[ind,"allocation qtt"+str(keys[0])] = min(min(floor(df_portfolio_2.loc[ind,"Quantity"]/2), possible_allocation[keys[0]]["allocation qtt"]) + max(0,ceil(df_portfolio_2.loc[ind,"Quantity"]/2)- possible_allocation[keys[1]]["allocation qtt"]),possible_allocation[keys[0]]["allocation qtt"])
                                        df_portfolio_2.loc[ind,"allocation qtt"+str(keys[1])] = min(min(ceil(df_portfolio_2.loc[ind,"Quantity"]/2), possible_allocation[keys[1]]["allocation qtt"]) + max(0,floor(df_portfolio_2.loc[ind,"Quantity"]/2)-possible_allocation[keys[0]]["allocation qtt"]),possible_allocation[keys[1]]["allocation qtt"])
                                        df_portfolio_2.loc[ind,"allocation "+str(keys[0])]=df_portfolio_2.loc[ind,"allocation qtt"+str(keys[0])]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(keys[0])]
                                        df_portfolio_2.loc[ind,"allocation "+str(keys[1])]=df_portfolio_2.loc[ind,"allocation qtt"+str(keys[1])]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(keys[1])]

                        if len(eligible)==3:
                                keys = [k for k in possible_allocation.keys()]
                                values = [df_portfolio_2.loc[ind,"Value "+str(k)] for k in keys]
                                for combin in [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,1,0],[2,0,1]]:
                                        high = keys[combin[0]]
                                        mid = keys[combin[1]]
                                        low = keys[combin[2]]
                                        if (df_portfolio_2.loc[ind,"Value "+str(high)] > df_portfolio_2.loc[ind,"Value "+str(low)]) & (df_portfolio_2.loc[ind,"Value "+str(high)] > df_portfolio_2.loc[ind,"Value "+str(mid)]) & (df_portfolio_2.loc[ind,"Value "+str(mid)] > df_portfolio_2.loc[ind,"Value "+str(low)]):
                                                df_portfolio_2.loc[ind,"allocation "+str(high)] = possible_allocation[high]["allocation"]
                                                df_portfolio_2.loc[ind,"allocation qtt"+str(high)] = possible_allocation[high]["allocation qtt"]
                                                if df_portfolio_2.loc[ind,"allocation qtt"+str(high)] != df_portfolio_2.loc[ind,"Quantity"+str(high)]:
                                                        df_portfolio_2.loc[ind,"allocation qtt"+str(mid)] = min(df_portfolio_2.loc[ind,"Quantity"]-df_portfolio_2.loc[ind,"allocation qtt"+str(high)],possible_allocation[mid]["allocation qtt"])
                                                        df_portfolio_2.loc[ind,"allocation "+str(mid)] = df_portfolio_2.loc[ind,"allocation qtt"+str(mid)]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(mid)]
                                                        if df_portfolio_2.loc[ind,"allocation qtt"+str(high)] + df_portfolio_2.loc[ind,"allocation qtt"+str(mid)] != df_portfolio_2.loc[ind,"Quantity"+str(high)]:
                                                                df_portfolio_2.loc[ind,"allocation qtt"+str(low)] = min(df_portfolio_2.loc[ind,"Quantity"]-df_portfolio_2.loc[ind,"allocation qtt"+str(high)]-df_portfolio_2.loc[ind,"allocation qtt"+str(mid)],possible_allocation[low]["allocation qtt"])
                                                                df_portfolio_2.loc[ind,"allocation "+str(low)] = df_portfolio_2.loc[ind,"allocation qtt"+str(low)]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(low)]
                                        elif (df_portfolio_2.loc[ind,"Value "+str(high)] == df_portfolio_2.loc[ind,"Value "+str(mid)]) & (df_portfolio_2.loc[ind,"Value "+str(high)] > df_portfolio_2.loc[ind,"Value "+str(low)]):
                                                df_portfolio_2.loc[ind,"allocation qtt"+str(high)] = min(min(floor(df_portfolio_2.loc[ind,"Quantity"]/2), possible_allocation[high]["allocation qtt"]) + max(0,ceil(df_portfolio_2.loc[ind,"Quantity"]/2)- possible_allocation[mid]["allocation qtt"]),possible_allocation[high]["allocation qtt"])
                                                df_portfolio_2.loc[ind,"allocation qtt"+str(mid)] = min(min(ceil(df_portfolio_2.loc[ind,"Quantity"]/2), possible_allocation[mid]["allocation qtt"]) + max(0,floor(df_portfolio_2.loc[ind,"Quantity"]/2)-possible_allocation[high]["allocation qtt"]),possible_allocation[mid]["allocation qtt"])
                                                df_portfolio_2.loc[ind,"allocation "+str(high)]=df_portfolio_2.loc[ind,"allocation qtt"+str(high)]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(high)]
                                                df_portfolio_2.loc[ind,"allocation "+str(mid)]=df_portfolio_2.loc[ind,"allocation qtt"+str(mid)]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(mid)]

                                                if df_portfolio_2.loc[ind,"allocation qtt"+str(high)] + df_portfolio_2.loc[ind,"allocation qtt"+str(mid)] != df_portfolio_2.loc[ind,"Quantity"+str(high)]:
                                                        df_portfolio_2.loc[ind,"allocation qtt"+str(low)] = min(df_portfolio_2.loc[ind,"Quantity"]-df_portfolio_2.loc[ind,"allocation qtt"+str(high)]-df_portfolio_2.loc[ind,"allocation qtt"+str(mid)],possible_allocation[low]["allocation qtt"])
                                                        df_portfolio_2.loc[ind,"allocation "+str(low)] = df_portfolio_2.loc[ind,"allocation qtt"+str(low)]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(low)]
                                
                                if df_portfolio_2.loc[ind,"Value "+str(keys[0])] == df_portfolio_2.loc[ind,"Value "+str(keys[1])] == df_portfolio_2.loc[ind,"Value "+str(keys[2])]:
                                        df_portfolio_2.loc[ind,"allocation qtt"+str(keys[0])] = min(min(floor(df_portfolio_2.loc[ind,"Quantity"]/3), possible_allocation[keys[0]]["allocation qtt"]) + max(0,ceil(df_portfolio_2.loc[ind,"Quantity"]/3)- possible_allocation[keys[1]]["allocation qtt"])+ max(0,floor(df_portfolio_2.loc[ind,"Quantity"]/3)- possible_allocation[keys[2]]["allocation qtt"]),possible_allocation[keys[0]]["allocation qtt"])
                                        df_portfolio_2.loc[ind,"allocation qtt"+str(keys[1])] = min(min(ceil(df_portfolio_2.loc[ind,"Quantity"]/3), possible_allocation[keys[1]]["allocation qtt"]) + max(0,floor(df_portfolio_2.loc[ind,"Quantity"]/3)-possible_allocation[keys[0]]["allocation qtt"])+ max(0,floor(df_portfolio_2.loc[ind,"Quantity"]/3)- possible_allocation[keys[2]]["allocation qtt"]),possible_allocation[keys[1]]["allocation qtt"])
                                        df_portfolio_2.loc[ind,"allocation qtt"+str(keys[2])] = min(min(floor(df_portfolio_2.loc[ind,"Quantity"]/3), possible_allocation[keys[2]]["allocation qtt"]) + max(0,floor(df_portfolio_2.loc[ind,"Quantity"]/3)- possible_allocation[keys[0]]["allocation qtt"])+ max(0,ceil(df_portfolio_2.loc[ind,"Quantity"]/3)- possible_allocation[keys[1]]["allocation qtt"]),possible_allocation[keys[2]]["allocation qtt"])
                                        df_portfolio_2.loc[ind,"allocation "+str(keys[0])]=df_portfolio_2.loc[ind,"allocation qtt"+str(keys[0])]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(keys[0])]
                                        df_portfolio_2.loc[ind,"allocation "+str(keys[1])]=df_portfolio_2.loc[ind,"allocation qtt"+str(keys[1])]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(keys[1])]
                                        df_portfolio_2.loc[ind,"allocation "+str(keys[2])]=df_portfolio_2.loc[ind,"allocation qtt"+str(keys[2])]*df_portfolio_2.loc[ind,"Price EUR"]*df_portfolio_2.loc[ind, 'Haircut '+str(keys[2])]

                                        
                                        


        #filter portfolio : eligible_counterparty_i, sort by score
        # go through each line, check that respect conditions

        return df_portfolio_2

def apply_filter(data,constraint):
        filter_type = ~data["Type"].isin(constraint["non_eligible_type"])
        filter_issuer_country = ~data["Issuer Country"].isin(constraint["non_eligible_issue_country"])
        filter_issuer_sector = ~data["Issuer Sector"].isin(constraint["non_eligible_issuer_sector"]) 
        filter_ratings = ~data['Ratings Agency 2'].isin(constraint["non_eligible_ratings_agency_2"])
        filter_adtv = ~(data["ADTV_test"] == 0)
        #df=data[data["Type"].isin(constraint["non_eligible_type"]) | data["Issuer Country"].isin(constraint["non_eligible_issue_country"]) | data["Issuer Sector"].isin(constraint["non_eligible_issuer_sector"]) | data['new_ratings'] > constraint["non_eligible_ratings_agency_2"] | data["ADTV_test"] == 0]
        #st.write(data[filter_type & filter_issuer_country & filter_issuer_sector & filter_ratings & filter_adtv])
        return data[filter_type & filter_issuer_country & filter_issuer_sector & filter_ratings & filter_adtv].index
                       
def apply_haircut(data,constraint):
  if data['Type'] == 'Corporate Equity Securities' and data['ranks']<4:
    return  constraint["haircut_rule_corpo_eq_AAA_BBB"] 
  elif data['Type'] == 'Corporate Equity Securities' and data['ranks']<8:
    return constraint["haircut_rule_corpo_eq_BB_CC"] 
  elif data['Type'] == 'Corporate Debt Securities' and data['ranks']<4:
    return constraint["haircut_rule_corpo_debt_AAA_BBB"] 
  elif data['Type'] == 'Corporate Debt Securities' and data['ranks']<8:
    return constraint["haircut_rule_corpo_debt_BB_CC"] 
  elif data['Type'] == 'Government Securities' and data['ranks']<4:
    return constraint["haircut_rule_govies_AAA_BBB"]  
  elif data['Type'] == 'Government Securities' and data['ranks']<8:
    return constraint["haircut_rule_govies_BB_CC"] 
  else:
    return 0

def get_exchange_rate(data):
        if data["Currency"]=="EUR":
                return 1
        else:
                return yf.Ticker(data["Currency"]+'EUR=X').history()['Close'].iloc[-1]

st.set_page_config(layout="wide")
pages = {
        "Import Portfolio in Cloud Database":app()
        }
st.sidebar.title("Collateral Manager")
select_page = st.sidebar.radio("GO TO : ", list(pages.keys()))
pages[select_page]