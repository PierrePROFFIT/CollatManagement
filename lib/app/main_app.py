path='C:/Users/kunas/OneDrive - Université Paris-Dauphine/Magistère BFA/BFA 3/01_Cours/01_Python/Project/'
df_portfolio = pd.read_excel (path+'Portfolio.xlsx')
df_portfolio_2=df_portfolio.drop(df_portfolio.columns[0], axis=1).dropna( how='all',subset=['Code ISIN'])

df_ratings = pd.read_excel (path+'ratings_project.xlsx')
df_constraints = pd.read_excel (path+'constraints_project.xlsx')

#**************Defining constraints following the counterparts*******************

cols_vides = [col for col in df_constraints.columns if df_constraints[col].isnull().all()]
df_constraints.drop(cols_vides, axis=1, inplace=True)
df_constraints.columns = ['type_rules','type_constraints','ctp_A','ctp_B','ctp_C']
df_constraints['type_rules'] =  df_constraints['type_rules'].fillna(method='ffill')
df_constraints=df_constraints.drop(df_constraints.columns[0], axis=1).dropna( how='all',subset=['type_constraints'])

constraint_counterparty_A = ["None","Japan",["Financials", "Oil & Gas"],4,
                             "ATDV > 3*average 3 month ADTV","Max 15%","Max 25%",
                             0.05,0,0.02,0,0.02,0,0.05,300000000]

constraint_counterparty_B = ["Corporate Equity Securities","Italy","Financials",
                             6,"ATDV > 3*average 3 month ADTV","Max 20%",
                             "Max 20%",0.05,0.08,0.02,0.05,0.02,0.05,0.05,300000000]

constraint_counterparty_C = ["Corporate Debt Securities","Hong-Kong","Oil & Gas",
                             7,"ATDV > 3*average 3 month ADTV","Max 25%",
                             "Max 15%",0.05,0.08,0.02,0.05,0.02,0.05,0.05,300000000]

#*************Adding rates***************
df_portfolio_2 = pd.merge(df_portfolio_2, df_ratings,  how='left', left_on=['Type','Ratings Agency 2'], right_on = ['Type','Ratings Agency worst of 2'])
df_portfolio_2.rename(columns  = {'Scarce Rating': 'new_ratings'}, inplace = True)
data_ratings_2 =  pd.DataFrame({'grades': ['AAA', 'AA', 'A', 'BBB','BB', 'B', 'CCC', 'CC','C'], 'ranks': [1,2,3,4,5,6,7,8,9]})  
df_portfolio_2 = pd.merge(df_portfolio_2, data_ratings_2,  how='left', left_on=['Ratings Agency 2'], right_on = ['grades'])


df_portfolio_2['Market Capitalization'] = df_portfolio_2['Price']*df_portfolio_2['Quantity']
df_portfolio_2['ADTV_test'] = np.where(df_portfolio_2['ADTV']>3*df_portfolio_2['Average 3 Months ADTV'],0,1)

print(df_portfolio_2)

 #***************Checking the constraints for the different counterparts and applying them to the portfolio************

#Mode bourrin

def check_constraint_A(data):
  if data['Type'] == constraint_counterparty_A[0] or data[' Issuer Sector'] in constraint_counterparty_A[2] or data['Issuer Country'] == constraint_counterparty_A[1] or \
     data['new_ratings'] > constraint_counterparty_A[3] or data['ADTV_test'] == 0:
    return 0
  else:
    return 1

def check_constraint_B(data):
  if data['Type'] == constraint_counterparty_B[0] or data[' Issuer Sector'] == constraint_counterparty_B[2] or data['Issuer Country'] == constraint_counterparty_B[1] or \
     data['new_ratings'] > constraint_counterparty_B[3] or data['ADTV_test'] == 0:
    return 0
  else:
    return 1

def check_constraint_C(data):
  if data['Type'] == constraint_counterparty_C[0] or data[' Issuer Sector'] == constraint_counterparty_C[2] or data['Issuer Country'] == constraint_counterparty_C[1] or \
     data['new_ratings'] > constraint_counterparty_C[3] or data['ADTV_test'] == 0:
    return 0
  else:
    return 1


df_portfolio_2['eligible_A'] = df_portfolio_2.apply(check_constraint_A, axis=1)
df_portfolio_2['eligible_B'] = df_portfolio_2.apply(check_constraint_B, axis=1)
df_portfolio_2['eligible_C'] = df_portfolio_2.apply(check_constraint_C, axis=1)
print(df_portfolio_2)


#Value following haircuts

def haircuts_A(data):
  if data['Type'] == 'Corporate Equity Securities' and data['ranks']<4:
    return  constraint_counterparty_A[7] 
  elif data['Type'] == 'Corporate Equity Securities' and data['ranks']<8:
    return constraint_counterparty_A[8] 
  elif data['Type'] == 'Corporate Debt Securities' and data['ranks']<4:
    return constraint_counterparty_A[9] 
  elif data['Type'] == 'Corporate Debt Securities' and data['ranks']<8:
    return constraint_counterparty_A[10] 
  elif data['Type'] == 'Government Securities' and data['ranks']<4:
    return constraint_counterparty_A[11] 
  elif data['Type'] == 'Government Securities' and data['ranks']<8:
    return constraint_counterparty_A[12] 
  else:
    return 0

def haircuts_B(data):
  if data['Type'] == 'Corporate Equity Securities' and data['ranks']<4:
    return  constraint_counterparty_B[7] 
  elif data['Type'] == 'Corporate Equity Securities' and data['ranks']<8:
    return constraint_counterparty_B[8] 
  elif data['Type'] == 'Corporate Debt Securities' and data['ranks']<4:
    return constraint_counterparty_B[9] 
  elif data['Type'] == 'Corporate Debt Securities' and data['ranks']<8:
    return constraint_counterparty_B[10] 
  elif data['Type'] == 'Government Securities' and data['ranks']<4:
    return constraint_counterparty_B[11] 
  elif data['Type'] == 'Government Securities' and data['ranks']<8:
    return constraint_counterparty_B[12] 
  else:
    return 0

def haircuts_C(data):
  if data['Type'] == 'Corporate Equity Securities' and data['ranks']<4:
    return  constraint_counterparty_C[7] 
  elif data['Type'] == 'Corporate Equity Securities' and data['ranks']<8:
    return constraint_counterparty_C[8] 
  elif data['Type'] == 'Corporate Debt Securities' and data['ranks']<4:
    return constraint_counterparty_C[9] 
  elif data['Type'] == 'Corporate Debt Securities' and data['ranks']<8:
    return constraint_counterparty_C[10] 
  elif data['Type'] == 'Government Securities' and data['ranks']<4:
    return constraint_counterparty_C[11] 
  elif data['Type'] == 'Government Securities' and data['ranks']<8:
    return constraint_counterparty_C[12] 
  else:
    return 0

df_portfolio_2['value_A'] = (1-df_portfolio_2.apply(haircuts_A, axis=1))*(1-(df_portfolio_2['Currency']!='EUR')*0.05)
df_portfolio_2['value_B'] = (1-df_portfolio_2.apply(haircuts_B, axis=1))*(1-(df_portfolio_2['Currency']!='EUR')*0.05)
df_portfolio_2['value_C'] = (1-df_portfolio_2.apply(haircuts_C, axis=1))*(1-(df_portfolio_2['Currency']!='EUR')*0.05)

print(df_portfolio_2)

saving_path = "C:/Users/kunas/OneDrive - Université Paris-Dauphine/Magistère BFA/BFA 3/01_Cours/01_Python/Project/" 
file_name = 'Output_Excel.xlsx'
df_portfolio_2.to_excel(saving_path + file_name, index = True)


#Mode moins bourrin

# def check_constraint(data,constraints_list):
#   if data[' Issuer Sector'] == constraints_list[2] :
# #   if data['Type'] == constraints_list[0] or data[' Issuer Sector'] == constraints_list[2] or data['Issuer Country'] == constraints_list[1] or \
# #      data['new_ratings'] > constraints_list[3] or data['ADTV_test'] == 0:
#     return 0
#   else:
#     return 1

# df_portfolio_2['eligible_A'] = df_portfolio_2.apply(lambda x: check_constraint(df_portfolio_2,constraint_counterparty_A), axis=1)
# df_portfolio_2['eligible_B'] = df_portfolio_2.apply(lambda x: check_constraint(df_portfolio_2,constraint_counterparty_B), axis=1)
# df_portfolio_2['eligible_C'] = df_portfolio_2.apply(lambda x: check_constraint(df_portfolio_2,constraint_counterparty_C), axis=1)

# def check_constraint(data):
#   if data['Type'] == constraint_counterparty_A[0] or data[' Issuer Sector'] in constraint_counterparty_A[2] or data['Issuer Country'] == constraint_counterparty_A[1] or \
#      data['new_ratings'] > constraint_counterparty_A[3] or data['ADTV_test'] == 0:
# #   if data['Type'] == constraints_list[0] or data[' Issuer Sector'] == constraints_list[2] or data['Issuer Country'] == constraints_list[1] or \
# #      data['new_ratings'] > constraints_list[3] or data['ADTV_test'] == 0:
#     return 0
#   else:
#     return 1

# df_portfolio_2['eligible_A'] = df_portfolio_2.apply(check_constraint, axis=1)
# print(df_portfolio_2)
