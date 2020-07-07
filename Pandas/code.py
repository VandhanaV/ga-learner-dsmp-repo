# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank=pd.read_csv(path)
categorical_var=bank.select_dtypes(include = 'object')
print(categorical_var)

numerical_var=bank.select_dtypes(include = 'number')
print(numerical_var)

# code ends here


# --------------
# code starts here
banks=bank.drop(columns='Loan_ID')
print(banks.isnull().sum())
bank_mode=banks.mode()

for x in banks.columns.values:
    banks[x]=banks[x].fillna(value=bank_mode[x].iloc[0])

#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks, values='LoanAmount', index=['Gender', 'Married', 'Self_Employed'],aggfunc=np.mean, fill_value=0)
print(avg_loan_amount)

# code ends here



# --------------
# code starts here
loan_approved_se=len(banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')])
loan_approved_nse=len(banks[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')])
Loan_Status=614
percentage_se=(loan_approved_se/Loan_Status)*100
percentage_nse=(loan_approved_nse/Loan_Status)*100
print("Percentage of loan approval for self employed people ",percentage_se)
print("Percentage of loan approval for people who are not self-employed",percentage_nse)
# code ends here


# --------------
# code starts here
loan_term=banks['Loan_Amount_Term'].apply(lambda x:int(x/12))
big_loan_term=0
for i in loan_term:
    if i>=25:
        big_loan_term=big_loan_term+1
print(big_loan_term)
# code ends here


# --------------
# code starts here
loan_groupby=banks.groupby(['Loan_Status'])[['ApplicantIncome', 'Credit_History']]
mean_values=loan_groupby.agg(np.mean)
print(mean_values)

# code ends here


