# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 22:06:09 2021

@author: AK
"""

import pandas as pd
from LDA import LDA

# test avec bdd normale
DT = pd.read_excel("Data_Illustration_Livre_ADL.xlsx",
                       sheet_name="HEART_STATLOG", engine="openpyxl")
X = DT[DT.columns[2:]]
y = DT[DT.columns[1]]

lda=LDA()

lda.variables_explicatives(X)