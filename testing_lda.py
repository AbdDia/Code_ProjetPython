# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:54:26 2021

@author: AK
"""

import pandas as pd
from LDA import LDA

help(LDA.stepdisc) # pour afficher la documentation du module

# test avec bdd normale
DT = pd.read_excel("Data_Illustration_Livre_ADL.xlsx",
                       sheet_name="DATA_2_TRAIN", engine="openpyxl")
X = DT[DT.columns[1:]]
y = DT.TYPE

lda=LDA()
print(dir(lda))

intr, cf = lda.fit(X=X,y=y)

lda.stepdisc(DT, "TYPE", 0.01, METHOD="forward", BIGDATA = True, 
           CONSOLELOG=False, HTMLFILE="fwd_test_with_graph.html")
lda.stepdisc(DT, "TYPE", 0.01, METHOD="backward", 
           CONSOLELOG=False,HTMLFILE="bwd_test.html")

lda.create_HTML()

lda.create_pdf()

# dataset_predict
DP = pd.read_excel("Data_Illustration_Livre_ADL.xlsx",sheet_name="DATA_2_TEST")
y_true = DP.TYPE
XP = DP[DP.columns[1:]]
y_pred = lda.predict(XP)
print('Confusion matrix: ', lda.confusion_matrix(y_true,y_pred))
lda.confusion_matrix_graph(y_true,y_pred)
print('Accuracy score: ', lda.accuracy_score(y_true, y_pred))