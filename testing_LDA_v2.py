# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 19:03:20 2020

@author: Valinquish
"""

import pandas as pd
#import stepdisc as s
from LDA_v2 import LinearDiscriminantAnalysis as LDA


# test avec bdd normale
DTrain = pd.read_excel("Data_Illustration_Livre_ADL.xlsx",
                       sheet_name="DATA_2_TRAIN", engine="openpyxl")

# création d'un objet LDA permettant de faire une analyse discriminante 
# linéaire avec les données et le nom de la variable catégorielle en entrée
lda = LDA(DTrain, "TYPE")

# PROC DISCRIM
lda.fit()
print(lda.intercept_, lda.coef_)
lda.discrim_html_output() # sortie html
#lda.discrim_pdf_output() # sortie pdf

# PROC STEPDISC
lda.wilks_decay()
lda.stepdisc(0.01, "forward")
lda.stepdisc(0.01, "backward")

# PREDICTION
DP = pd.read_excel("Data_Illustration_Livre_ADL.xlsx",sheet_name="DATA_2_TEST")
y_true = DP.TYPE
valeurs_a_predire = DP[DP.columns[1:]]
y_pred = lda.predict(valeurs_a_predire)
lda.confusion_matrix(y_true,y_pred)
lda.accuracy_score(y_true, y_pred)
print('Confusion matrix: ', lda.confusionMatrix)
print('Accuracy score: ', lda.accuracy)