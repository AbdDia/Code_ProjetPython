# -*- coding: utf-8 -*-
"""
Démonstration de l'application d'Analyse Discriminante Linéaire
"""

import pandas as pd
from discriminant_analysis import LinearDiscriminantAnalysis as LDA
from reporting import HTML, PDF

# test avec bdd normale
DTrain = pd.read_excel("data\Data_Illustration_Livre_ADL.xlsx",
                       sheet_name="DATA_2_TRAIN", engine="openpyxl")

# création d'un objet LDA permettant de faire une analyse discriminante 
# linéaire avec les données et le nom de la variable catégorielle en entrée
lda = LDA(DTrain, "TYPE")

# PROC DISCRIM
lda.fit()
print(lda.intercept_, lda.coef_)
# REPORTING
HTML().discrim_html_output(lda, "output\discrim_results.html") # sortie html
PDF().discrim_pdf_output(lda, "output\discrim_results.pdf") # sortie pdf

# PROC STEPDISC
lda.wilks_decay()
lda.stepdisc(0.01, "forward")
HTML().stepdisc_html_output(lda, "output\stepdisc_forward_results.html")
lda.stepdisc(0.01, "backward")
HTML().stepdisc_html_output(lda, "output\stepdisc_backward_results.html")

# PREDICTION
DP = pd.read_excel("data\Data_Illustration_Livre_ADL.xlsx",
                   sheet_name="DATA_2_TEST")
y_true = DP.TYPE
valeurs_a_predire = DP[DP.columns[1:]]
y_pred = lda.predict(valeurs_a_predire)
lda.confusion_matrix(y_true, y_pred)
lda.accuracy_score(y_true, y_pred)
print('Confusion matrix: ', lda.confusionMatrix)
print('Accuracy score: ', lda.accuracy)