# -*- coding: utf-8 -*-
"""
Application créée dans le cadre d'un projet de L3 IDS à l'Université Lyon 2.

Groupe constitué de Mamadou DIALLO, Aymeric DELEFOSSE et Aleksandra
KRUCHININA.
"""

#---- data analysis librairies
import pandas as pd
import numpy as np
import scipy.stats as stats
#---- data visualization librairies
import matplotlib.pyplot as plt
import seaborn as sns
#---- data restitution librairies
from fpdf import FPDF
import datapane as dp


class PDF(FPDF):
    """Facilite l'affichage du bas de page automatique dès la création d'une 
    instance PDF.
    Ref. : https://pyfpdf.readthedocs.io/en/latest/Tutorial/index.html
    """
    # Page footer

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Text color in gray
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')


def verification_NA(inputData, targetValues):
    """Vérification pour les valeurs nulles.
    Les observations avec des valeurs nulles ne sont pas prises
    en compte pour l'analyse.
    La fonction affiche le nombre des observation supprimées.

    Paramètres
    ----------
    inputData : array-like of shape (n_samples, n_features)
        Input data.
    targetValues : array-like of shape (n_samples,) or (n_samples, n_targets) 
        Valeurs cibles.

    Sortie
    -------
    inputDataWithoutNA : 
        Input data sans NA.
    targetValuesWithoutNA : 
        Valeurs cibles sans NA.

    """
    n, p = inputData.shape
    # il faut concatener les inputs pour supprimer les lignes
    # avec des valeurs nulles
    df = pd.concat((inputData, targetValues), axis=1)
    df.dropna(axis=0, inplace=True)
    n_del = n - df.shape[0]
    inputDataWithoutNA = df.iloc[:, :-1]
    targetValuesWithoutNA = df.iloc[:, -1]
    print('Attention : ', n_del, ' observations ont été supprimées.')
    return inputDataWithoutNA, targetValuesWithoutNA


def freq_relat(targetValues, n):
    """Calcul des fréquences relatives.

    Paramètres
    ----------
    targetValues : array-like of shape (n_samples,) or (n_samples, n_targets) 
        Target values.

    Sortie
    -------
    freqClassValues : numeric
        fréquences relatives par classe
    effClassValues : numeric
        nombre d'occurences par classe (effectifs)
    """
    # Nb nombre d'effectifs par classes
    effClassValues = np.unique(targetValues, return_counts=True)[1]
    freqClassValues = effClassValues / n
    return effClassValues, freqClassValues


def means_class(inputData, targetValues):
    """Calcul des moyennes conditionnelles selon le groupe d'appartenance.

    Paramètres
    ----------
    inputData : array-like of shape (n_samples, n_features)
                Input data.
    targetValues : array-like of shape (n_samples,) or (n_samples, n_targets) 
                   Target values.
        
    Sortie
    -------
    means_cl : moyennes conditionnelles par classe
    """
    # y1 convertion des valeurs cibles en numérique
    classNames, classValues = np.unique(targetValues, return_inverse=True)
    # Nk nombre d'effectifs par classes
    effClassValues = np.bincount(classValues)
    # initiation d'une matrice remplie de 0
    means = np.zeros(shape=(len(classNames), inputData.shape[1]))
    np.add.at(means, classValues, inputData)
    means_cls = means / effClassValues[:, None]
    return means_cls


def cov_matrix(dataset):
    """Calcul de la matrice de covariance totale (V) ainsi que sa version 
    biaisée (Vb).

    Paramètres
    ----------
    dataset : DataFrame (pandas)
        Jeu de données
        
    Notes
    ----------
    Les DataFrame (pandas) permettent de garder un résultat avec le nom des 
    variables.
    """
    n = dataset.shape[0]  # taille de l'échantillon
    V = dataset.cov()  # matrice de covariace totale
    Vb = (n-1)/n * V  # matrice de covariance totale biaisée
    return (V, Vb)


def pooled_cov_matrix(dataset, className):
    """Calcul de la matrice de covariance intra-classe (W) ainsi que sa 
    version biaisée (Wb).

    Paramètres
    ----------
    dataset : DataFrame (pandas)
        Jeu de données
    className : string
        Nom de la colonne contenant les différentes classes du jeu de données
    
    Notes
    ----------
    Les DatFrame (pandas) permettent de garder un résultat avec le nom des 
    variables.
    """
    n = dataset.shape[0]  # taille de l'échantillon
    K = len(dataset[className].unique())  # nombre de classes
    W = 0  # initialisation de W
    for modalities in dataset[className].unique():
        Vk = dataset.loc[dataset[className] == modalities].cov()
        W += (dataset[className].value_counts()[modalities] - 1) * Vk
    W *= 1/(n-K)  # matrice de covariance intra-classes
    Wb = (n-K)/n * W  # matrice de covariance intra-classes biaisée
    return (W, Wb)


def wilks(Vb, Wb):
    """Calcul du Lambda de Wilks par le rapport entre les déterminants des 
    estimateurs biaisés des matrices de variance covariance intra-classes et 
    totales.

    Paramètres
    ----------
    Vb : Matrice Numpy / Pandas
        Matrice de covariance totale
    Wb : Matrice Numpy / Pandas
        Matrice de covariance intra-classes
    """
    # les paramètres d'entrée doivent être des matrices numpy ou
    # des DataFrame (pandas)
    detVb = np.linalg.det(Vb)  # dét. de la matrice de cov. totale biaisée
    detWb = np.linalg.det(Wb)  # dét. de la matrice de cov.
    # intra-classes biaisée
    return (detWb / detVb)


def wilks_log(Vb, Wb):
    """Calcul du Lambda de Wilks par le rapport entre les logarithmes naturels
    des déterminants des estimateurs biaisés des matrices de variance 
    covariance intra-classes et totales.
    Permet la gestion de bases de données avec beaucoup de variables (> 90).

    Paramètres
    ----------
    Vb : Matrice Numpy / Pandas
        Matrice de covariance totale
    Wb : Matrice Numpy / Pandas
        Matrice de covariance intra-classes
    """
    detVb = np.linalg.slogdet(Vb)  # log. nat. du dét. de la matrice de cov.
    # totale biaisée
    detWb = np.linalg.slogdet(Wb)  # log. nat. du dét. de la matrice de cov.
    # intra-classes biaisée
    return np.exp((detWb[0]*detWb[1])-(detVb[0]*detVb[1]))


def p_value(F, ddl1, ddl2):
    """Calcul de la p-value d'un test unilatéral de Fisher.

    Paramètres
    ----------
    F : numeric
        statistique de Fisher
    ddl1, ddl2 : integer
        degrés de liberté (int)
    """
    if (F < 1): 
        return (1.0 - stats.f.cdf(1.0/F, ddl1, ddl2))
    
    return (1.0 - stats.f.cdf(F, ddl1, ddl2))


def variables_exlicatives(inputData):
        """La fonction variables_einputDataplicatives() permet de convertir des 
        variables einputDataplicatives en variables numériques. 
        Dans le cadre de notre projet LDA il suffit de lui donner en entrée 
        les variables (einputDataplicatives) à convertir et il retourne 
        variable un dataframe de variables numériques.
        """

        d = dict()  # Creation d'un dictionnaire vide
        # Apply permet ici de faire une boucle comme avec R.
        # Test puis conversion de la variable en numérique si elle ne l'est pas
        d = inputData.apply(lambda s: pd.to_numeric(
            s, errors='coerce').notnull().all())
        # Renvoie False si la variable n'est pas numérique, True sinon.
        liste = d.values
        for i in range(len(inputData.columns)):
            # Conversion de toutes les variables qui ne sont pas numériques 
            # en objet.
            if liste[i] == False:
                # Conversion des types "non-numeric" en "objet"
                inputData.iloc[:, i] = inputData.iloc[:, i].astype(object)

        # Recodage des colonnes (variables einputDataplicatives) grâce à la
        # fonction get_dummies de pandas
        for i in range(inputData.shape[1]):
            if inputData.iloc[:, i].dtype == object:
                dummy = pd.get_dummies(inputData.iloc[:, i], drop_first=True)
                for j in range(dummy.shape[1]):
                    # Concatenation (rajout des variables recodees à inputData) 
                    # pour chaque colonne de dummy, avec inputData le 
                    # dataframe de base
                    inputData = pd.concat(
                        [inputData, dummy.iloc[:, j]], ainputDatais=1)
        # Suppression des colonnes non numerics
        inputData = inputData._get_numeric_data()
        return inputData


def createWebFile(filename):
    with open(filename, "w") as f:
        f.write("""<!DOCTYPE html>
<html lang="fr" dir="ltr">
  <head>
    <title>R&#233;sultats : STEPDISC</title>
    <meta charset="utf-8" />
    <style></style>
    <link
      rel="stylesheet"
      href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
      integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z"
      crossorigin="anonymous"
    />
  </head>
  <body>
    <div class="container text-center">
        <h2>Proc&#233;dure STEPDISC</h2>""")
        f.close()


class LinearDiscriminantAnalysis:
    """Analyse Discriminante Linéaire Prédictive basée sur les méthodes 
    PROC DISCRIM et STEPDISC de SAS.
    """
    def __init__(self, dataset, classEtiquette, varNames=None):
        # jeu de données
        self.dataset = dataset
        # nom de la variable catégorielle
        self.classEtiquette = classEtiquette
        # noms des valeurs prises pour la variable catégorielle
        self.classNames = list(dataset[classEtiquette].unique()) 
        # nom des variables explicatives
        if varNames is None:
            varNames = list(dataset.columns)
            varNames.remove(classEtiquette)
            self.varNames = varNames
        else:
            self.varNames = list(varNames)
        self.n = dataset.shape[0]  # taille de l'échantillon
        self.p = len(varNames)  # nombre de variables explicatives
        self.K = len(dataset[classEtiquette].unique())  # nombre de classes 
        self.V, self.Vb = cov_matrix(dataset) # matrices de cov. totale
        # matrices de cov. intra-classes
        self.W, self.Wb = pooled_cov_matrix(dataset, classEtiquette) 
    
        
    def stats_dataset(self):
        """Informations de bases sur le jeu de données.
        """
        self.infoDataset = pd.DataFrame(
            [self.n, self.p, self.K, self.n - 1,self.n - self.K, self.K - 1],
            index=["Taille d'echantillon totale","Variables", "Classes",
                   "Total DDL","DDL dans les classes", "DDL entre les classes"],
            columns=["Valeur"])
        
        
    def stats_pooled_cov_matrix(self):
        """Calcul des statistiques de la matrice de cov. intra-classes
        """
        # rang de la matrice de cov. intra-classes 
        rangW = np.linalg.matrix_rank(self.W) 
        # logarithme naturel du déterminant de la matrice de cov. intra-classes
        logDetW = np.linalg.slogdet(self.W)[0] * np.linalg.slogdet(self.W)[1] 
        self.infoCovMatrix = pd.DataFrame([rangW, logDetW],
            index=["Rang de la mat. de cov. intra-classes",
                   "Log. naturel du det. de la mat. de cov. intra-classes"],
            columns=["Valeurs"])
    
    
    def stats_classes(self):
        """ Calcul des statistiques des classes
        """
        # effectifs et fréquences relatives des classes
        targetValues = self.dataset[self.classEtiquette] 
        effClassValues, freqClassValues = freq_relat(targetValues, self.n) 
        self.infoClasses = pd.DataFrame(
            [effClassValues, freqClassValues],
             columns=self.classNames,
             index=["Effectifs", "Frequences"]).transpose()

    
    def stats_wilks(self):
        #---- Statistiques du Lambda de Wilks ----#
        L = wilks(self.Vb, self.Wb) # lambda de Wilks
        ddlNum = self.p * (self.K - 1) # ddl du numérateur
        # Calcul du ddl du dénominateur
        temp = self.p**2 + (self.K-1)**2 - 5
        temp = np.where(temp > 0,
                        np.sqrt(((self.p**2) * ((self.K-1)**2) - 4)/temp),
                        1)
        
        ddlDenom = (2 * self.n - self.p - self.K - 2)/2*temp-(ddlNum - 2)/2
        # Fin calcul du ddl du dénominateur
        # Calcul de la F-statistique
        F_Rao = L ** (1 / temp)
        F_Rao = ((1 - F_Rao) / F_Rao) * (ddlDenom / ddlNum)
        # Fin calcul de la F-statistique
        p_val = p_value(F_Rao, ddlNum, ddlDenom) # p-value du test
        self.infoWilksStats = pd.DataFrame([L, F_Rao, ddlNum, ddlDenom, p_val],
                                  index=["Valeur", "F-Valeur", "DDL num.",
                                         "DDL den.", "p-value"],
                                  columns=["Lambda de Wilks"]).transpose()
    
    
    def fit(self):
        """Apprentissage d'un modèle d'analyse discrimnante linéiare.
        Calcul également des valeurs supplèmentaire pour l'affichage tels que
        la matrice de covariance intra-classe, le lambda de Wilks, la F-stat
        et la p-value.
        """
        #---- Données ----#
        # sélection des valeurs des variables explicatives
        inputValues = self.dataset.drop(self.classEtiquette, axis=1) 
        # sélection des valeurs de la variable catégorielle           
        targetValues = self.dataset[self.classEtiquette] 
        # suppression des valeurs nulles de l'analyse
        inputValues, targetValues = verification_NA(inputValues, targetValues) 
        # transformation des données en numpy object
        inputValues, targetValues = inputValues.values, targetValues.values 
        
        #---- Fonction de classement ----#
        effClassValues, freqClassValues = freq_relat(targetValues, self.n) 
        pi_k = pd.DataFrame(freqClassValues.reshape(1, self.K),
                            columns=self.classNames)
        means = means_class(inputValues, targetValues) # moy cond
        invW = np.linalg.inv(self.W) # matrice inverse de W
        self.intercept_ = np.log(pi_k.values).reshape(1, self.K) - \
            0.5 * np.diagonal(means @ invW @ means.T)
        # coefficients associés aux variables de la fonction de classement
        self.coef_ = (means @ invW).T 
        # récupération des valeurs de la fonction de classement
        self.infoFuncClassement = pd.concat(
            [pd.DataFrame(self.intercept_,
                          columns=self.classNames,
                          index=["Const"]), 
             pd.DataFrame(self.coef_,
                          columns=self.classNames,
                          index=self.varNames)])


    def predict(self, inputData):
        """Prédiction des classes sur des valeurs d'entrée.

        Paramètres
        ----------
        inputData : array-like of shape (n_samples, n_features)
            Valeurs à predire.        
        """
        p = inputData.shape[1] # nombre de descripteurs
        predictedValues = [] # liste contenant les valeurs prédites
        for i in range(inputData.shape[0]):
            omega = inputData.iloc[i].values
            x = omega.reshape(1, p) @ self.coef_ + self.intercept_
            predictedValues.append(np.argmax(x))
        prediction = np.array(self.classNames).take(predictedValues)
        return prediction


    def confusion_matrix(self, y_true, y_pred, graphShow=True):
        """Calcul d'une matrice de confusion.
        
        Paramètres
        ----------
        y_true : Series ou DataFrame
            Vraies valeurs de la variable cible.
        y_pred : array-like of shape (n_samples,) 
            Valeurs predites par le modèle de la variable cible.
            
        Renvoie
        ----------
        conf_mat : matrice de confusion
        """
        # class_names = list(np.unique(y_true))
        class_to_num = {cl: num for num, cl in enumerate(np.unique(y_true))}
        y_true = np.array(y_true.apply(lambda cl: class_to_num[cl]))
        y_pred = np.array(pd.Series(y_pred).apply(lambda cl: class_to_num[cl]))
        confMatrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_true))))
        for ind_p in range(len(np.unique(y_true))):
            for ind_t in range(len(np.unique(y_true))):
                confMatrix[ind_p, ind_t] = (
                    (np.sum((y_pred == ind_p) & (y_true == ind_t))))
        self.confusionMatrix = confMatrix
        
        if graphShow:
            infoConfusionMatrix = pd.DataFrame(
                self.confusionMatrix,index=self.classNames,columns=self.classNames)
            self.confusionMatrixGraph = plt.figure(figsize=(10, 7))
            sns.heatmap(infoConfusionMatrix, annot=True)
            

    def accuracy_score(self, y_true, y_pred):
        """Calcul du taux de précision.
        
        Paramètres
        ----------
        y_true : Series ou DataFrame
            Vraies valeurs de la variable cible.
        y_pred : array-like of shape (n_samples,) 
            Valeurs predites de la variable cible.
            
        Retour:
        ----------
        accuracy : (TP + TN) / (P + N)
        """
        classNumericValues = {cl: num for num, cl in enumerate(self.classNames)}
        y_true = np.array(y_true.apply(lambda cl: classNumericValues[cl]))
        y_pred = np.array(pd.Series(y_pred).apply(lambda cl: classNumericValues[cl]))
        self.accuracy = (np.sum(y_pred == y_true))/y_true.shape[0]

    # à revoir pour le style CSS des tableaux ?
    def discrim_html_output(self):
        """Création d'un reporting en format HTML grâce à la librairie datapane.
        """
        self.stats_dataset()
        self.stats_classes()
        self.stats_pooled_cov_matrix()
        self.stats_wilks()
        report = dp.Report(dp.Text("# Linear Discriminant Analysis"), 
                           dp.Text("## General information about the data"),
                           dp.Table(self.infoDataset),
                           dp.Table(self.infoClasses), 
                           dp.Text("## Informations on the covariance matrix"),
                           dp.Table(self.W), 
                           dp.Table(self.infoCovMatrix),
                           dp.Text("## Function of lda and its' intercept "
                                   "and coefficients"),
                           dp.Table(self.infoFuncClassement),
                           dp.Text("## Statistics. Wilks' Lambda"),
                           dp.Table(self.infoWilksStats))
        report.save(path='DISCRIM-Results.html', open=True)


    def discrim_pdf_output(self):
        """Création d'un reporting en format PDF grâce à la librairie FPDF.
        Les sorites ressemblent à celles de la procédure DISCRIM de SAS.
        """
        self.stats_dataset()
        self.stats_classes()
        self.stats_pooled_cov_matrix()
        self.stats_wilks()
        #---- Création du PDF
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        #---- Information du jeu de données
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, 'General information about the data',
                 border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        for indx, elem in enumerate(self.infoDataset.index):
            pdf.cell(180, 10, str(self.infoDataset.index[indx]) + ': ' +
                     str(self.infoDataset.iloc[indx, 0]), border=0, align='L')
            pdf.ln()
        pdf.ln()
        
        # Statistiques des classes
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, ' '*(len(max(self.infoClasses.index, key=len))*2) + 
                 'Frequences ' + 'Proportions', border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        j = 0
        for indx, elem in enumerate(self.infoClasses.index):
            pdf.cell(80, 10, str(self.infoClasses.index[indx]) + ': ' +
                     str(self.infoClasses.iloc[indx, j]) + '   ' +
                     str(round(self.infoClasses.iloc[indx, j+1], 4)),
                     border=0, align='L')
            pdf.ln()
        pdf.ln()
        #----
        
        #---- Matrice de covariance
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, 'Informations on the covariance matrix',
                 border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        pdf.cell(180, 10, ' '*(len(max(self.infoCovMatrix.index, key=len))*2) +
                 'Values ', border=0, align='L')
        pdf.ln()
        for indx, elem in enumerate(self.infoCovMatrix.index):
            pdf.cell(180, 10, str(self.infoCovMatrix.index[indx]) + ': ' +
                     str(self.infoCovMatrix.iloc[indx, 0]), border=0, align='L')
            pdf.ln()
        pdf.ln()
        #----

        #---- Fonction de classement
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(
            180, 10, "Function of lda and its' intercept and coefficients", 
            border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        # lign with column names
        my_str = ' '*(len(max(self.infoFuncClassement.index, key=len))*2)
        for indx, elem in enumerate(self.infoFuncClassement.columns):
            sub_str = str(self.infoFuncClassement.columns[indx]) + ' '
            my_str += sub_str
        pdf.cell(180, 10, my_str, border=0, align='L')
        pdf.ln()

        my_str = ''
        for indx, elem in enumerate(self.infoFuncClassement.index):
            #print(indx, elem)
            my_str = str(self.infoFuncClassement.index[indx])
            # print(my_str)
            for j in range(len(self.infoFuncClassement.columns)):
                # print(j)
                my_str += ' ' + str(round(
                    self.infoFuncClassement.iloc[indx, j], 6))
                # print(my_str)
            if j == (len(self.infoFuncClassement.columns)-1):
                pdf.cell(150, 10, my_str, border=0, align='L')
                pdf.ln()

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, "Statistics. Wilks' Lambda", border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        
        #---- Lambda de Wilks
        for indx, elem in enumerate(self.infoWilksStats.T.index):
            pdf.cell(180, 10, str(self.infoWilksStats.T.index[indx]) + ': ' +
                     str(self.infoWilksStats.T.iloc[indx, 0]), border=0, 
                         align='L')
            pdf.ln()
        pdf.ln()

        #---- Rendu du PDF
        title = input("Enter the name of the file (without '.pdf'): ")
        pdf.set_compression(True)
        pdf.set_display_mode('fullpage')
        pdf.output(title + '.pdf', 'F')
    
    
    def wilks_decay(self, graphShow=True):
        """Calcul des différentes valeurs de chaque valeur du Lambda de Wilks 
        pour chaque q (allant de 1 à p (nombre de variables)).
        """
        wilksValues, varSelection = [], []
        for name in self.varNames:
            varSelection.append(name)
            # calcul du Lambda de Wilks
            L = wilks_log(self.Vb.loc[varSelection, varSelection],
                          self.Wb.loc[varSelection, varSelection])
            # fin du calcul du Lambda de Wilks
            wilksValues.append(L)
            
        #---- Récupération des valeurs dans un DataFrame ----#
        self.infoWilksDecay = pd.DataFrame(wilksValues,
                                           index=range(1, self.p+1)).transpose()         
       
        #---- Création et récupération du graphique ----#
        if graphShow:
            self.figWilksDecay = plt.figure() 
            plt.title("Décroissance du Lambda de Wilks")
            plt.xlabel("Nombre de variables sélectionnées")
            plt.ylabel("Valeur du Lambda de Wilks")
            plt.xticks(range(1, self.p+1, 2))
            plt.plot(range(1, self.p+1), wilksValues, 
                     '-cx', color='c', mfc='k', mec='k')
        #---- Fin création du graphique ----#
    
    ### à retravailler / to rework
    def stepdisc(self, slentry, method,
                 CONSOLELOG=True, HTMLFILE=None):
        """Sélection de variables avec approche ascendante et descendante.
        Affiche le détail de chaque étape, avec la possibilité d'afficher les
        résultats en console ou dans un fichier HTML.
        Il y a également la possibilité d'afficher la courbe de la décroissance
        du lambda de Wilks si l'utilisateur le spécifie.


        Paramètres
        ----------
        slentry : float
            Risque alpha.
        method : ["forward", "backward"], string
            Méthode de sélection de variables.
        CONSOLELOG : booléen, optionnel
            Affiche les résultats de la fonction directement dans la console 
            Python. La valeur par défaut est True (Vraie).
        HTMLFILE : string, optionnel
            Reporting automatique des résultats dans un fichier html dont 
            l'utilisateur spécifiera le nom (ex : "fichier.html"). Le reporting
            ne se fait pas par défaut.

        Raises
        ------
        ValueError
            Si l'utilisateur tente d'utiliser une méthode non programmée, par 
            exemple "stepwise" (ou tout autre valeur), la fonction renverra une
            erreur.
            Si le nom du fichier pour la sortie automatique html ne comporte pas
            l'extension ".html", la fonction renverra une erreur.

        Returns
        -------
        WilksCurveInfo : pandas DataFrame
            Valeur du lambda de Wilks pour chaque nombre de variables sélectionnées.
        dfResults : pandas DataFrame
            Résultats finaux des variables non retenues ou non éliminées ainsi
            que leurs différentes statistiques.
        Summary : pandas DataFrame
            Résumé des variables retenues ou éliminées ainsi que leurs 
            statistiques.

        """
        #---- GESTION D'ERREURS POTENTIELLES  ----#
        ValidMethod = ["forward", "backward"]
        method = method.lower()
        if method not in ValidMethod:
            raise ValueError(
                "Procédure STEPDISC : le paramètre METHOD doit être l'une des "
                "valeurs suivantes %r" % ValidMethod)
        if HTMLFILE is not None:
            if HTMLFILE[-5:] != ".html":
                raise ValueError(
                    "Le nom du fichier n'est pas valide. Il doit se terminer "
                    "par .html")
        #--------#

        if HTMLFILE is not None:
            createWebFile(HTMLFILE)

        #---- Variables nécessaires pour les différents calculs ----#
        varNames = self.varNames
        colnames = ["R-carre", "F-statistique", "p-value", "Lambda de Wilks"]
        # pour la sortie des résultats
        #--------#

        #---- PROCÉDURE STEPDISC ----#
        if method == "forward":
            #---- DÉBUT DE L'APPROCHE ASCENDANTE ----#
            ListeVarRetenues = []
            SummaryVarRetenues = []
            L_initial = 1  # Valeur du Lambda de Wilks pour q = 0
            for q in range(self.p):
                InfosVariables = []
                for name in varNames:
                    # Calcul du Lambda de Wilks (q+1)
                    WilksVarSelection = ListeVarRetenues+[name]
                    L = wilks_log(self.Vb.loc[WilksVarSelection, WilksVarSelection],
                                  self.Wb.loc[WilksVarSelection, WilksVarSelection])
                    # Fin du calcul du Lambda de Wilks
                    ddl1, ddl2 = self.K-1, self.n-self.K-q  # Calcul des degrés de liberté
                    # Calcul de la statistique F
                    F = ddl2/ddl1 * (L_initial/L-1)
                    R = 1-(L/L_initial)  # Calcul du R² partiel
                    # Calcul de la p-value du test
                    pval = p_value(F, ddl1, ddl2)
                    InfosVariables.append((R, F, pval, L))

                self.dfResults = pd \
                    .DataFrame(
                        InfosVariables,
                        index=varNames,
                        columns=colnames) \
                    .sort_values(
                        by=["F-statistique"],
                        ascending=False)

                enteredVar = self.dfResults["Lambda de Wilks"].idxmin()

                #---- AFFICHAGE CONSOLE ----#
                if CONSOLELOG:
                    print("--------")
                    print("Procédure STEPDISC")
                    print("Sélection ascendante : Étape n°%i" % (q+1))
                    print("DF = %i, %i" % (ddl1, ddl2))
                    print("-------- Détail des résultats --------")
                    print(self.dfResults)
                    print("--------")
                #--------#

                #---- SORTIE HTML ----#
                if HTMLFILE is not None:
                    with open(HTMLFILE, "a") as f:
                        f.write("<h3>S&#233;lection ascendante : &#201;tape "
                                "n&#176;%i</h3>" % (q+1))
                        f.write("<h4>D&#233;tail des r&#233;sultats</h4>")
                        f.write("<p>DF = %i, %i</p><div class='row justify-"
                                "content-md-center'>" % (ddl1, ddl2))
                        f.write(
                            self.dfResults.to_html(classes="table table-striped",
                                              float_format="%.6f",
                                              justify="center",
                                              border=0)
                        )
                        f.write("</div>")
                        f.close()
                #--------#

                if (self.dfResults.loc[enteredVar, "p-value"] > slentry):
                    if CONSOLELOG:
                        print("La valeur de la p-value de la meilleure variable "
                              "(%s) est supérieure au risque fixé (= %f)." %
                              (enteredVar, slentry))
                        print("Aucune variable ne peut être choisie.")
                    if HTMLFILE is not None:
                        with open(HTMLFILE, "a") as f:
                            f.write("<p>La valeur de la p-value de la meilleure "
                                    "variable (%s) est sup&#233;rieure au risque "
                                    "fix&#233; (= %f).\r" % (
                                        enteredVar, slentry))
                            f.write(
                                "Aucune variable ne peut &#234;tre choisie.</p>")
                            f.close()
                    break
                else:
                    ListeVarRetenues.append(enteredVar)
                    varNames.remove(enteredVar)
                    L_initial = self.dfResults.loc[enteredVar,
                                              "Lambda de Wilks"]
                    SummaryVarRetenues.append(
                        list(self.dfResults.loc[enteredVar]))
                    if CONSOLELOG:
                        print("La variable %s est retenue." %
                              (enteredVar))
                    if HTMLFILE is not None:
                        with open(HTMLFILE, "a") as f:
                            f.write("<p>La variable %s est retenue.</p>" % (
                                enteredVar))
                            f.close()

            self.Summary = pd.DataFrame(
                SummaryVarRetenues,
                index=ListeVarRetenues,
                columns=colnames)

            if CONSOLELOG:
                print("Arrêt de la procédure STEPDISC")
                print("--------")
                print("Synthèse de la procédure STEPDISC : Sélection ascendante")
                print("Nombre de variables choisies : %i" % (
                    len(ListeVarRetenues)))
                print("-------- Variables retenues --------")
                print(self.Summary)

            if HTMLFILE is not None:
                with open(HTMLFILE, "a") as f:
                    f.write("<h3>Synth&#232;se de la proc&#233;dure STEPDISC : "
                            "S&#233;lection ascendante</h3>")
                    f.write("<p>Nombre de variables choisies : %i</p>" % (
                        len(ListeVarRetenues)))
                    f.write("<h4>Variables retenues</h4><div class='row justify-"
                            "content-md-center'>")
                    f.write(
                        self.Summary.to_html(classes="table table-striped",
                                        float_format="%.6f",
                                        justify="center",
                                        border=0)
                    )
                    f.write("</div></div></body></html>")
                    f.close()
            #---- FIN DE L'APPROCHE ASCENDANTE ----#

        elif method == "backward":
            #---- DÉBUT DE L'APPROCHE DESCENDANTE ----#
            ListeVarEliminees = []
            SummaryVarEliminees = []
            L_initial = wilks(self.Vb, self.Wb)  # Lamba de Wilks pour q = p
            for q in range(self.p, -1, -1):
                InfosVariables = []
                for name in varNames:
                    # calcul du Lambda de Wilks (q-1)
                    WilksVarSelection = [var for var in varNames if var != name]
                    L = wilks_log(self.Vb.loc[WilksVarSelection, WilksVarSelection],
                                  self.Wb.loc[WilksVarSelection, WilksVarSelection])
                    # fin du calcul du Lambda de Wilks
                    ddl1, ddl2 = self.K-1, self.n-self.K-self.q+1  # calcul des degrés de liberté
                    F = ddl2/ddl1*(L/L_initial-1)  # calcul de la statistique F
                    R = 1-(L_initial/L)  # calcul du R² partiel
                    # calcul de la p-value du test
                    pval = p_value(F, ddl1, ddl2)
                    InfosVariables.append((R, F, pval, L))

                self.dfResults = pd \
                    .DataFrame(
                        InfosVariables,
                        index=varNames,
                        columns=colnames
                    ) \
                    .sort_values(
                        by=["F-statistique"]
                    )

                removedVar = self.dfResults["Lambda de Wilks"].idxmin()

                #---- AFFICHAGE CONSOLE----#
                if CONSOLELOG:
                    print("--------")
                    print("Procédure STEPDISC")
                    print("Sélection descendante : Étape n°%i" % (self.p-q+1))
                    print("DF = %i, %i" % (ddl1, ddl2))
                    print("-------- Détail des résultats --------")
                    print(self.dfResults)
                    print("--------")

                #---- SORTIE HTML ----#
                if HTMLFILE is not None:
                    with open(HTMLFILE, "a") as f:
                        f.write("<h3>S&#233;lection descendante : &#201;tape "
                                "n&#176;%i</h3>" % (self.p-q+1))
                        f.write("<h4>D&#233;tail des r&#233;sultats</h4>")
                        f.write("<p>DF = %i, %i</p><div class='row justify-"
                                "content-md-center'>" % (ddl1, ddl2))
                        f.write(
                            self.dfResults.to_html(classes="table table-striped",
                                              float_format="%.6f",
                                              justify="center",
                                              border=0)
                        )
                        f.write("</div>")
                        f.close()
                    #--------#

                if (self.dfResults.loc[removedVar, "p-value"] < slentry):
                    if CONSOLELOG:
                        print("La valeur de la p-value de la pire variable (%s) "
                              "est inférieure au risque fixé (= %f)." %
                              (removedVar, slentry))
                        print("Aucune variable ne peut être retirée.")
                    if HTMLFILE is not None:
                        with open(HTMLFILE, "a") as f:
                            f.write("<p>La valeur de la p-value de la pire "
                                    "variable (%s) est inf&#233;rieure au risque "
                                    "fix&#233; (= %f)." % (
                                        removedVar, slentry))
                            f.write("Aucune variable ne peut &#234;tre "
                                    "retir&#233;e.</p>")
                    break
                else:
                    ListeVarEliminees.append(removedVar)
                    varNames.remove(removedVar)
                    SummaryVarEliminees.append(
                        list(self.dfResults.loc[removedVar])
                    )
                    L_initial = self.dfResults.loc[removedVar,
                                              "Lambda de Wilks"]
                    if CONSOLELOG:
                        print("La variable %s est éliminée." % (
                            removedVar))
                    if HTMLFILE is not None:
                        with open(HTMLFILE, "a") as f:
                            f.write("<p>La variable %s est &#233;limin&#233;"
                                    "e.</p>" % (removedVar))

            self.Summary = pd.DataFrame(
                SummaryVarEliminees,
                index=ListeVarEliminees,
                columns=colnames
            )

            if CONSOLELOG:
                print("Arrêt de la procédure STEPDISC")
                print("--------")
                print("Résumé de la procédure STEPDISC : Sélection descendante")
                print("Nombre de variables éliminées : %i" % (
                    len(ListeVarEliminees)))
                print("-------- Variables éliminées --------")
                print(self.Summary)

            if HTMLFILE is not None:
                with open(HTMLFILE, "a") as f:
                    f.write("<h3>Synth&#232;se de la proc&#233;dure STEPDISC : "
                            "S&#233;lection ascendante</h3>")
                    f.write("<p>Nombre de variables &#233;limin&#233;es : %i</p>"
                            % (len(ListeVarEliminees)))
                    f.write("<h4>Variables &#233;limin&#233;es</h4><div "
                            "class='row justify-content-md-center'>")
                    f.write(
                        self.Summary.to_html(classes="table table-striped",
                                        float_format="%.6f",
                                        justify="center",
                                        border=0)
                    )
                    f.write("</div></div></body></html>")
                    f.close()
            #---- FIN DE L'APPROCHE DESCENDANTE ----#
        # return dfResults, Summary
        #---- FIN PROCÉDURE STEPDISC ----#
        
    

