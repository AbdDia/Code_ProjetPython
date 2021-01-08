# -*- coding: utf-8 -*-
"""
Application créée dans le cadre d'un projet de L3 IDS à l'Université Lyon 2.

Groupe constitué de Mamadou DIALLO, Aymeric DELEFOSSE et Aleksandra
KRUCHININA.
"""


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from fpdf import FPDF
import datapane as dp
import seaborn as sns


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


def verification_NA(X, Y):
    """Vérification pour les valeurs nulles.
    Les observations avec des valeurs nulles ne sont pas prises
    en compte pour l'analyse.
    La fonction affiche le nombre des observation supprimées.

    Paramètres
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets) 
        Valeurs cibles.

    Retour
    -------
    X : Input data sans NA.
    y : Valeurs cibles sans NA.

    """
    n, p = X.shape
    # il faut concatener les inputs pour supprimer les lignes
    # avec des valeurs nulles
    df = pd.concat((X, Y), axis=1)
    df.dropna(axis=0, inplace=True)
    n_del = n - df.shape[0]
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    print('Attention : ', n_del, ' observations ont été supprimees.')
    return X, Y


def freq_relat(y, n):
    """Calcul des fréquences relatives.

    Paramètres
    ----------
    y : array-like of shape (n_samples,) or (n_samples, n_targets) 
        Target values.

    Retour
    -------
    freq : fréquences relatives par classe
    cnt : nombre d'occurences par classe

    """
    # Nk nombre d'effectifs par classes
    classes, cnt = np.unique(y, return_counts=True)
    freq = cnt/n
    return freq, cnt


def means_class(X, Y):
    """Calcul des moyennes conditionnelles selon le groupe d'appartenance.

    Paramètres
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets) 
        Target values.
        
    Retour
    -------
    means_cl : moyennes conditionnelles par classe
    """
    # y1 convertion des valeurs cibles en numérique
    classes, numericY = np.unique(Y, return_inverse=True)
    # Nk nombre d'effectifs par classes
    cnt = np.bincount(numericY)
    # initiation d'une matrice remplie de 0
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, numericY, X)
    means_cl = means / cnt[:, None]
    return means_cl


def cov_matrix(DATA):
    """Calcul de la matrice de covariance totale (V) ainsi que sa version 
    biaisée (Vb).

    Paramètres
    ----------
    DATA : pandas DataFrame 
        Jeu de données
    """
    n = DATA.shape[0]  # taille de l'échantillon
    V = DATA.cov()  # matrice de covariace totale
    Vb = (n-1)/n * V  # matrice de covariance totale biaisée
    return (V, Vb)


def pooled_cov_matrix(DATA, CLASS):
    """Calcul de la matrice de covariance intra-classe (W) ainsi que sa 
    version biaisée (Wb).

    Paramètres
    ----------
    DATA : pandas DataFrame
        Jeu de données
    CLASS : string
        Nom de la colonne contenant les différentes classes du jeu de données
    """
    n = DATA.shape[0]  # taille de l'échantillon
    K = len(DATA[CLASS].unique())  # nombre de classes
    W = 0  # initialisation de W
    for modalities in DATA[CLASS].unique():
        Vk = DATA.loc[DATA[CLASS] == modalities].cov()
        W += (DATA[CLASS].value_counts()[modalities] - 1) * Vk
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

    Notes
    ----------
    Les matrices Pandas permettent de garder un résultat avec le nom des 
    variables.
    """
    # les paramètres d'entrée doivent être des matrices numpy ou
    # des DataFrame pandas
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

    Notes
    ----------
    Les matrices Pandas permettent de garder un résultat avec le nom des 
    variables.
    """
    detVb = np.linalg.slogdet(Vb)  # log. nat. du dét. de la matrice de cov.
    # totale biaisée
    detWb = np.linalg.slogdet(Wb)  # log. nat. du dét. de la matrice de cov.
    # intra-classes biaisée
    WilksValue = np.exp((detWb[0]*detWb[1])-(detVb[0]*detVb[1]))
    return WilksValue


def wilks_decay(VAR, Vb, Wb):
    """Calcul des différentes valeurs de chaque valeur du Lambda de Wilks pour
    chaque q (allant de 1 à p (nombre de variables)).
    Renvoie une liste de coordonées pour création d'un graphique.

    Paramètres 
    ----------
    VAR : liste de variables explicatives
    Vb : matrice de covariance totale biaisée
    Wb : matrice de covariance intra-classes biaisée
    """
    y, ListeVar = [], []
    for i in VAR:
        ListeVar.append(i)
        # calcul du Lambda de Wilks
        L = wilks_log(Vb.loc[ListeVar, ListeVar],
                      Wb.loc[ListeVar, ListeVar])
        # fin du calcul du Lambda de Wilks
        y.append(L)
    return y


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
        p = 1.0 - stats.f.cdf(1.0/F, ddl1, ddl2)
    else:
        p = 1.0 - stats.f.cdf(F, ddl1, ddl2)
    return p


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


class LDA():
    '''Analyse Discriminante Linéaire et Prédictive reproduisant les calculs 
    et les sorties des méthodes PROC DISCRIM et STEPDISC de SAS
    TBA = to be updated.
    '''

    def __init__(self):
        pass

    def variables_explicatives(self, x):
        """La fonction variables_explicatives() permet de convertir des 
        variables explicatives en variables numériques. 
        Dans le cadre de notre projet LDA il suffit de lui donner en entrer 
        les variables à convertir(explicatives) et il retourne variable un d
        ataframe de variables numériques.
        """

        d = dict()  # Creation d'un dictionnaire vide
        # Apply permet ici de faire une boucle comme avec R.
        # Test puis conversion de la variable en numérique si elle ne l'est pas
        d = x.apply(lambda s: pd.to_numeric(
            s, errors='coerce').notnull().all())
        # Renvoie False si la variable n'est pas numérique, True sinon.
        liste = d.values
        for i in range(len(x.columns)):
            # Conversion de toutes les variables qui ne sont pas numériques 
            # en objet.
            if liste[i] == False:
                # Conversion des types "non-numeric" en "objet"
                x.iloc[:, i] = x.iloc[:, i].astype(object)

        # Recodage des colonnes (variables explicatives) grâce à la fonction
        # get_dummies de pandas
        for i in range(x.shape[1]):
            if x.iloc[:, i].dtype == object:
                dummy = pd.get_dummies(x.iloc[:, i], drop_first=True)
                for j in range(dummy.shape[1]):
                    # Concatenation (rajout des variables recodees à x) pour 
                    # chaque colonne de dummy, avec x le dataframe de base
                    x = pd.concat([x, dummy.iloc[:, j]], axis=1)
        # Suppression des colonnes non numerics
        x = x._get_numeric_data()
        return x

    def fit(self, X, Y):
        """Apprentissage d'un modèle d'analyse discrimnante linéiare.
        Calcul également des valeurs supplèmentaire pour l'affichage tels que
        la matrice de covariance intra-classe, le lambda de Wilks, la F-stat
        et la p-value.

        Paramètres
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        Y : array-like of shape (n_samples,)
            Target values.
            
        Retour
        -------
        self.intercept_ : l'intercept 
        self.coef_ : les coefficients
        """

        # nombre d'effectif (n) et nombre de descripteurs (p)
        n, p = X.shape
        # noms des variables
        cols = X.columns
        # suppression des valeurs nulles de l'analyse
        X, Y = verification_NA(X, Y)
        X = X.values
        Y = Y.values
        # nombre de classes
        K = len(np.unique(Y))
        # noms des classes
        classes = np.unique(Y)
        self.classes_ = classes

        # affichage
        self.Info_ = pd.DataFrame([n, p, K, n-1, n-K, K-1],
                                  index=["Total sample size", "Variables",
                                         "Classes", "Total DDL", 
                                         "DDL in classes", 
                                         "DDL between classes"],
                                  columns=["Count"])

        # frequences relatives des classes
        freq, cnt = freq_relat(Y, n)
        pi_k = pd.DataFrame(freq.reshape(1, K), columns=classes)

        # affichage
        pi_classe = [cnt, freq]
        self.pi_classe_ = pd.DataFrame(pi_classe,
                                       columns=classes,
                                       index=["Frequence", "Proportion"]).transpose()

        # moy cond
        means = means_class(X, Y)

        # matrice de cov. total
        Vt = np.cov(X.T)
        # matrice de cov. total biaisée
        Vb = (n-1)/n*Vt

        # restitution intermediaire
        my_pd = pd.concat((pd.DataFrame(X, columns=cols),
                           pd.DataFrame(Y.reshape(n, 1), columns=["target"])), axis=1)

        # matrice de variance covariance par classe
        # determinants respectifs et log(det)
        mvc = dict()
        dvc = dict()
        logdvc = dict()
        for elem in np.unique(Y):
            mvc[elem] = np.cov(my_pd[my_pd['target'] == elem][cols].T)
            dvc[elem] = np.linalg.det(mvc[elem])
            logdvc[elem] = np.log(dvc[elem])

        # retrouve la matrice W de SAS + LW per classe
        W = 0
        for i, elem in enumerate(mvc.keys()):
            W += (cnt[i]-1)*mvc[elem]
        W = (W/(n-K))  # matrice de covariance intra-classe
        Wb = (n-K)/n * W  # matrice de covariance intra-classes biaisée
        self.W_ = W
        self.Wb_ = Wb

        # matrice inverse de W
        invW = np.linalg.inv(W)

        # rang, determinant et log(det) de la matrice W
        RangW = np.linalg.matrix_rank(W)
        DetW = np.linalg.det(W)
        LogDetW = np.log(DetW)

        # affichage de la matrice
        self.CovI_ = pd.DataFrame([RangW, LogDetW],
                                  index=["Rang of the covariance matrix W",
                                         "Log of the determinant of cov. matrix"],
                                  columns=["Values"])

        # intercept
        self.intercept_ = np.log(pi_k.values).reshape(
            1, K) - 0.5*np.diagonal(means@invW@means.T)

        # coef
        self.coef_ = (means@invW).T

        # affichage
        Intercept_ = pd.DataFrame(
            self.intercept_, columns=classes, index=["Const"])
        Coef_ = pd.DataFrame(self.coef_, columns=classes, index=cols)
        self.InterCoef_ = pd.concat([Intercept_, Coef_])

        # lambda de Wilks
        LW = np.linalg.det(Wb)/np.linalg.det(Vb)
        ddlNum = p * (K-1)
        # valeur intermédiaire pour calcul du ddl dénominateur
        temp = p**2 + (K-1)**2 - 5
        temp = np.where(temp > 0, np.sqrt(((p**2) * ((K-1)**2) - 4)/temp), 1)
        # ddl dénominateur
        ddlDenom = (2*n-p-K-2)/2 * temp - (ddlNum - 2)/2

        # stat de test
        FRao = LW**(1/temp)
        FRao = ((1-FRao)/FRao)*(ddlDenom/ddlNum)

        # p-value i
        p_val = p_value(FRao, ddlNum, ddlDenom)

        # affichage
        Statistiques = [LW, FRao, ddlNum, ddlDenom, p_val]
        self.Stat_ = pd.DataFrame(Statistiques,
                                  index=["Valeur", "Valeur F", "DDL num.",
                                         "DDL den.", "p_value"],
                                  columns=["Wilks' Lambda"]).transpose()

        return self.intercept_, self.coef_

    def predict(self, XP):
        """Prédiction des classes sur les valeurs à prédire XP.

        Paramètres
        ----------
        XP : array-like of shape (n_samples, n_features)
        Valeurs à predire.

        Retour
        -------
        pred : ndarray of shape (n_samples,)
        """

        # nombre de descripteurs
        p = XP.shape[1]
        my_pred = list()
        for i in range(XP.shape[0]):
            omega = XP.iloc[i].values
            x = omega.reshape(1, p)@self.coef_ + self.intercept_
            my_pred.append(np.argmax(x))
        pred = self.classes_.take(my_pred)
        return pred

    def confusion_matrix(self, y_true, y_pred):
        '''Calcul d'une matrice de confusion.
        
        Paramètres
        ----------
        y_true : Series ou DataFrame
            Vraies valeurs de la variable cible.
        y_pred : array-like of shape (n_samples,) 
            Valeurs predites par le modèle de la variable cible.
            
        Retour
        ----------
        conf_mat : matrice de confusion
        '''
        class_names = list(np.unique(y_true))
        class_to_num = {cl: num for num, cl in enumerate(np.unique(y_true))}
        y_true = np.array(y_true.apply(lambda cl: class_to_num[cl]))
        y_pred = np.array(pd.Series(y_pred).apply(lambda cl: class_to_num[cl]))
        conf_mat = np.zeros((len(np.unique(y_true)), len(np.unique(y_true))))
        for ind_p in range(len(np.unique(y_true))):
            for ind_t in range(len(np.unique(y_true))):
                conf_mat[ind_p, ind_t] = (
                    (np.sum((y_pred == ind_p) & (y_true == ind_t))))
        self.confusion_matrix = conf_mat
        return self.confusion_matrix

    def confusion_matrix_graph(self, y_true, y_pred):
        '''Affichage heatmap d'une matrice de confusion.
        
        Paramètres
        ----------
        y_true : Series ou DataFrame
            Vraies valeurs de la variable cible.
        y_pred : array-like of shape (n_samples,) 
            Valeurs predites de la variable cible.
        '''
        class_names = list(np.unique(y_true))
        df_cm = pd.DataFrame(self.confusion_matrix,
                             index=class_names,
                             columns=class_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True)

    def accuracy_score(self, y_true, y_pred):
        '''Calcul du taux de précision.
        
        Paramètres
        ----------
        y_true : Series ou DataFrame
            Vraies valeurs de la variable cible.
        y_pred : array-like of shape (n_samples,) 
            Valeurs predites de la variable cible.
            
        Retour:
        ----------
        accuracy : (TP + TN) / (P + N)
        '''

        class_to_num = {cl: num for num, cl in enumerate(np.unique(y_true))}
        y_true = np.array(y_true.apply(lambda cl: class_to_num[cl]))
        y_pred = np.array(pd.Series(y_pred).apply(lambda cl: class_to_num[cl]))
        accuracy = (np.sum(y_pred == y_true))/y_true.shape[0]
        return accuracy

    def create_HTML(self):
        """Creation du rapport publiable HTML via la fonction report de la 
        librairie datapane.

        """

        # Les variaables d1,d2,d3,d4,d5,d6 reçoivent les valeurs de la fonction fit()
        d1, d2, d3, d4, d5, d6 = self.Info_, self.pi_classe_, self.W_, self.CovI_, self.InterCoef_, self.Stat_
        r = dp.Report(dp.Text("# Linear Discriminant Analysis"), dp.DataTable(d1),
                      dp.DataTable(d2), dp.DataTable(d3), dp.DataTable(d4),
                      dp.DataTable(d5), dp.Table(d6))
        return r.save(path='Rapport_HTML_ProcDiscrim.html', open=True)

    def create_pdf(self):
        """Création d'un fichier pdf.
        Les sorites ressemblent à celles de la procédure DISCRIM de SAS.

        """
        x = self.Info_
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, 'General information about the data',
                 border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        for indx, elem in enumerate(x.index):
            pdf.cell(180, 10, str(x.index[indx]) + ': ' +
                     str(x.iloc[indx, 0]), border=0, align='L')
            pdf.ln()
        pdf.ln()

        x = self.pi_classe_
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, ' '*(len(max(x.index, key=len))*2) + 'Frequences ' +
                 'Proportions', border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        j = 0
        for indx, elem in enumerate(x.index):
            pdf.cell(80, 10, str(x.index[indx]) + ': ' + str(x.iloc[indx, j]) + '   ' +
                     str(round(x.iloc[indx, j+1], 4)), border=0, align='L')
            pdf.ln()
        pdf.ln()

        x = self.CovI_
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, 'Informations on the covariance matrix',
                 border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        pdf.cell(180, 10, ' '*(len(max(x.index, key=len))*2) +
                 'Values ', border=0, align='L')
        pdf.ln()
        for indx, elem in enumerate(x.index):
            pdf.cell(180, 10, str(x.index[indx]) + ': ' +
                     str(x.iloc[indx, 0]), border=0, align='L')
            pdf.ln()
        pdf.ln()

        x = self.InterCoef_
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(
            180, 10, "Function of lda and its' intercept and coefficients", border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)
        # lign with column names
        my_str = ' '*(len(max(x.index, key=len))*2)
        for indx, elem in enumerate(x.columns):
            sub_str = str(x.columns[indx]) + ' '
            my_str += sub_str
        pdf.cell(180, 10, my_str, border=0, align='L')
        pdf.ln()

        my_str = ''
        for indx, elem in enumerate(x.index):
            #print(indx, elem)
            my_str = str(x.index[indx])
            # print(my_str)
            for j in range(len(x.columns)):
                # print(j)
                my_str += ' ' + str(round(x.iloc[indx, j], 6))
                # print(my_str)
            if j == (len(x.columns)-1):
                pdf.cell(150, 10, my_str, border=0, align='L')
                pdf.ln()

        x = self.Stat_.T
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(180, 10, "Statistics. Wilks' Lambda", border=0, align='C')
        pdf.ln()
        pdf.set_font('Arial', '', 12)

        for indx, elem in enumerate(x.index):
            pdf.cell(180, 10, str(x.index[indx]) + ': ' +
                     str(x.iloc[indx, 0]), border=0, align='L')
            pdf.ln()
        pdf.ln()

        title = input("Enter the name of the file (without '.pdf'): ")
        pdf.set_compression(True)
        pdf.set_display_mode('fullpage')
        pdf.output(title + '.pdf', 'F')

    def stepdisc(self, DATA, CLASS, SLENTRY, METHOD,
                 VAR=None, BIGDATA=False, CONSOLELOG=True, HTMLFILE=None):
        """Sélection de variables avec approche ascendante et descendante.
        Affiche le détail de chaque étape, avec la possibilité d'afficher les
        résultats en console ou dans un fichier HTML.
        Il y a également la possibilité d'afficher la courbe de la décroissance
        du lambda de Wilks si l'utilisateur le spécifie.


        Paramètres
        ----------
        DATA : Pandas DataFrame
            Jeu de données
        CLASS : string
            Nom de la variable représentant les classes.
        SLENTRY : float
            Risque alpha.
        METHOD : ["forward", "backward"], string
            Méthode de sélection de variables.
        VAR : array-like, optionnel
            Permet à l'utilisateur de spécifier les valeurs qu'ils souhaitent
            utiliser. Par défaut, le programme prend en compte toutes les variables
            sans la variable représentant la classe.
        BIGDATA : booléen, optionnel
            Si la méthode de sélection de variables est ascendante, permet à
            l'utilisateur d'afficher un graphique de la courbe de décroissance du 
            Lambda de Wilks. Ce paramètre peut être recommandé lors l'utilisateur
            travaille sur des très grandes bases. La valeur par défaut est False
            (Fausse).
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
            Si l'utilisateur tente d'afficher la courbe de décroissance de Wilks
            avec la méthode descendante de sélection de variables, la fonction
            renverra une erreur.
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
        METHOD = METHOD.lower()
        if METHOD not in ValidMethod:
            raise ValueError(
                "Procédure STEPDISC : le paramètre METHOD doit être l'une des "
                "valeurs suivantes %r" % ValidMethod)
        if BIGDATA == True and METHOD == "backward":
            raise ValueError(
                "L'étude de la courbe de décroissance du Lambda de Wilks n'est "
                "possible qu'avec l'approche ascendante.")
        if HTMLFILE is not None:
            if HTMLFILE[-5:] != ".html":
                raise ValueError(
                    "Le nom du fichier n'est pas valide. Il doit se terminer "
                    "par .html")
        #--------#

        if VAR is None:
            # permet d'éviter à l'utilisateur de rentrer toutes les variables
            # tout en lui permettant également de faire un stepdisc sur une
            # sélection de variables
            VAR = list(DATA.columns)
            VAR.remove(CLASS)
        elif VAR is not None and type(VAR) != list:
            # permet de travailler sur des listes (optimisation) quelque soit la
            # saisie utilisateur
            VAR = list(VAR)

        if HTMLFILE is not None:
            createWebFile(HTMLFILE)

        #---- Variables nécessaires pour les différents calculs ----#
        n = DATA.shape[0]  # taille de l'échantillon
        p = len(VAR)  # nombre de variables explicatives
        K = len(DATA[CLASS].unique())  # nombre de classes
        Vb = cov_matrix(DATA)[1]  # matrice de cov. totale biaisée
        Wb = pooled_cov_matrix(DATA, CLASS)[1]  # matrice de cov. intra-classes
        # biaisée
        colnames = ["R-carre", "F-statistique", "p-value", "Lambda de Wilks"]
        # pour la sortie des résultats
        #--------#

        # affichage du graphique de la décroissance du Lambda de Wilks
        if BIGDATA == True and METHOD == "forward":
            fig = plt.figure()  # création d'un objet pour récup. html
            y = wilks_decay(VAR, Vb, Wb)
            #---- Récupération des valeurs ----#
            WilksCurveInfo = pd \
                .DataFrame(
                    y,
                    index=range(1, p+1)) \
                .transpose()  # permet l'affichage sous forme de "liste"
            #---- Création du graphique ----#
            WilksCurveFig = fig.add_subplot(111)
            WilksCurveFig.set_title("Décroissance du Lambda de Wilks")
            WilksCurveFig.set_xlabel("Nombre de variables sélectionnées")
            WilksCurveFig.set_ylabel("Valeur du Lambda de Wilks")
            WilksCurveFig.set_xticks(range(1, p+1, 2))
            WilksCurveFig.plot(range(1, p+1), y, '-cx', color='c',
                               mfc='k', mec='k')
            #---- Fin création du graphique ----#
            if CONSOLELOG:
                print(WilksCurveInfo)
            if HTMLFILE is not None:
                """Écriture de  l'image dans un fichier temporaire et encodage
                avec `base64`, puis intégration de l'image encodée en `base64` 
                dans le fichier `html`. 
                La majorité des navigateurs modernes restituent correctement 
                l'image.
                Évite de devoir sauvegarder l'image "physiquement", sur le 
                disque.
                """
                tmpfile = BytesIO()
                fig.savefig(tmpfile, format='png')
                encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                with open(HTMLFILE, "a") as f:
                    f.write('<h3>D&#233;croissance du Lambda de Wilks</h3>'
                            '<img src=\'data:image/png;base64,{}\'>'.format(encoded))
                    f.write("<h4>Table de valeurs</h4>")
                    f.write(
                        WilksCurveInfo.to_html(classes="table table-striped "
                                               "table-responsive",
                                               float_format="%.6f",
                                               justify="center",
                                               border=0,
                                               index=False)
                    )
                    f.write("<p>&#192; vous de jouer quant au choix du nombre "
                            "optimal de classes !</p>")
                    f.close()

        #---- PROCÉDURE STEPDISC ----#
        if METHOD == "forward":
            #---- DÉBUT DE L'APPROCHE ASCENDANTE ----#
            ListeVarRetenues = []
            SummaryVarRetenues = []
            L_initial = 1  # Valeur du Lambda de Wilks pour q = 0
            for q in range(p):
                InfosVariables = []
                for i in VAR:
                    # Calcul du Lambda de Wilks (q+1)
                    WilksVarSelection = ListeVarRetenues+[i]
                    L = wilks_log(Vb.loc[WilksVarSelection, WilksVarSelection],
                                  Wb.loc[WilksVarSelection, WilksVarSelection])
                    # Fin du calcul du Lambda de Wilks
                    ddl1, ddl2 = K-1, n-K-q  # Calcul des degrés de liberté
                    # Calcul de la statistique F
                    F = ddl2/ddl1 * (L_initial/L-1)
                    R = 1-(L/L_initial)  # Calcul du R² partiel
                    # Calcul de la p-value du test
                    pval = p_value(F, ddl1, ddl2)
                    InfosVariables.append((R, F, pval, L))

                dfResults = pd \
                    .DataFrame(
                        InfosVariables,
                        index=VAR,
                        columns=colnames) \
                    .sort_values(
                        by=["F-statistique"],
                        ascending=False)

                VariableRetenue = dfResults["Lambda de Wilks"].idxmin()

                #---- AFFICHAGE CONSOLE ----#
                if CONSOLELOG:
                    print("--------")
                    print("Procédure STEPDISC")
                    print("Sélection ascendante : Étape n°%i" % (q+1))
                    print("DF = %i, %i" % (ddl1, ddl2))
                    print("-------- Détail des résultats --------")
                    print(dfResults)
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
                            dfResults.to_html(classes="table table-striped",
                                              float_format="%.6f",
                                              justify="center",
                                              border=0)
                        )
                        f.write("</div>")
                        f.close()
                #--------#

                if (dfResults.loc[VariableRetenue, "p-value"] > SLENTRY):
                    if CONSOLELOG:
                        print("La valeur de la p-value de la meilleure variable "
                              "(%s) est supérieure au risque fixé (= %f)." %
                              (VariableRetenue, SLENTRY))
                        print("Aucune variable ne peut être choisie.")
                    if HTMLFILE is not None:
                        with open(HTMLFILE, "a") as f:
                            f.write("<p>La valeur de la p-value de la meilleure "
                                    "variable (%s) est sup&#233;rieure au risque "
                                    "fix&#233; (= %f).\r" % (
                                        VariableRetenue, SLENTRY))
                            f.write(
                                "Aucune variable ne peut &#234;tre choisie.</p>")
                            f.close()
                    break
                else:
                    ListeVarRetenues.append(VariableRetenue)
                    VAR.remove(VariableRetenue)
                    L_initial = dfResults.loc[VariableRetenue,
                                              "Lambda de Wilks"]
                    SummaryVarRetenues.append(
                        list(dfResults.loc[VariableRetenue]))
                    if CONSOLELOG:
                        print("La variable %s est retenue." %
                              (VariableRetenue))
                    if HTMLFILE is not None:
                        with open(HTMLFILE, "a") as f:
                            f.write("<p>La variable %s est retenue.</p>" % (
                                VariableRetenue))
                            f.close()

            Summary = pd.DataFrame(
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
                print(Summary)

            if HTMLFILE is not None:
                with open(HTMLFILE, "a") as f:
                    f.write("<h3>Synth&#232;se de la proc&#233;dure STEPDISC : "
                            "S&#233;lection ascendante</h3>")
                    f.write("<p>Nombre de variables choisies : %i</p>" % (
                        len(ListeVarRetenues)))
                    f.write("<h4>Variables retenues</h4><div class='row justify-"
                            "content-md-center'>")
                    f.write(
                        Summary.to_html(classes="table table-striped",
                                        float_format="%.6f",
                                        justify="center",
                                        border=0)
                    )
                    f.write("</div></div></body></html>")
                    f.close()
            #---- FIN DE L'APPROCHE ASCENDANTE ----#

        elif METHOD == "backward":
            #---- DÉBUT DE L'APPROCHE DESCENDANTE ----#
            ListeVarEliminees = []
            SummaryVarEliminees = []
            L_initial = wilks(Vb, Wb)  # Lamba de Wilks pour q = p
            for q in range(p, -1, -1):
                InfosVariables = []
                for i in VAR:
                    # calcul du Lambda de Wilks (q-1)
                    WilksVarSelection = [var for var in VAR if var != i]
                    L = wilks_log(Vb.loc[WilksVarSelection, WilksVarSelection],
                                  Wb.loc[WilksVarSelection, WilksVarSelection])
                    # fin du calcul du Lambda de Wilks
                    ddl1, ddl2 = K-1, n-K-q+1  # calcul des degrés de liberté
                    F = ddl2/ddl1*(L/L_initial-1)  # calcul de la statistique F
                    R = 1-(L_initial/L)  # calcul du R² partiel
                    # calcul de la p-value du test
                    pval = p_value(F, ddl1, ddl2)
                    InfosVariables.append((R, F, pval, L))

                dfResults = pd \
                    .DataFrame(
                        InfosVariables,
                        index=VAR,
                        columns=colnames
                    ) \
                    .sort_values(
                        by=["F-statistique"]
                    )

                VariableEliminee = dfResults["Lambda de Wilks"].idxmin()

                #---- AFFICHAGE CONSOLE----#
                if CONSOLELOG:
                    print("--------")
                    print("Procédure STEPDISC")
                    print("Sélection descendante : Étape n°%i" % (p-q+1))
                    print("DF = %i, %i" % (ddl1, ddl2))
                    print("-------- Détail des résultats --------")
                    print(dfResults)
                    print("--------")

                #---- SORTIE HTML ----#
                if HTMLFILE is not None:
                    with open(HTMLFILE, "a") as f:
                        f.write("<h3>S&#233;lection descendante : &#201;tape "
                                "n&#176;%i</h3>" % (p-q+1))
                        f.write("<h4>D&#233;tail des r&#233;sultats</h4>")
                        f.write("<p>DF = %i, %i</p><div class='row justify-"
                                "content-md-center'>" % (ddl1, ddl2))
                        f.write(
                            dfResults.to_html(classes="table table-striped",
                                              float_format="%.6f",
                                              justify="center",
                                              border=0)
                        )
                        f.write("</div>")
                        f.close()
                    #--------#

                if (dfResults.loc[VariableEliminee, "p-value"] < SLENTRY):
                    if CONSOLELOG:
                        print("La valeur de la p-value de la pire variable (%s) "
                              "est inférieure au risque fixé (= %f)." %
                              (VariableEliminee, SLENTRY))
                        print("Aucune variable ne peut être retirée.")
                    if HTMLFILE is not None:
                        with open(HTMLFILE, "a") as f:
                            f.write("<p>La valeur de la p-value de la pire "
                                    "variable (%s) est inf&#233;rieure au risque "
                                    "fix&#233; (= %f)." % (
                                        VariableEliminee, SLENTRY))
                            f.write("Aucune variable ne peut &#234;tre "
                                    "retir&#233;e.</p>")
                    break
                else:
                    ListeVarEliminees.append(VariableEliminee)
                    VAR.remove(VariableEliminee)
                    SummaryVarEliminees.append(
                        list(dfResults.loc[VariableEliminee])
                    )
                    L_initial = dfResults.loc[VariableEliminee,
                                              "Lambda de Wilks"]
                    if CONSOLELOG:
                        print("La variable %s est éliminée." % (
                            VariableEliminee))
                    if HTMLFILE is not None:
                        with open(HTMLFILE, "a") as f:
                            f.write("<p>La variable %s est &#233;limin&#233;"
                                    "e.</p>" % (VariableEliminee))

            Summary = pd.DataFrame(
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
                print(Summary)

            if HTMLFILE is not None:
                with open(HTMLFILE, "a") as f:
                    f.write("<h3>Synth&#232;se de la proc&#233;dure STEPDISC : "
                            "S&#233;lection ascendante</h3>")
                    f.write("<p>Nombre de variables &#233;limin&#233;es : %i</p>"
                            % (len(ListeVarEliminees)))
                    f.write("<h4>Variables &#233;limin&#233;es</h4><div "
                            "class='row justify-content-md-center'>")
                    f.write(
                        Summary.to_html(classes="table table-striped",
                                        float_format="%.6f",
                                        justify="center",
                                        border=0)
                    )
                    f.write("</div></div></body></html>")
                    f.close()
            #---- FIN DE L'APPROCHE DESCENDANTE ----#
        if BIGDATA:
            return WilksCurveInfo, dfResults, Summary
        return dfResults, Summary
        #---- FIN PROCÉDURE STEPDISC ----#
