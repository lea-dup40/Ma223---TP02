"""
    DUPIN Léa - Aéro 2 classe F2
    Ma 223 - Tp 2 : Méthode de Cholesky pour la résolution de systèmes linéaires.
    Programme permettant d'obtenir les graphiques de temps de calcul avec les deux méthodes de Cholesky.
    Institut Polytechnique des Sciences Avancées - IPSA Paris
"""
import numpy as np
import math
import matplotlib as mp
import time
import matplotlib.pyplot as plt
from Cholesky import *
from time import process_time


def ResolCholesky2(A, B):
    # Décomposition de Cholesky:
    L = np.linalg.cholesky(A)

    # Résolution de L * Y = B:
    Aaug = np.hstack([L, B])
    Y = ResolutionSystTriInf(Aaug)

    # Résolution de L(transposée) * X = Y:
    Y = Y.reshape(-1, 1)
    LT = np.transpose(L)
    Taug = np.hstack([LT, Y])
    X = ResolutionSystTriSup(Taug)
    return(X)


def graph():
    # Mise en page pour mettre les 3 graphiques
    plt.gcf().subplots_adjust(wspace=0.5, hspace=0.5)
    plt.subplot(2, 1, 1)

    # Demande de la taille de la matrice maximale à calculer
    nb = int(input("Taille max de la matrice souhaitée ? \n"))

    time_list_Cholesky = []

    for taille in range(0, nb + 1):
        A, B = init(taille)
        np.linalg.solve(A, B)
        t = process_time()
        time_list_Cholesky.append(t)

    # ------ Temps de calcul
    T_list_Cholesky = []
    for i in range(len(time_list_Cholesky)):
        if i == len(time_list_Cholesky)-1:
            T = time_list_Cholesky[-1] - time_list_Cholesky[0]
        elif i == len(time_list_Cholesky):
            None
        else:
            T = time_list_Cholesky[i + 1] - time_list_Cholesky[i]
        T_list_Cholesky.append(T)

    # Affichage des temps en console
    if taille > 50:
        for i in range(0, len(T_list_Cholesky) - 1, 50):
            print("Le temps de calcul pour une matrice de taille ", i, "est de :", T_list_Cholesky[i], "secondes.")
    print("Le temps de calcul pour une matrice de taille ", taille, "est de :", T_list_Cholesky[-2], "secondes.")

    print("\nLe temps de calcul total est de", T_list_Cholesky[-1], "secondes")
    minutes = int(T_list_Cholesky[-1]//60)
    secondes = int(T_list_Cholesky[-1] % 60)
    if minutes == 1:
        print("Soit environ", minutes, "minute et", secondes, "secondes.")
    elif minutes > 1:
        print("Soit environ", minutes, "minutes et", secondes, "secondes.")

    # On supprime le temps total afin de pouvoir afficher les temps de calcul
    del(T_list_Cholesky[- 1])

    abscisse = []
    for i in range(0, taille):
        abscisse.append(i)

    #  -- Création de la courbe
    plt.plot(abscisse, T_list_Cholesky, color='tab:orange', label='linalg.solve')

    time_list_Cholesky2 = []

    for taille in range(0, nb + 1):
        A, B = init(taille)
        ResolCholesky2(A, B)
        t = process_time()
        time_list_Cholesky2.append(t)

    # ------ Temps de calcul
    T_list_Cholesky2 = []
    for i in range(len(time_list_Cholesky2)):
        if i == len(time_list_Cholesky2)-1:
            T = time_list_Cholesky2[-1] - time_list_Cholesky2[0]
        elif i == len(time_list_Cholesky2):
            None
        else:
            T = time_list_Cholesky2[i + 1] - time_list_Cholesky2[i]
        T_list_Cholesky2.append(T)

    # Affichage des temps en console
    if taille > 50:
        for i in range(0, len(T_list_Cholesky2) - 1, 50):
            print("Le temps de calcul pour une matrice de taille ", i, "est de :", T_list_Cholesky2[i], "secondes.")
    print("Le temps de calcul pour une matrice de taille ", taille, "est de :", T_list_Cholesky2[-2], "secondes.")

    print("\nLe temps de calcul total est de", T_list_Cholesky2[-1], "secondes")
    minutes = int(T_list_Cholesky2[-1]//60)
    secondes = int(T_list_Cholesky2[-1] % 60)
    if minutes == 1:
        print("Soit environ", minutes, "minute et", secondes, "secondes.")
    elif minutes > 1:
        print("Soit environ", minutes, "minutes et", secondes, "secondes.")

    # On supprime le temps total afin de pouvoir afficher les temps de calcul
    del(T_list_Cholesky2[- 1])

    abscisse = []
    for i in range(0, taille):
        abscisse.append(i)

    #  -- Création de la courbe
    plt.plot(abscisse, T_list_Cholesky2, color='b', label='linal.cholesky')

    # -- Affichage de la courbe
    # Graphique 1 : Temps / taille
    plt.title("Temps de calcul en fonction de la taille de la matrice")
    plt.ylabel('Temps de calcul en secondes')
    plt.xlabel('Taille de la matrice')
    plt.grid(True)
    plt.legend(loc='best')
    # Graphique 2 : Temps en échelle logarithmique / taille
    plt.subplot(2, 1, 2)
    plt.plot(abscisse, T_list_Cholesky, color='tab:orange', label='linalg.solve')
    plt.plot(abscisse, T_list_Cholesky2, color='b', label='linal.cholesky')
    plt.title("Temps de calcul en fonction de la taille de la matrice\n Echelle logarithmique ")
    plt.ylabel('Temps de calcul en secondes (log)')
    plt.xlabel('Taille de la matrice')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='best')

    plt.show()

graph()
