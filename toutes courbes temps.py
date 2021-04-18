"""
    DUPIN Léa - Aéro 2 classe F2
    Ma 223 - Tp 2 : Méthode de Cholesky pour la résolution de systèmes linéaires.
    Programme fournissant les graphiques d'analyse de temps de calcul avec différentes méthodes.
    Institut Polytechnique des Sciences Avancées - IPSA Paris
"""

# ---------------- Import des modules nécessaires
import numpy as np
import math
import matplotlib.pyplot as plt
from time import process_time
from Cholesky import *


# ---------------- Partie décomposition LU
def DecompositionLU(A):
    # On récupère la taille de A
    n, m = np.shape(A)
    # On créé une copie de A
    U = np.copy(A)
    # On créé une matrice identité de taille n
    L = np.eye(n)
    if m != n:
        print("La matrice n'est pas carrée.")
        return(None)
    # On calcule les éléments de L et U
    for j in range(n):
        for i in range(j + 1, n):
            pivot = U[i, j] / U[j, j]
            U[i, :] = U[i, :] - pivot * U[j, :]
            L[i, j] = pivot
    # On renvoie les matrices L et U
    return(L, U)


def ResolutionLU(L, U, B):
    Y = []
    # On récupère la taille de B
    n, m = np.shape(B)
    # On résoud le sytème grâce à la décomposition L U
    for i in range(n):
        Y.append(B[i])
        for j in range(i):
            Y[i] = Y[i] - (L[i, j] * Y[j])
        Y[i] = Y[i] / L[i, i]
    X = np.zeros(n)
    for i in range(n, 0, - 1):
        X[i - 1] = (Y[i - 1] - np.dot(U[i - 1, i:], X[i:])) / U[i - 1, i - 1]
    # On renvoie la matrice solution X
    return(X)


def LU(A, B):
    # On décompose la matrice A en deux matrices L et U
    L, U = DecompositionLU(A)
    # On trouve la solution du système
    solution = ResolutionLU(L, U, B)
    # On renvoie la matrice solution du système
    return(solution)


# ---------------- Partie Cholesky
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


# ---------------- Graphiques
def graph():
    # Mise en page pour mettre les 3 graphiques
    plt.gcf().subplots_adjust(wspace=0.5, hspace=0.5)
    plt.subplot(2, 1, 1)

    # Demande de la taille de la matrice maximale à calculer
    taille = int(input("Taille max de la matrice souhaitée ? \n"))

    # ------ Temps de calcul
    print("\n------------LU------------")
    time_list_LU = []
    for i in range(0, taille + 1, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        LU(A, B)
        t = process_time()
        time_list_LU.append(t)
    T_list_LU = []
    for i in range(len(time_list_LU)):
        if i == len(time_list_LU)-1:
            T = time_list_LU[-1] - time_list_LU[0]
        elif i == len(time_list_LU):
            None
        else:
            T = time_list_LU[i + 1] - time_list_LU[i]
        T_list_LU.append(T)
    if taille > 50:
        for i in range(0, len(T_list_LU) - 1):
            print("Le temps de calcul pour une matrice de taille ", i*50, "est de :", T_list_LU[i], "secondes.")
    print("Le temps de calcul pour une matrice de taille ", taille, "est de :", T_list_LU[-2], "secondes.")
    print("\nLe temps de calcul total est de", T_list_LU[-1], "secondes")
    minutes = int(T_list_LU[-1]//60)
    secondes = int(T_list_LU[-1] % 60)
    if minutes == 1:
        print("Soit environ", minutes, "minute et", secondes, "secondes.")
    elif minutes > 1:
        print("Soit environ", minutes, "minutes et", secondes, "secondes.")
    del(T_list_LU[- 1])

    print("\n------------Cholesky------------")
    time_list_Cholesky = []

    for i in range(0, taille + 1, 50):
        A, B = init(i)
        ResolCholesky2(A, B)
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

    if taille > 50:
        for i in range(0, len(T_list_Cholesky) - 1):
            print("Le temps de calcul pour une matrice de taille ", i*50, "est de :", T_list_Cholesky[i], "secondes.")
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

    print("\n------------Solveur python------------")
    time_list_solveur = []

    for i in range(0, taille + 1, 50):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        np.linalg.solve(A, B)
        t = process_time()
        time_list_solveur.append(t)

    # Calculs de temps & création de la liste les contenant tous
    T_list_solveur = []
    for i in range(len(time_list_solveur)):
        if i == len(time_list_solveur)-1:
            T = time_list_solveur[-1] - time_list_solveur[0]
        elif i == len(time_list_solveur):
            None
        else:
            T = time_list_solveur[i + 1] - time_list_solveur[i]
        T_list_solveur.append(T)

    # Affichage des temps en console
    if taille > 50:
        for i in range(0, len(time_list_solveur) - 1):
            print("Le temps de calcul pour une matrice de taille ", i*50, "est de :", T_list_solveur[i], "secondes.")

    print("Le temps de calcul pour une matrice de taille ", taille, "est de :", T_list_solveur[-2], "secondes.")

    print("\nLe temps de calcul total est de", T_list_solveur[-1], "secondes")
    minutes = int(T_list_solveur[-1]//60)
    secondes = int(T_list_solveur[-1] % 60)
    if minutes == 1:
        print("Soit environ", minutes, "minute et", secondes, "secondes.")
    elif minutes > 1:
        print("Soit environ", minutes, "minutes et", secondes, "secondes.")

    # On supprime le temps total afin de pouvoir afficher les temps de calcul
    del(T_list_solveur[- 1])

    abscisse = []
    for i in range(0, taille, 50):
        abscisse.append(i)

    #  -- Création des courbes : LU
    plt.plot(abscisse, T_list_LU, color='g', label='Méthode de Gauss LU')
    #  -- Création des courbes : Cholesky
    plt.plot(abscisse, T_list_Cholesky, color='r', label='linalg.cholesky')
    # -- Création des courbes : Solveur python
    plt.plot(abscisse, T_list_solveur, color='tab:orange', label='linalg.solve')

    # -- Affichage des courbes
    # Graphique 1 : Temps / taille
    plt.title("Temps de calcul en fonction de la taille de la matrice")
    plt.ylabel('Temps de calcul en secondes')
    plt.xlabel('Taille de la matrice')
    plt.legend(loc='best')

    # Graphique 2 : Temps en échelle logarithmique / taille
    plt.subplot(2, 1, 2)
    #  -- Création des courbes : LU
    plt.plot(abscisse, T_list_LU, color='g', label='Méthode de Gauss LU')
    #  -- Création des courbes : Cholesky
    plt.plot(abscisse, T_list_Cholesky, color='r', label='linalg.cholesky')
    # -- Création des courbes : Solveur python
    plt.plot(abscisse, T_list_solveur, color='tab:orange', label='linalg.solve')

    plt.title("Temps de calcul en fonction de la taille de la matrice \n Echelle logarithmique ")
    plt.ylabel('Temps de calcul en secondes (log)')
    plt.xlabel('Taille de la matrice')
    plt.yscale('log')
    plt.legend(loc='best')

    plt.suptitle('Courbes TP 2 - Temps de calcul - Ma223')
    plt.show()

graph()
