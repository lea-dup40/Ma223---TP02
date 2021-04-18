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


# ---------------- Calculs de précision
def precision(A, X, B):
    # On récupère la taille de B (nombre de colonnes)
    n = len(B)
    # On la redimensionne
    B = np.reshape(B, (1, n))
    # on récupère les éléments de X et B dans une matrice 1-D
    X = np.ravel(X)
    B = np.ravel(B)
    # On calcule l'erreur
    a = np.dot(A, X) - B
    # On renvoie la norme ||A X - B||
    return(np.linalg.norm(a))


# ---------------- Graphiques
def graph():
    # Mise en page pour mettre les 3 graphiques
    plt.gcf().subplots_adjust(wspace=0.5, hspace=0.5)
    plt.subplot(2, 1, 1)

    # Demande de la taille de la matrice maximale à calculer
    taille = int(input("Taille max de la matrice souhaitée ? \n"))

    print("\n------------LU------------")
    size = []
    erreur = []
    for i in range(0, taille + 1):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        X = LU(A, B)
        result = precision(A, X, B)
        size.append(i)
        erreur.append(result)

    #  -- Création des courbes : LU
    plt.plot(size, erreur, color = 'g', label = 'Méthode de Gauss LU')
    plt.subplot(2, 1, 2) 
    plt.plot(size, erreur, color = 'g', label = 'Méthode de Gauss LU')
    plt.subplot(2, 1, 1) 

    print("\n------------Solveur python------------")
    size = []
    erreur = []
    for i in range(0, taille + 1):
        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)
        X = np.linalg.solve(A, B)
        result = precision(A, X, B)
        size.append(i)
        erreur.append(result)

    #  -- Création des courbes : Solveur Python
    plt.plot(size, erreur, color = 'tab:orange', label = 'linalg.solve')
    plt.subplot(2, 1, 2) 
    plt.plot(size, erreur, color = 'tab:orange', label = 'linalg.solve')
    plt.subplot(2, 1, 1) 

    print("\n------------Cholesky------------")
    size = []
    erreur = []
    for i in range(0, taille + 1):
        A, B = init(i)
        X = ResolCholesky2(A, B)
        result = precision(A, X, B)
        size.append(i)
        erreur.append(result)

    #  -- Création des courbes : Cholesky
    plt.plot(size, erreur, color='r', label='linalg.cholesky')
    plt.subplot(2, 1, 2) 
    plt.plot(size, erreur, color='r', label='linalg.cholesky')
    plt.subplot(2, 1, 1) 

    # -- Affichage des courbes
    # Graphique 1 : Erreur / taille
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Erreur ||A X - B||')
    plt.title('Erreur en fonction de la taille de la matrice')
    plt.grid(True)
    plt.legend(loc='best')

    # Graphique 2 : Erreur (log) / taille
    plt.subplot(2, 1, 2) 
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Erreur ||A X - B|| (log)')
    plt.title('Erreur en fonction de la taille de la matrice')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='best')

    plt.suptitle('Courbes TP 2 - Calcul de l\'erreur - Ma223')
    plt.show()

graph()
