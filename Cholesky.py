"""
    DUPIN Léa - Aéro 2 classe F2
    Ma 223 - Tp 2 : Méthode de Cholesky pour la résolution de systèmes linéaires.
    Institut Polytechnique des Sciences Avancées - IPSA Paris
"""
import numpy as np
import math
import matplotlib as mp
import time
from matplotlib import pyplot as pp


def init(taille):
    A = np.random.randint(-5, 5, size=(taille, taille))
    B = np.random.randint(-10, 10, size=(taille, 1))

    # On créé la matrice A définie positive symétrique:
    At = A.transpose()
    A = np.matmul(A, At)
    if np.linalg.det(A) == 0:
        A, B = init(taille)
    else:
        None
    return(A, B)


def Cholesky(A):
    n, m = A.shape
    L = np.zeros((n, n))

    # --- on rentre les formules de décomposition de Cholesky
    # Pour chaque colonne:
    for k in range(n):
        # Pour chaque ligne:
        for i in range(n):
            j = 0
            # Si terme diagonal:
            if i == k:
                somme = 0
                for j in range(0, k):
                    somme += (L[k, j])**2
                L[k, k] = math.sqrt(A[k, k] - somme)
            # Si le terme n'est pas à calculer:
            elif i < k:
                L[i, k] = 0
            # Sinon, si le termes n'est pas diagonal & à calculer:
            else:
                somme = 0
                for j in range(0, k):
                    somme += L[i, j]*L[k, j]
                L[i, k] = (A[i, k] - somme) / (L[k, k])
    return(L)


def ResolutionSystTriInf(Aaug):
    # On créé une copie de Taug
    A = np.copy(Aaug)
    # On récupère la taille de A
    n, m = np.shape(A)
    # On créé une matrice colonne X remplie de 0
    X = np.zeros(n)
    # On calcule les termes solutions de cette matrice X
    for i in range(0, n):
        s = 0
        for j in range(0, i):
            s = s + Aaug[i, j] * X[j]
        X[i] = (Aaug[i, n] - s) / Aaug[i, i]
    # On renvoie la matrice solution X
    return(X)


def ResolutionSystTriSup(Taug):
    # On créé une copie de Taug
    A = np.copy(Taug)
    # On récupère la taille de A
    n, m = np.shape(A)
    # On créé une matrice colonne X remplie de 0
    X = np.zeros(n)
    # On calcule les termes solutions de cette matrice X
    for k in range(n - 1, -1, -1):
        S = 0
        for j in range(k + 1, n):
            S = S + A[k, j] * X[j]
        X[k] = (A[k, -1] - S) / A[k, k]
    # On renvoie la matrice solution X
    return(X)


if __name__ == "__main__":
    taille = int(input("Taille max de la matrice souhaitée ? \n"))

    A, B = init(taille)

    # Décomposition de Cholesky:
    L = Cholesky(A)
    print("\nRésulat:")
    print(L)
    print("\nVérification:")
    print(np.linalg.cholesky(A))

    # Résolution de L * Y = B:
    Aaug = np.hstack([L, B])
    Y = ResolutionSystTriInf(Aaug)

    # Résolution de L(transposée) * X = Y:
    Y = Y.reshape(-1, 1)
    LT = np.transpose(L)
    Taug = np.hstack([LT, Y])
    X = ResolutionSystTriSup(Taug)

    print("Résolution:\n", X)
