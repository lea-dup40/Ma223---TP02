"""
    DUPIN Léa - Aéro 2 classe F2
    Ma 223 - Tp 2 : Méthode de Cholesky pour la résolution de systèmes linéaires.
    Programme permettant d'obtenir les graphiques attendus sur les erreurs de manière plus précise.
    Institut Polytechnique des Sciences Avancées - IPSA Paris
"""
import numpy as np
import math
import matplotlib as mp
import time
import matplotlib.pyplot as plt
from Cholesky import *
from time import process_time


def ResolCholesky(A, B):
    # Décomposition de Cholesky:
    L = Cholesky(A)

    # Résolution de L * Y = B:
    Aaug = np.hstack([L, B])
    Y = ResolutionSystTriInf(Aaug)

    # Résolution de L(transposée) * X = Y:
    Y = Y.reshape(-1, 1)
    LT = np.transpose(L)
    Taug = np.hstack([LT, Y])
    X = ResolutionSystTriSup(Taug)
    return(X)


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


def calcul(nb):
    size = []
    erreur = []
    try:
        for taille in range(0, nb):
            erreur_list = []
            size.append(taille)
            for i in range(200):
                A, B = init(taille)
                X = ResolCholesky(A, B)
                result = precision(A, X, B)
                erreur_list.append(result)
            somme = 0
            elements = len(erreur_list)
            for i in range(elements):
                somme += erreur_list[i]
            erreur.append(somme/elements)
    except:
        calcul(nb)
    return(size, erreur)


def graph():
    # Mise en page pour mettre les 3 graphiques
    plt.gcf().subplots_adjust(wspace=0.5, hspace=0.5)
    plt.subplot(2, 1, 1)

    # Demande de la taille de la matrice maximale à calculer
    nb = int(input("Taille max de la matrice souhaitée ? \n"))

    # ------ Erreur en fonction de la taille
    plt.subplot(2, 1, 1)

    size, erreur = calcul(nb)

    plt.plot(size, erreur, color='r', label='Méthode de Cholesky')
    # -- Affichage des courbes
    plt.xlabel('Taille de la matrice')
    plt.ylabel('Erreur ||A X - B||')
    plt.title('Erreur en fonction de la taille de la matrice')
    plt.grid(True)
    plt.legend(loc='best')

    # Graphique 2 : Erreur en échelle logarithmique / taille
    plt.subplot(2, 1, 2)
    plt.plot(size, erreur, color='r', label='Méthode de Cholesky')
    plt.title("Erreur en fonction de la taille de la matrice\n Echelle logarithmique ")
    plt.ylabel('Erreur ||A X - B|| (log))')
    plt.xlabel('Taille de la matrice')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

graph()
