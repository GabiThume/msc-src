# encoding:utf-8

# Autor: Gabriela Thumé
# Data: 2/07/2014
# Instituto de Ciências Matemáticas e de Computação – ICMC

# Para executar: python classificacao.py

from numpy import zeros, array, int, loadtxt
import matplotlib.pyplot as plt
import time

def matriz_porcentagem(m):
    """ Calcula a porcentagem em que cada classe ficou classificada para todas as classes. """

    size = m.shape[0]

    for i in range(size):
        acertos = m[i][i]
        erros = sum(m[i][:i]) + sum(m[i][i + 1:])
        total = acertos + erros

        for j in range(size):
            if total != 0:
                m[i][j] = (float(m[i][j]) / total) * 100.0
            else:
                m[i][j] = 0

        if sum(m[i]) < 100:
            m[i][i] += (100 - sum(m[i]))
    return m

def plota_matriz(matriz_confusao, nome):
    """ Plota a matriz de confusão. """

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(array(matriz_confusao), cmap=plt.cm.jet, interpolation='nearest')

    width = len(matriz_confusao)
    height = len(matriz_confusao[0])

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(matriz_confusao[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), range(1, width + 1))
    plt.yticks(range(height), range(1, height + 1))
    plt.savefig(nome + '.png', format='png')

def acuracia_individual(m):
    """ Calcula a acurácia para cada classe. """

    size = m.shape[0]

    for i in range(size):
        acertos = m[i][i]
        erros = sum(m[i][:i])
        if i != size - 1:
            erros += sum(m[i][i + 1:])

        total = acertos + erros
        if acertos != 0:
            acertos = acertos * 100 / (total)
        if erros != 0:
            erros = erros * 100 / (total)

        print "\nClasse {0:d}: Acertos = {1:1.2f} %, Erros = {2:1.2f} %".format(i+1, acertos, erros)

        for j in range(size):
            if i != j and m[i][j] != 0:
                print "\tPorcentagem de erro classificado como {0:d} = {1:1.2f} %".format(j+1, m[i][j] * 100 / (
                    total))

def main():

    data = "dataLabels_50_original.csv"
    result = "resultLabels_50_original.csv"
    dataLabels = loadtxt(open(data, "rb"))
    resultLabels = loadtxt(open(result, "rb"))

    maior_rotulo = dataLabels[len(dataLabels)-1]
    matriz_confusao = zeros([maior_rotulo, maior_rotulo], dtype=int)

    for i in range(dataLabels.size):
        matriz_confusao[dataLabels[i] - 1][resultLabels[i] - 1] += 1

    acuracia_individual(matriz_confusao);
    plota_matriz(matriz_confusao, "matriz_de_confusao"+data)
    plota_matriz(matriz_porcentagem(matriz_confusao), "matriz_de_porcentagens"+data)

main()
