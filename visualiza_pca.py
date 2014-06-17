# coding: utf-8
#
# Visualização do resultado gerado pelo PCA nos vetores gerados por Haralick
# com quantização Luminance
#
# Para executar: python visualiza_pca 
#

import numpy as np
import pylab as py
import string as str

arq_pca = 'resultado_pca_haralick_luminance.txt'
arq_orig = 'caracteristicas/BaseImagens_Haralick6_Luminance_256c_100r.txt'

# abrindo arquivo
f = open(arq_pca, 'r')
linhas = f.readlines()
f.close()
# limpando caracteres indesejaveis
conteudo = str.join(linhas)[1:-2]
vecs = conteudo.split(';')
vecs = [[float(y) for y in x.strip().split(',')] for x in vecs]
# matriz limpa
mat = np.array(vecs)
# components principais
pc1 = mat[:,0]
pc2 = mat[:,1]

# abrindo arquivo original (para ler classes)
f = open(arq_orig, 'r')
linhas = f.readlines()[1:]
f.close()

classes = np.array([int(l.split('\t')[1]) for l in linhas])

print set(classes)

# plotando e identificando cada classe
i = 0
for c in set(classes):
    py.plot(pc1[classes == c], pc2[classes == c], 'o', color=py.cm.jet(i*10))
    i += 1
py.show()
#py.plot(pc1, pc2, 'o')
#py.show()
