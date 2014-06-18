
Descrição
---------


1. __Extração de características de imagens, utilizando alguns métodos de quantização e descritores.__


Métodos de quantização:

    Gleam
    Intensity
    Luminance
    MSB

Descritores:

    BIC
    GCH
    CCV
    Haralick
    ACC

2. __Redução de dimensionalidade dos vetores de características extraídos.__

Métodos de redução:
    
    PCA
    Entropia

3. __Classificação das imagens utilizando os vetores reduzidos.__

Classificador:

    Naive Bayes


Uso
---

Antes de executar, crie um link simbólico para o diretório de imagens:

    ln -s <SEU DIRETORIO DA BASE DE IMAGENS> BaseImagens
    
Para compilar o código, apenas rodar o Makefile:

    make
    
Para executar e gerar todos os descritores:

    ./runAllDescriptors.sh
    
Para reduzir a dimensão dos vetores e aplicar a classificação, após terem sido gerados os descritores:

    ./reducao_dimensao <DIRETORIO DOS VETORES> <METODO> <LISTA DE PARAMETROS>

Opcões para métodos e parâmetros:

    [0] Nenhum:
        Somente o classificador será utilizado, sobre os vetores extraídos sem redução.
    [1] PCA: 
        - <nAtributos>: número de atributos da projeção
    [2] Entropia:
        - <tJanela>: tamanho da janela
    [3] Todos:
        - <nAtributos>: número de atributos da projeção
        - <tJanela>: tamanho da janela
      

