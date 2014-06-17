
Descrição
---------

Extração de características de imagens, utilizando alguns métodos de quantização e descritores.


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

Redução de dimensionalidade dos vetores de características extraídos.

Métodos de redução:
    
    PCA


Uso
---

Antes de executar, crie um link simbólico para o diretório de imagens:

    ln -s <SEU DIRETÓRIO DA BASE DE IMAGENS> BaseImagens
    
Para compilar o código, apenas rodar o Makefile:

    make
    
Para executar e gerar todos os descritores:

    ./runAllDescriptors.sh
    
Para executar o PCA sobre os vetores, após gerar os descritores:
    
    ./reducao_dimensao
    
    
