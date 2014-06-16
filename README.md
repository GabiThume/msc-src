
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


Uso
---

Antes de executar, crie um link simbólico para o diretório de imagens:

    ln -s <SEU DIRETÓRIO DA BASE DE IMAGENS> BaseImagens
    
Para compilar o código, apenas rodar o Makefile:

    make
    
Para executar e gerar todos os descritores:

    ./runAllDescriptors.sh
