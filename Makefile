#
#	Para executar:
#	./criaArq<NOME DO DESCRITOR> <numero do banco de imagens> <quantidade de cores para quantizar>
#	<numero da primeira imagem> <numero da segunda imagem>
#
#	Banco de imagens: 1- Corel	/	2- Covers	/	3-Paintings
#	Quantidade de cores: 256/128/64/32/16/8
#


all: descritores funcoesAux funcoesArquivo reducao_dimensao
	@g++ descritores.o funcoesAux.o funcoesArquivo.o mainDescritor.cpp -o mainDescritor -I /usr/include/opencv `pkg-config opencv --libs`

debug: descritores funcoesAux funcoesArquivo
	@g++ -g descritores.o funcoesAux.o funcoesArquivo.o mainDescritor.cpp -o mainDescritor -I /usr/include/opencv `pkg-config opencv --libs`

teste: descritores funcoesAux
	@g++ -g descritores.o funcoesAux.o teste.cpp -o teste -I /usr/include/opencv `pkg-config opencv --libs`
	
descritores:
	@g++ -c -g descritores.cpp -I /usr/include/opencv `pkg-config opencv --libs`

funcoesAux:
	@g++ -c -g funcoesAux.cpp -I /usr/include/opencv `pkg-config opencv --libs`
	
funcoesArquivo:
	@g++ -c -g funcoesArquivo.cpp -I /usr/include/opencv `pkg-config opencv --libs`

reducao_dimensao:
	@g++ -o reducao_dimensao reducao_dimensao.cpp -I /usr/include/opencv `pkg-config opencv --libs`

clean:
	rm *.o *.*~ teste *~ mainDescritor reducao_dimensao


