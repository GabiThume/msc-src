# descriptors = {"BIC": 0, "GCH": 1, "CCV": 2, "Haralick6: 3", "ACC": 4, "LBP": 5, "HOG": 6}
# quantizations = ["Intensity": 0, "Luminance": 1, "Gleam": 2, "MSB": 3, "MSBModified": 4,  "Luma": 5}

# Duas classes

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/artificial/ praia-montanha 0 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/fscores_general.txt;
# best fscore CCV + Luma
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/original/ 2 5 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/artificial/ praia-montanha 1 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/artificial/ elefante-cavalo 0 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/fscores_general.txt;
# best fscore ACC + Gleam
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/original/ 4 2 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/artificial/ elefante-cavalo 1 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/elefante-cavalo/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/artificial/ leafs 0 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/fscores_general.txt;
# best fscore HOG + Luminance
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/original/ 6 1 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/artificial/ leafs 1 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/leafs/fscores.txt;

# Naturalmente desbalanceado

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/BigBen-EiffelTower-PisaTower/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/BigBen-EiffelTower-PisaTower/original/  > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/BigBen-EiffelTower-PisaTower/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/BigBen-EiffelTower-PisaTower/artificial/ BigBen-EiffelTower-PisaTower 0 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/BigBen-EiffelTower-PisaTower/fscores_general.txt;
# best fscore HOG + Intensity
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/BigBen-EiffelTower-PisaTower/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced//BigBen-EiffelTower-PisaTower/original/ 6 0 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced//BigBen-EiffelTower-PisaTower/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/BigBen-EiffelTower-PisaTower/artificial/ BigBen-EiffelTower-PisaTower 1 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/BigBen-EiffelTower-PisaTower/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/original/  > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/artificial/ eiffel-ponte 0 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/fscores_general.txt;
# best fscore HOG + Luminance
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/original/ 6 1 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/artificial/ eiffel-ponte 1 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-ponte/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/artificial/ eiffel-41 0 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/fscores_general.txt;
# best fscore HOG + MSB
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/original/ 6 3 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/artificial/ eiffel-41 1 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/eiffel-41/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/artificial/ bigben-museu 0 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/fscores_general.txt;
# best fscore HOG + Gleam
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/original/ 6 2 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/artificial/ bigben-museu 1 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/bigben-museu/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/artificial/ industrial-suburb 0 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/fscores_general.txt;
# best fscore HOG + Luma
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/original/ 6 5 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/artificial/ industrial-suburb 1 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/industrial-suburb/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/artificial/ TrafalgarSquare-MadeleineChurch-Pantheon 0 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/fscores_general.txt;
# best fscore HOG + GLeam
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/original/ 6 2 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/artificial/ TrafalgarSquare-MadeleineChurch-Pantheon 1 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon/fscores.txt;

# Multiclasses

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/artificial/ corel 0 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/fscores_general.txt;
# best fscore LBP + Gleam
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/original/ 5 2 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/artificial/ corel 1 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/artificial/ tropical_fruits 0 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/fscores_general.txt;
# best fscore LBP + Luma
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/original/ 5 5 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/artificial/ tropical_fruits 1 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/tropical_fruits1400/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/artificial/ caltech 0 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/fscores_general.txt;
# best fscore HOG + Intensity
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/original/ 6 0 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/artificial/ caltech 1 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Caltech_600/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/artificial/ dtd-3 0 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/fscores_general.txt;
# best fscore LBP + Luminance
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/original/ 5 1 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/saida_melhor.txt
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/artificial/ dtd-3 1 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/dtd-3/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/artificial/ leafsnap 0 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/fscores_general.txt;
# best fscore LBP + Luminance
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/original/ 5 1 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/artificial/ corel 1 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/leafsnap-1000/fscores.txt;

# Muitas imagens

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/original > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/artificial/ shark-fish 0 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/fscores_general.txt;
# best fscore LBP + Luma
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/original/ 5 5 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/artificial/ shark-fish 1 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/shark-fish/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/original/ > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/saida_todos.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/artificial/ deer-ship 0 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/fscores_general.txt;
# best fscore HOG + Gleam
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/original/ 6 2 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/saida_melhor.txt;
# best fscore HOG + Luminance
./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/original/ 6 1 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/artificial/ deer-ship 1 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/muitas_imagens/deer-ship/fscores.txt;

# Fine tunning

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha-fine-tunning3/ /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha-fine-tunning3/original/ 2 5 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha-fine-tunning3/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha-fine-tunning3/artificial/ praia-montanha-fine-tunning3 1 > /media/gabi/Dados\ 1/data/experiments/datasets/duas_classes/praia-montanha-fine-tunning3/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon-fine-tunning/ /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon-fine-tunning/original/ 6 2 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon-fine-tunning/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon-fine-tunning/artificial/ TrafalgarSquare-MadeleineChurch-Pantheon-fine-tunning 1 > /media/gabi/Dados\ 1/data/experiments/datasets/naturaly_unbalanced/TrafalgarSquare-MadeleineChurch-Pantheon-fine-tunning/fscores.txt;

./bin/experiment /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000-fine-tunning/ /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000-fine-tunning/original/ 5 2 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000-fine-tunning/saida_melhor.txt;
python scripts/plotFscores.py /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000-fine-tunning/artificial/ corel-tunning 1 > /media/gabi/Dados\ 1/data/experiments/datasets/multiclasses/Corel1000-fine-tunning/fscores.txt;
