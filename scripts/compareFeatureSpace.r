
compara <- function(path = '/home/gabi/Desktop/fatore-mp-a75e8d7ef486/'){ 

 	name <- sprintf("%s%s", path,'original.txt')
 	original <- read.csv(name, sep = " ", header = FALSE)
 	last <- dim(original)[2]-1
	original <- original[,2:last]
 	name <- sprintf("%s%s", path,'artificial.txt')
	artificial <- read.csv(name, sep = " ", header = FALSE)[,2:last]
 	name <- sprintf("%s%s", path,'desbalanceado.txt')
	desbalanceado <- read.csv(name, sep = " ", header = FALSE)[,2:last]
 	name <- sprintf("%s%s", path,'smote.txt')
	smote <- read.csv(name, sep = " ", header = FALSE)[,2:last]

	which(pca$x[,2]==max(pca$x[,2]))

	pca <- prcomp(original)
	plot(pca$x[,1:2])
	art <- scale(artificial,pca$center,pca$scale) %*% pca$rotation
	des <- scale(desbalanceado,pca$center,pca$scale) %*% pca$rotation
	smo <- scale(smote,pca$center,pca$scale) %*% pca$rotation

	classes <- c(rep(1, 100), rep(2,100))
	classes_desbalanceado <- c(rep(1, 50), rep(2,100))
	classes_rebalanceado <- c(rep(1, 75), rep(2,100))

	xmin <- min(pca$x[,1])
	xmax <- max(pca$x[,1])
	ymax <- max(pca$x[,2])
	ymin <- min(pca$x[,2])

	plot(pca$x[,1:2], main="Base original", col=classes)
	plot(des[,1:2], main="Base desbalanceada", col=classes_desbalanceado, xlim=c(xmin, xmax), ylim=c(ymin, ymax))
	plot(art[,1:2], main="Geração Artificial", col=classes_rebalanceado, xlim=c(xmin, xmax), ylim=c(ymin, ymax))
	plot(smo[,1:2], main="SMOTE", col=classes_rebalanceado, xlim=c(xmin, xmax), ylim=c(ymin, ymax))
}
