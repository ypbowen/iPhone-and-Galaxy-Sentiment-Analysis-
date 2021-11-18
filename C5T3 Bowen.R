# Required
library(readr)
library(parallel)
library(foreach)
library(iterators)
library(lattice)
library(ggplot2)
library(doParallel)
library(caret)
library(dplyr)
library(tidyr)
library(doParallel)
library(corrplot)
library(e1071)
library(plotly)
library(mlbench)
library(Hmisc)

# Find how many cores are on your machine
detectCores() # Result = 4 

# Create Cluster with desired number of cores. 
cl <- makeCluster(2)

# Register Cluster
registerDoParallel(cl)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 

# Stop Cluster. After performing your tasks, stop your cluster. 
stopCluster(cl)

#################################### Iphone Dataset ##############################

data<-read.csv('C:/Users/ypbow/Documents/C5T3/iphone_smallmatrix_labeled_8d.csv')
str(data) 
summary(data)


hist(data$iphonesentiment,main=" Distribution",xlab="Scale",ylab="Frequency")
plot_ly(data, x = data$iphonesentiment, type = 'histogram')




##########Check which column has missing values
names(which(colSums(is.na(data))>0))


##########Correlation 
options(max.print = 1000000)
cor(data)
corrData <- cor(data)

corrplot(corrData,tl.cex = .9)
corrData


#Significance levels (p-values) can also be generated using the rcorr function which is found in the Hmisc package. 
#The default method is Pearson, but you can also compute Spearman or Kendall coefficients.

data.cor = cor(data, method = c("spearman"))


data.rcorr = rcorr(as.matrix(data))
data.rcorr

data.coeff = data.rcorr$r
data.p = data.rcorr$P

corrplot(data.cor,tl.cex = .9)

palette = colorRampPalette(c("green", "white", "red")) (40)
heatmap(x = data.cor, col = palette, symm = TRUE)


##########Highly correlation features check
x=cor(data)

highCorr <- findCorrelation(x, cutoff = 0.75, names = TRUE)
print(highCorr)


###########Converting dependant variable  into a factor
data$iphonesentiment<-as.factor(data$iphonesentiment )
str(data)


############################################Model Building###############################

############# C5.0

#Accuracy  0.7727094 
#Kappa     0.5587939
inTrain <- createDataPartition(data$iphonesentiment, p=.75, list = FALSE)
training <- data[ inTrain,]
testing  <- data[-inTrain,]


str(data$iphonesentiment)

fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats=1)
C5Fit1 <- train(iphonesentiment~., data = training, method = "C5.0", trControl=fitControl)

#check the results
C5Fit1

########## Random Forest, 

#Accuracy 0.7695714,
#Kappa    0.548053

inTrain <- createDataPartition(data$iphonesentiment, p=.70, list = FALSE)
training <- data[ inTrain,]
testing  <- data[-inTrain,]
#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train Random Forest with a tuneLenght = 1 (trains with 1 mtry value for RandomForest)

rfFit1 <- train(iphonesentiment~., data = training, method = "rf", trControl=fitControl, tuneLength = 1)
#training results

rfFit1
#plot(rfFit0)


########## SVM

#Accuracy  Kappa    
#0.709819  0.4174809


inTrain <- createDataPartition(data$iphonesentiment, p=.75, list = FALSE)
training <- data[ inTrain,]
testing  <- data[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

SVMFit1 <- train(iphonesentiment~., data = training, method = "svmLinear", trControl=fitControl)


#check the results
SVMFit1
summary(SVMFit1)


########## kknn

#Accuracy     Kappa    
#0.3342553  0.1652510


inTrain <- createDataPartition(data$iphonesentiment, p=.75, list = FALSE)
training <- data[ inTrain,]
testing  <- data[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

kknnFit1 <- train(iphonesentiment~., data = training, method = "kknn", trControl=fitControl)


#check the results
kknnFit1
summary(kknnFit1)


########## PREDICTIONS 

#C5.0
predC5.01<-predict(C5Fit1,testing)
predC5.0
testing$modelPred<-predC5.0
plot(testing$modelPred, main="Model Prediction C5.0",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predC5.0,testing$iphonesentiment)


#Rf
predrf1<-predict(rfFit1,testing)
predrf1
testing$modelPred<-predrf1
plot(testing$modelPred, main="Model Prediction RF",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predrf,testing$iphonesentiment)


#SVM
predSVM1<-predict(SVMFit1,testing)
predSVM1
testing$modelPred<-predSVM1
plot(testing$modelPred, main="Model Prediction SVM",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predSVM1,testing$iphonesentiment)


#kknn
predkknn1<-predict(kknnFit1,testing)
predkknn1
testing$modelPred<-predkknn1
plot(testing$modelPred, main="Model Prediction kknn",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predkknn1,testing$iphonesentiment)

 

#################Confusion Matrix 
#C5.0
cmc5.01 <- confusionMatrix(predC5.01, testing$iphonesentiment) 
cmc5.01

#Create a confusion matrix from random forest predictions 
cmRF1 <- confusionMatrix(predrf1, testing$iphonesentiment) 
cmRF1


#SVM
cmcSVM1 <- confusionMatrix(predSVM1, testing$iphonesentiment) 
cmcSVM1

#kknn
cmckknn1 <- confusionMatrix(predkknn1, testing$iphonesentiment) 
cmckknn1










#########################################Near-Zero Variance#############################################

nzvMetrics<-nearZeroVar(data, saveMetrics = FALSE) 
nzvMetrics


nzv<-nearZeroVar(data, saveMetrics = TRUE) 
nzv


##########Create a new data set and remove near zero variance features
iphoneNZV <- data[,-nzv]
str(iphoneNZV)


iphoneNZV <- data[,-nzvMetrics]
str(iphoneNZV)

#Converting dependent variable into a factor
iphoneNZV$iphonesentiment<-as.factor(iphoneNZV$iphonesentiment )
str(iphoneNZV)



##############################Near-Zero Variance Model Building##########################################

############# C5.0  NZV
#Accuracy   Kappa    
#0.7561643  0.5199937

inTrain <- createDataPartition(iphoneNZV$iphonesentiment, p=.75, list = FALSE)
training <- iphoneNZV[ inTrain,]
testing  <- iphoneNZV[-inTrain,]


str(iphoneNZV$iphonesentiment)

fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats=1)
C5FitNZV <- train(iphonesentiment~., data = training, method = "C5.0", trControl=fitControl)


#check the results
C5FitNZV


##########Random Forest NZV
#Accuracy   Kappa    
#0.7591158  0.5267769

inTrain <- createDataPartition(iphoneNZV$iphonesentiment, p=.70, list = FALSE)
training <-iphoneNZV[ inTrain,]
testing  <- iphoneNZV[-inTrain,]
#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train Random Forest with a tuneLenght = 1 (trains with 1 mtry value for RandomForest)

rfFitNZV <- train(iphonesentiment~., data = training, method = "rf", trControl=fitControl, tuneLength = 1)
#training results

rfFitNZV
#plot(rfFit0)


##########SVM NZV
#Accuracy   Kappa    
# 0.6839292  0.3471833

inTrain <- createDataPartition(iphoneNZV$iphonesentiment, p=.75, list = FALSE)
training <- iphoneNZV[ inTrain,]
testing  <- iphoneNZV[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

SVMFitNZV <- train(iphonesentiment~., data = training, method = "svmLinear", trControl=fitControl)


#check the results
SVMFitNZV
summary(SVMFitNZV)


##########kknn NZV
#Accuracy   Kappa 
#  0.3036350  0.1383702


inTrain <- createDataPartition(iphoneNZV$iphonesentiment, p=.75, list = FALSE)
training <- iphoneNZV[ inTrain,]
testing  <- iphoneNZV[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

kknnFitNZV <- train(iphonesentiment~., data = training, method = "kknn", trControl=fitControl)


#check the results
kknnFitNZV
summary(kknnFitNZV)



##########PREDICTIONS NZV
#C5.0
predC5.0NZV<-predict(C5FitNZV,testing)
predC5.0NZV
testing$modelPred<-predC5.0NZV
plot(testing$modelPred, main="NZV Model Prediction C5.0 ",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predC5.0NZV,testing$iphonesentiment)


#Rf
predrfNZV<-predict(rfFitNZV,testing)
predrfNZV
testing$modelPred<-predrfNZV
plot(testing$modelPred, main="NZV Model Prediction RF",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predrfNZV,testing$iphonesentiment)


#SVM
predSVMNZV<-predict(SVMFitNZV,testing)
predSVMNZV
testing$modelPred<-predSVMNZV
plot(testing$modelPred, main="NZV  Model Prediction SVM",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predSVMNZV,testing$iphonesentiment)




#kknn
predkknnNZV<-predict(kknnFitNZV,testing)
predkknnNZV
testing$modelPred<-predkknnNZV
plot(testing$modelPred, main="NZV   Model Prediction kknn",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predkknnNZV,testing$iphonesentiment)





################# Confusion Matrix NZV
#C5.0
cmc5.0NZV <- confusionMatrix(predC5.0NZV, testing$iphonesentiment) 
cmc5.0NZV

#Create a confusion matrix from random forest predictions 
cmRFNZV <- confusionMatrix(predrfNZV, testing$iphonesentiment) 
cmRFNZV


#SVM
cmSVMNZV <- confusionMatrix(predSVMNZV, testing$iphonesentiment) 
cmSVMNZV

#kknn
cmckknnNZV <- confusionMatrix(predkknnNZV, testing$iphonesentiment) 
cmckknnNZV






##########################################Recursive Feature Elimination ###################################################

#Sampling the data before using RFE
set.seed(123)
iphoneSample <- data[sample(1:nrow(data), 1000, replace=FALSE),]

# Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResults

# Plot results
plot(rfeResults, type=c("g", "o"),main='RFE')


#After identifying features for removal, create a new data set and add the dependent variable.  

# create new data set with rfe recommended features
iphoneRFE <- data[,predictors(rfeResults)]

# add the dependent variable to iphoneRFE
iphoneRFE$iphonesentiment <- data$iphonesentiment

# review outcome
str(iphoneRFE)


###########Converting dependent variable into a factor
iphoneRFE$iphonesentiment<-as.factor(iphoneRFE$iphonesentiment )
str(iphoneRFE)




##############################Building Models  RFE ###############################
############# C5.0 RFE 
#Accuracy     Kappa    
# 0.7713725   0.5564822


inTrain <- createDataPartition(iphoneRFE$iphonesentiment, p=.75, list = FALSE)
training <- iphoneRFE[ inTrain,]
testing  <- iphoneRFE[-inTrain,]

str(iphoneRFE$iphonesentiment)

fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats=1)
C5FitRFE <- train(iphonesentiment~., data = training, method = "C5.0", trControl=fitControl)


#check the results
C5FitRFE

##########Random Forest RFE
#Accuracy   Kappa   
#0.7690188  0.546201

inTrain <- createDataPartition(iphoneRFE$iphonesentiment, p=.70, list = FALSE)
training <- iphoneRFE[ inTrain,]
testing  <- iphoneRFE[-inTrain,]
#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train Random Forest with a tuneLenght = 1 (trains with 1 mtry value for RandomForest)

rfFitRFE <- train(iphonesentiment~., data = training, method = "rf", trControl=fitControl, tuneLength = 1)
#training results

rfFitRFE




##########SVM RFE
#Accuracy   Kappa    
#0.7133163  0.4241382

inTrain <- createDataPartition(iphoneRFE$iphonesentiment, p=.75, list = FALSE)
training <-iphoneRFE[ inTrain,]
testing  <- iphoneRFE[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

SVMFitRFE <- train(iphonesentiment~., data = training, method = "svmLinear", trControl=fitControl)


#check the results
SVMFitRFE
summary(SVMFitRFE)





##########kknn RFE
#Accuracy   Kappa    
#0.3102104  0.1552100


inTrain <- createDataPartition(iphoneRFE$iphonesentiment, p=.75, list = FALSE)
training <- iphoneRFE[ inTrain,]
testing  <- iphoneRFE[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

kknnFitRFE <- train(iphonesentiment~., data = training, method = "kknn", trControl=fitControl)


#check the results
kknnFitRFE
summary(kknnFitRFE)



##########PREDICTIONS RFE
#C5.0
predC5.0RFE<-predict(C5FitRFE,testing)
predC5.0RFE
testing$modelPred<-predC5.0RFE
plot(testing$modelPred, main="RFE Model Prediction C5.0 ",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predC5.0RFE,testing$iphonesentiment)


#Rf
predrfRFE<-predict(rfFitRFE,testing)
predrfRFE
testing$modelPred<-predrfRFE
plot(testing$modelPred, main="RFE Model Prediction RF",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predrfRFE,testing$iphonesentiment)


#SVM
predSVMRFE<-predict(SVMFitRFE,testing)
predSVMRFE
testing$modelPred<-predSVMRFE
plot(testing$modelPred, main="RFE  Model Prediction SVM",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predSVMRFE,testing$iphonesentiment)


#kknn
predkknnRFE<-predict(kknnFitRFE,testing)
predkknnRFE
testing$modelPred<-predkknnRFE
plot(testing$modelPred, main="RFE   Model Prediction kknn",xlab="Scale",ylab="count")
summary(testing$modelPred)


#postResample
postResample(predkknnRFE,testing$iphonesentiment)



################# Confusion Matrix RFE
#C5.0
cmc5.0RFE <- confusionMatrix(predC5.0RFE, testing$iphonesentiment) 
cmc5.0RFE

#Create a confusion matrix from random forest predictions 
cmRFRFE <- confusionMatrix(predrfRFE, testing$iphonesentiment) 
cmRFRFE


#SVM
cmSVMRFE <- confusionMatrix(predSVMRFE, testing$iphonesentiment) 
cmSVMRFE

#kknn
cmckknnRFE <- confusionMatrix(predkknnRFE, testing$iphonesentiment) 
cmckknnRFE




############################# Moving to a Large Matrix###############################


data1<-read.csv('C:/Users/ypbow/Documents/C5T3/iphoneLargeMatrix.csv')
str(data1) 
summary(data1)


hist(data1$iphonesentiment,main=" Distribution",xlab="Scale",ylab="Frequency")
plot_ly(data1, x = data1$iphonesentiment, type = 'histogram')


names(which(colSums(is.na(data1))>0))


data1$iphonesentiment<-as.factor(data1$iphonesentiment )
str(data1)



D1predc5.0<-predict(C5Fit1,data1)
data1$modelPred<-D1predc5.0
plot(data1$modelPred, main="Model Prediction Large Matrixs C5.0",xlab="Scale",ylab="count")

summary(D1predc5.0)
#summary(data1$modelPred)
#  0     1     2     3     4     5 
#18514    0    1972  1361  27    5982 



D1predrf<-predict(rfFit1,data1)
data1$modelPredrf<-D1predrf
plot(data1$modelPredrf, main="Model Prediction Large Matrixs RF",xlab="Scale",ylab="count")
plot(D1predrf, main="Model Prediction Large Matrixs RF",xlab="Scale",ylab="count")

summary(D1predrf)
# 0       1    2      3     4     5 
#19096    0   1964   732   0    6064 
 




scale <- c(Unclear=18514, "Somewhat Negative"=1972,
            "Somewhat Neutral"=1361, "Very Positive"=5982)
scale1
barplot(scale1,col=c( "green4", "yellow3", "olivedrab2", "orange3"), main="iPhone Sentiment Prediction",
        ylab="COUNT", xlab=("SENTIMENT"),
        legend.text = c("18514", "1972", "1361", "5982"),
        args.legend=list(cex=1,x="top"))

    
     
######################################################################################
####################################   Galaxy Dataset  ###############################


dg<-read.csv('C:/Users/ypbow/Documents/C5T3/galaxy_smallmatrix_labeled_9d.csv')
str(dg) 
#summary(dg)


hist(dg$galaxysentiment,main=" Distribution",xlab="Scale",ylab="Frequency")
plot_ly(dg, x = dg$galaxysentiment, type = 'histogram')




##########Check which column has missing values
names(which(colSums(is.na(dg))>0))



##########Correlation 
options(max.print = 1000000)
cor(dg)
corrData <- cor(dg)
corrplot(corrData,tl.cex = .8)
corrData


#Significance levels (p-values) can also be generated using the rcorr function which is found in the Hmisc package. 
#The default method is Pearson, but you can also compute Spearman or Kendall coefficients.
library("Hmisc")
data.cor = cor(dg, method = c("spearman"))


data.rcorr = rcorr(as.matrix(dg))
data.rcorr

data.coeff = data.rcorr$r
data.p = data.rcorr$P

corrplot(data.cor,tl.cex = .7)

palette = colorRampPalette(c("green", "white", "red")) (40)
heatmap(x = data.cor, col = palette, symm = TRUE)


##########Highly correlation features check
x=cor(dg)

highCorrgal <- findCorrelation(x, cutoff = 0.75, names = TRUE)
print(highCorrgal)


###########Converting dependent variable into a factor
dg$galaxysentiment<-as.factor(dg$galaxysentiment)
str(dg)





############################################Model Building Galaxy ###############################

############# C5.0 Galaxy
set.seed(123)

#Accuracy   Kappa    
#0.7684330  0.5369504

#Accuracy Ihone  0.7727094 
#Kappa Ipone     0.5587939
inTrain <- createDataPartition(dg$galaxysentiment, p=.75, list = FALSE)
training <- dg[ inTrain,]
testing  <- dg[-inTrain,]


str(dg$galaxysentiment)

fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats=1)
C5Fit1gal <- train(galaxysentiment~., data = training, method = "C5.0", trControl=fitControl)

#check the results
C5Fit1gal

########## Random Forest Galaxy

#Accuracy   Kappa    
#0.7508926  0.4862217

#Accuracy iPhone 0.7695714,
#Kappa iPhone   0.548053

inTrain <- createDataPartition(dg$galaxysentiment, p=.70, list = FALSE)
training <- dg[ inTrain,]
testing  <- dg[-inTrain,]
#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train Random Forest with a tuneLenght = 1 (trains with 1 mtry value for RandomForest)

rfFit1gal <- train(galaxysentiment~., data = training, method = "rf", trControl=fitControl, tuneLength = 1)
#training results

rfFit1gal



########## SVM Galaxy

#Accuracy   Kappa   
#0.7044196  0.385158


#Accuracy iPhone    Kappa  Iphone    
#0.709819           0.4174809


inTrain <- createDataPartition(dg$galaxysentiment, p=.75, list = FALSE)
training <- dg[ inTrain,]
testing  <- dg[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

SVMFit1gal <- train(galaxysentiment~., data = training, method = "svmLinear", trControl=fitControl)


#check the results
SVMFit1gal
summary(SVMFit1gal)


########## kknn Galaxy


#  Accuracy   Kappa    
#  0.6532428  0.4038071
#  0.7277479  0.4803937
#  0.7409677  0.4952657

#Accuracy iPhone     Kappa  iPhone   
#0.3342553           0.1652510


inTrain <- createDataPartition(dg$galaxysentiment, p=.75, list = FALSE)
training <- dg[ inTrain,]
testing  <- dg[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

kknnFit1gal <- train(galaxysentiment~., data = training, method = "kknn", trControl=fitControl)


#check the results
kknnFit1gal
summary(kknnFit1gal)


########## PREDICTIONS Galaxy

#C5.0 Galaxy
predC5.01gal<-predict(C5Fit1gal,testing)
predC5.0gal
testing$modelPred<-predC5.01gal
plot(testing$modelPred, main="Model Prediction C5.0 Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predC5.01gal,testing$galaxysentiment)


#summary(testing$modelPred)
# 0    1    2    3    4    5 
#340   0   20  190   143  2532 

#Rf Galaxy
predrf1gal<-predict(rfFit1gal,testing)
testing$modelPred<-predrf1gal
plot(testing$modelPred, main="Model Prediction RF Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predrf1gal,testing$galaxysentiment)


#summary(testing$modelPred)
# 0    1    2    3     4    5 
#336   0   19  113   133  2624 

#SVM Galaxy
predSVM1gal<-predict(SVMFit1gal,testing)

testing$modelPred<-predSVM1gal
plot(testing$modelPred, main="Model Prediction SVM  Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predSVM1gal,testing$galaxysentiment)


#summary(testing$modelPred)
# 0    1    2    3     4    5 
#352   1    2  126    73   2671 

#kknn Galaxy
predkknn1gal<-predict(kknnFit1gal,testing)

testing$modelPred<-predkknn1gal
plot(testing$modelPred, main="Model Prediction kknn Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predkknn1gal,testing$galaxysentiment)


#summary(testing$modelPred)
#0     1    2    3      4    5 
#353   11   30  195    187 2449 

#################Confusion Matrix Galaxy
#C5.0 Galaxy
cmc5.01gal <- confusionMatrix(predC5.01gal, testing$galaxysentiment) 
cmc5.01gal

#Create a confusion matrix from random forest predictions 
cmRF1gal <- confusionMatrix(predrf1gal, testing$galaxysentiment) 
cmRF1gal


#SVM Galaxy
cmcSVM1gal <- confusionMatrix(predSVM1gal, testing$galaxysentiment) 
cmcSVM1gal

#kknn Galaxy 
cmckknn1gal <- confusionMatrix(predkknn1gal, testing$galaxysentiment) 
cmckknn1gal





################################################################################################################
#########################################Near-Zero Variance Galaxy #############################################

nzvMetrics<-nearZeroVar(dg, saveMetrics = FALSE) 
nzvMetrics


nzv<-nearZeroVar(dg, saveMetrics = TRUE) 
nzv


##########Create a new data set and remove near zero variance features
galaxyNZV1 <- dg[,-nzv]
str(galaxyNZV)


galaxyNZV <- dg[,-nzvMetrics]
str(galaxyNZV1)

#Converting dependent variable into a factor
galaxyNZV$galaxysentiment<-as.factor(galaxyNZV$galaxysentiment )
str(galaxyNZV)


##############################Near-Zero Variance Model Building Galaxy ##########################################

############# C5.0  NZV Galaxy


# Accuracy      Kappa    
# 0.7496389   0.4902734

#Accuracy iPhone    Kappa iPhone   
#0.7755921           0.5649562

inTrain <- createDataPartition(galaxyNZV$galaxysentimentt, p=.75, list = FALSE)
training <- galaxyNZV[ inTrain,]
testing  <- galaxyNZV[-inTrain,]


str(galaxyNZV$galaxysentiment)

fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats=1)
C5FitNZVgal <- train(galaxysentiment~., data = training, method = "C5.0", trControl=fitControl)


#check the results
C5FitNZVgal


##########Random Forest NZV galaxy

#Accuracy   Kappa    
#0.7534609  0.4983801


#Accuracy iPhone    Kappa iPhone   
#0.7680303           0.5440691

inTrain <- createDataPartition(galaxyNZV$galaxysentimentt, p=.70, list = FALSE)
training <- galaxyNZV[ inTrain,]
testing  <- galaxyNZV[-inTrain,]
#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train Random Forest with a tuneLenght = 1 (trains with 1 mtry value for RandomForest)

rfFitNZVgal <- train(galaxysentiment~., data = training, method = "rf", trControl=fitControl, tuneLength = 1)
#training results

rfFitNZVgal
#plot(rfFit0)


##########SVM NZV Galaxy

#Accuracy   Kappa    
#0.6803574  0.3135801

#Accuracy  iPhone      Kappa iPhone   
#0.7129046              0.4211166

inTrain <- createDataPartition(galaxyNZV$galaxysentimentt, p=.75, list = FALSE)
training <- galaxyNZV[ inTrain,]
testing  <- galaxyNZV[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

SVMFitNZVgal <- train(galaxysentiment~., data = training, method = "svmLinear", trControl=fitControl)


#check the results
SVMFitNZVgal
summary(SVMFitNZVgal)


##########kknn NZV Galaxy


#  Accuracy   Kappa    
#   0.6932761  0.4286671
#   0.7291973  0.4667826
#   0.7377656  0.4758710


#Accuracy iPhone   Kappa iPhone
#0.3321052           0.1635589


inTrain <- createDataPartition(galaxyNZV$galaxysentimentt, p=.75, list = FALSE)
training <- galaxyNZV[ inTrain,]
testing  <- galaxyNZV[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

kknnFitNZVgal <- train(galaxysentiment~., data = training, method = "kknn", trControl=fitControl)


#check the results
kknnFitNZVgal
summary(kknnFitNZVgal)



##########PREDICTIONS NZV Galaxy
#C5.0 Galaxy
predC5.0NZVgal<-predict(C5FitNZVgal,testing)
#predC5.0NZVgal
testing$modelPred<-predC5.0NZVgal
plot(testing$modelPred, main="NZV Model Prediction C5.0  Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predC5.0NZVgal,testing$galaxysentiment)


#summary(testing$modelPred)
# 0     1    2     3    4    5 
#357    0    0    170  128 2570 

#Rf Galaxy
predrfNZVgal<-predict(rfFitNZVgal,testing)
predrfNZVgal
testing$modelPred<-predrfNZVgal
plot(testing$modelPred, main="NZV Model Prediction RF Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predrfNZVgal,testing$galaxysentiment)

#summary(testing$modelPred)
# 0    1      2    3    4    5 
#360   0      0   169  133 2563 


#SVM Galaxy
predSVMNZVgal<-predict(SVMFitNZVgal,testing)
predSVMNZVgal
testing$modelPred<-predSVMNZVgal
plot(testing$modelPred, main="NZV  Model Prediction SVM Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predSVMNZVgal,testing$galaxysentiment)


#summary(testing$modelPred)
#  0    1    2    3    4    5 
# 411   0    0    3   77  2734 

#kknn Galaxy
predkknnNZVgal<-predict(kknnFitNZVgal,testing)
predkknnNZVgal
testing$modelPred<-predkknnNZVgal
plot(testing$modelPred, main="NZV   Model Prediction kknn Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predkknnNZVgal,testing$galaxysentiment)


#summary(testing$modelPred)
#  0    1     2    3     4    5 
# 366   11    8  169    168 2503 


################# Confusion Matrix NZV Galaxy
#C5.0 Galaxy
cmc5.0NZVgal <- confusionMatrix(predC5.0NZVgal, testing$galaxysentiment) 
cmc5.0NZVgal

#Random Forest Galaxy
cmRFNZVgal <- confusionMatrix(predrfNZVgal, testing$galaxysentiment) 
cmRFNZVgal


#SVM Galaxy
cmSVMNZVgal <- confusionMatrix(predSVMNZVgal, testing$galaxysentiment) 
cmSVMNZVgal

#kknn Galaxy
cmckknnNZVgal <- confusionMatrix(predkknnNZVgal, testing$galaxysentiment) 
cmckknnNZVgal



##########################################Recursive Feature Elimination  Galaxy###################################################

dg<-read.csv('C:/Users/ypbow/Documents/C5T3/galaxy_smallmatrix_labeled_9d.csv')


str(dg)
head(dg)
summary(dg)

#Sampling the data before using RFE Galaxy
set.seed(123)


galaxySample <- dg[sample(1:nrow(dg), 1000, replace = FALSE),]

str(galaxySample)
head( galaxySample)
summary(galaxySample$galaxysentiment)
summary(galaxysentiment)
    
    
    # Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

# Use rfe and omit the response variable (attribute 59 galaxysentiment) 
rfeResultgal <- rfe(galaxySample[,1:58], 
                  galaxySample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# Get results
rfeResultgal

# Plot results
plot(rfeResultgal, type=c("g", "o"),main='RFE')


#After identifying features for removal, create a new data set and add the dependent variable.  

# create new data set with rfe recommended features
galaxyRFE <- dg[,predictors(rfeResultgal)]

# add the dependent variable to iphoneRFE
galaxyRFE$galaxysentiment <- dg$galaxysentiment

# review outcome
str(galaxyRFE)


###########Converting dependent variable into a factor
galaxyRFE$galaxysentiment<-as.factor(galaxyRFE$galaxysentiment )
str(galaxyRFE)




##############################Building Models  RFE Galaxy ###############################
############# C5.0 RFE Galaxy


#Accuracy            Kappa    
#0.7666716           0.5321740

#Accuracy iPhone     Kappa  iPhone  
# 0.7713725          0.5564822


inTrain <- createDataPartition(galaxyRFE$galaxysentiment, p=.75, list = FALSE)
training <- galaxyRFE[ inTrain,]
testing  <- galaxyRFE[inTrain,]


fitControl <- trainControl(method = "repeatedcv",  number = 10, repeats=1)
C5FitRFEgal <- train(galaxysentiment~., data = training, method = "C5.0", trControl=fitControl)


#check the results
C5FitRFEgal

##########  Random Forest RFE Galaxy
# Accuracy          Kappa    
# 0.7662608         0.5271638

# Accuracy iPhone   Kappa   iPhone 
# 0.7690188         0.546201

inTrain <- createDataPartition(galaxyRFE$galaxysentiment, p=.70, list = FALSE)
training <- galaxyRFE[ inTrain,]
testing  <- galaxyRFE[-inTrain,]
#10 fold cross validation
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train Random Forest with a tuneLenght = 1 (trains with 1 mtry value for RandomForest)

rfFitRFEgal <- train(galaxysentiment~., data = training, method = "rf", trControl=fitControl, tuneLength = 1)
#training results

rfFitRFEgal




##########  SVM RFE Galaxy 

# Accuracy           Kappa    
# 0.7029721          0.3795827

#Accuracy  iPhone    Kappa  iPhone  
#0.7133163           0.4241382

inTrain <- createDataPartition(galaxyRFE$galaxysentiment, p=.75, list = FALSE)
training <-galaxyRFE[ inTrain,]
testing  <- galaxyRFE[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

SVMFitRFEgal <- train(galaxysentiment~., data = training, method = "svmLinear", trControl=fitControl)


#check the results
SVMFitRFEgal
summary(SVMFitRFEgal)





########## kknn RFE Galaxy

#  Accuracy        Kappa    
# 0.6633224        0.4144139
# 0.7410672        0.4976510

#Accuracy iPhone   Kappa iPhone  
#0.3102104         0.1552100


inTrain <- createDataPartition(galaxyRFE$galaxysentiment, p=.75, list = FALSE)
training <- galaxyRFE[ inTrain,]
testing  <- galaxyRFE[-inTrain,]
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

kknnFitRFEgal <- train(galaxysentiment~., data = training, method = "kknn", trControl=fitControl)


#check the results
kknnFitRFEgal
summary(kknnFitRFEgal)



########## PREDICTIONS RFE Galaxy
#C5.0 Galaxy
predC5.0RFEgal<-predict(C5FitRFEgal,testing)
predC5.0RFEgal
testing$modelPred<-predC5.0RFEgal
plot(testing$modelPred, main="RFE Model Prediction C5.0  Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predC5.0RFEgal,testing$galaxysentiment)

#summary(testing$modelPred)
#0      1    2    3     4     5 
#319    0    17  199   112  2578 


#postResample(predC5.0RFEgal,testing$galaxysentiment)
#Accuracy     Kappa 
#0.7658915 0.5268545 


#Rf Galaxy
predrfRFEgal<-predict(rfFitRFEgal,testing)
predrfRFEgal
testing$modelPred<-predrfRFEgal
plot(testing$modelPred, main="RFE Model Prediction RF Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predrfRFEgal,testing$galaxysentiment)


#summary(testing$modelPred)
# 0     1    2    3     4    5 
# 19    0    17  199   112  2578 

# postResample

  
# Accuracy     Kappa 
# 0.7652713    0.5242697 

#SVM Galaxy
predSVMRFEgal<-predict(SVMFitRFEgal,testing)
predSVMRFEgal
testing$modelPred<-predSVMRFEgal
plot(testing$modelPred, main="RFE  Model Prediction SVM",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predSVMRFEgal,testing$galaxysentiment)

# summary(testing$modelPred)
# 0      1    2    3     4    5 
# 304    1    3   153   64  2700 

# postResample
  # Accuracy     Kappa 
# 0.7048062    0.3793705 


#kknn Galaxy
predkknnRFEgal<-predict(kknnFitRFEgal,testing)
predkknnRFEgal
testing$modelPred<-predkknnRFEgal
plot(testing$modelPred, main="RFE   Model Prediction kknn Galaxy",xlab="Scale",ylab="count")
summary(testing$modelPred)
#postResample
postResample(predkknnRFEgal,testing$galaxysentiment)

# summary(testing$modelPred)
# 0      1    2    3    4    5 
# 334    8   28   208  131  2516 

# postResample
# Accuracy     Kappa 
# 0.7460465    0.4968179 



################# Confusion Matrix RFE Galaxy
#C5.0 Galaxy
cmc5.0RFEgal <- confusionMatrix(predC5.0RFEgal, testing$galaxysentiment) 
cmc5.0RFEgal

# Random Forest Galaxy
cmRFRFEgal <- confusionMatrix(predrfRFEgal, testing$galaxysentiment) 
cmRFRFEgal


#SVM Galaxy
cmSVMRFEgal <- confusionMatrix(predSVMRFEgal, testing$galaxysentiment) 
cmSVMRFEgal

#kknn Galaxy
cmckknnRFEgal <- confusionMatrix(predkknnRFEgal, testing$galaxysentiment) 
cmckknnRFEgal



############################# Moving to a Large Matrix Galaxy###############################


datagal<-read.csv('C:/Users/ypbow/Documents/C5T3/galaxyLargeMatrix.csv')
str(datagal) 
summary(datagal)



names(which(colSums(is.na(datagal))>0))


datagal$galaxysentiment<-as.factor(datagal$galaxysentiment )
str(datagal)



Dgalpredc5.0<-predict(C5Fit1gal,datagal)
datagal$modelPred<-Dgalpredc5.0
plot(datagal$modelPred, main="Model Prediction Large Matrixs C5.0 Galaxy",xlab="Scale",ylab="count")

summary(Dgalpredc5.0)
#summary(Dgalpredc5.0)
#0       1    2      3       4     5 
#19123   0    1972   801     5     5955 



Dgalpredrf<-predict(rfFit1gal,datagal)
datagal$modelPredrf<-Dgalpredrfgal
plot(datagal$modelPredrf, main="Model Prediction Large Matrixs RF",xlab="Scale",ylab="count")
plot(Dgalpredrf, main="Model Prediction Large Matrixs RF Galaxy ",xlab="Scale",ylab="count")

summary(Dgalpredrf)
#  0        1     2      3      4     5 
# 19086     0   1970     84     0   6716 




scale1 <- c(Unclear=19123, "Somewhat Negative"=1972,
           "Somewhat Neutral"=801, "Very Positive"=5955)
scale1
barplot(scale1,col=c( "green4", "yellow3", "olivedrab2", "orange3"), main="Galaxy Sentiment Prediction",
        ylab="COUNT", xlab=("SENTIMENT"),
        legend.text = c("19123", "1972", "801", "5955"),
        args.legend=list(cex=1,x="top"))

