library(caret)

# make a training and test set
library(datasets)
data(iris)
data = iris


# prediction = species
trainbool = createDataPartition(data$Species, p = 0.75, list=FALSE)
trainset = data[trainbool,]
testset = data[-trainbool,]

# train a model - check different resampling procedures!
# trControl can adjust function behaviour
# plotting a model with tuning parameters shows the effect of these parameters

# k nearest neighbours
knnFit = train(Species ~ ., data = trainset, method = "knn", preProcess = c("center","scale"), tuneLength=10)  
plot(knnFit)
knnPred = predict(knnFit, newdata = testset)
confusionMatrix(knnPred, testset$Species)

# logistic regression
lrFit = train(Species ~ ., data = trainset, method = "multinom")
lrPred = predict(lrFit, newdata = testset)
confusionMatrix(lrPred, testset$Species)

# random forest
rfFit = train(Species ~., data= trainset, method="rf")
plot(rfFit)
rfPred = predict(rfFit, newdata = testset)
confusionMatrix(rfPred, testset$Species)


###################
# cc dataset      #
###################

# make a training and test set
setwd("..")
data = read.csv("risk_factors_cervical_cancer.csv", na.strings="?")
# data = na.omit(data) very few values left
# change 1-0 columns into factors
factorlist = c("Smokes", "Hormonal.Contraceptives", "IUD", "STDs",                              
                "STDs.condylomatosis", "STDs.cervical.condylomatosis", "STDs.vaginal.condylomatosis",
               "STDs.vulvo.perineal.condylomatosis", "STDs.syphilis", "STDs.pelvic.inflammatory.disease",
               "STDs.genital.herpes", "STDs.molluscum.contagiosum", "STDs.AIDS", "STDs.HIV", "STDs.Hepatitis.B",
               "STDs.HPV", "Dx.Cancer", "Dx.CIN", "Dx.HPV", "Dx", "Hinselmann", "Schiller", "Citology", "Biopsy")
bools = which(colnames(data) %in% factorlist)
for (x in bools){
  data[,x] = as.factor(data[,x])
}
trainbool = createDataPartition(data$Dx.Cancer, p = 0.75, list=FALSE)
trainset = data[trainbool,]
testset = data[-trainbool,]


# train a model - check different resampling procedures!
# trControl can adjust function behaviour
# plotting a model with tuning parameters shows the effect of these parameters
# some of the variables cause errors with the algorithms, the script below removes these
# kappa is important; high accuracy may just lead to "no cancer" predictions

# check for near-zero variables
nzv = nearZeroVar(data, saveMetrics= TRUE)
keep = as.vector(nzv$nzv)
clean = data[,!keep]
clean = cbind(clean, data$Dx.Cancer)
colnames(clean)[19] = "Dx.Cancer"
clean = clean[,-c(14,15)] # too many NAs
trainbool = createDataPartition(clean$Dx.Cancer, p = 0.75, list=FALSE)
trainset = clean[trainbool,]
testset = clean[-trainbool,]
testset = na.omit(testset)

# k nearest neighbours
knnFit = train(Dx.Cancer ~ ., data = trainset, method = "knn", metric="Kappa",
               na.action = na.omit, preProcess = c("center","scale"), tuneLength=10)  
plot(knnFit)
knnPred = predict(knnFit, newdata = testset)
confusionMatrix(knnPred, testset$Dx.Cancer)

# logistic regression
lrFit = train(Dx.Cancer ~ ., data = trainset, method = "glm", metric="Kappa",
              na.action = na.omit, family="binomial")
lrPred = predict(lrFit, newdata = testset)
confusionMatrix(lrPred, testset$Dx.Cancer)

# random forest
rfFit = train(Dx.Cancer ~., data= trainset, method="rf", 
              na.action = na.omit, tuneLength=8, metric="Kappa")
plot(rfFit)
rfPred = predict(rfFit, newdata = testset)
confusionMatrix(rfPred, testset$Dx.Cancer)

# note that all performance is poor... The datasets are imbalanced!
trainctrl = trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter = FALSE, sampling = "down")
# k nearest neighbours
knnFit = train(Dx.Cancer ~ ., data = trainset, method = "knn", metric="Kappa",
               na.action = na.omit, preProcess = c("center","scale"), tuneLength=10, trControl=trainctrl)  
plot(knnFit)
knnPred = predict(knnFit, newdata = testset)
confusionMatrix(knnPred, testset$Dx.Cancer)

# logistic regression
lrFit = train(Dx.Cancer ~ ., data = trainset, method = "glm", metric="Kappa",
              na.action = na.omit, family="binomial", trControl=trainctrl)
lrPred = predict(lrFit, newdata = testset)
confusionMatrix(lrPred, testset$Dx.Cancer)

# random forest
rfFit = train(Dx.Cancer ~., data= trainset, method="rf", 
              na.action = na.omit, tuneLength=8, metric="Kappa", trControl=trainctrl)
plot(rfFit)
rfPred = predict(rfFit, newdata = testset)
confusionMatrix(rfPred, testset$Dx.Cancer)
# note that random forest predicts a lot of people without cancer to have it.. but also the 2 correctly!


# unlabeled data
library(upclass)
bool = createDataPartition(clean$Dx.Cancer, p = 0.5, list=FALSE)
nolab = clean[bool,]
withlab = clean[-bool,]
nolab$Dx.Cancer[nolab$Dx.Cancer == 1] = 0
ucFit = upclassify(withlab, withlab$Dx.Cancer, nolab)
