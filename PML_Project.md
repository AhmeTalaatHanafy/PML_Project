---
output: 
  html_document: 
    keep_md: yes
---
#Practical Machine Learning [Course Project] - coursera 

--------------------------------------------------

>By: Ahmed Talaat                    
         15 - 06 - 2018

------------------------------------------------

##Background:

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [ here ](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).


###Data:

The training data for this project are available here: [ pml-training.csv ](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here: [ pml-testing.csv ](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

The data for this project come from this [ source ](http://groupware.les.inf.puc-rio.br/har). If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.


###Goal:

The goal of your project is to **predict the manner in which they did the exercise**. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.


------------------------------------------------

##Getting and Cleaning Data

###Getting 


```r
# Preaparing the environment 
library(knitr)
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.4.4
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.4.4
```

```r
library(rpart)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.4.4
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.4.4
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.4.4
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(corrplot)
```

```
## Warning: package 'corrplot' was built under R version 3.4.4
```

```
## corrplot 0.84 loaded
```

```r
library(survival)
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```r
library(splines)
library(parallel)
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.4.4
```

```
## Loaded gbm 2.1.3
```

```r
set.seed(12345)

# After downloading data at your W-DIR 

training <- read.csv("./pml-training.csv")

testing <- read.csv("./pml-testing.csv")

#Splitting the training set into two datasets

inTrain <- createDataPartition(training$classe,
                               p = 0.7, list = FALSE)
trainSet <- training[inTrain, ]
testSet <- training[-inTrain, ]
dim(trainSet)
```

```
## [1] 13737   160
```

```r
dim(testSet)
```

```
## [1] 5885  160
```

###Cleaning 


```r
#remove near zero variance variables 
##for training set
nzv <- nearZeroVar(trainSet)
trainSet <- trainSet[, -nzv]

##for testing set
testSet <- testSet[, -nzv]

dim(trainSet)
```

```
## [1] 13737   106
```

```r
dim(testSet)
```

```
## [1] 5885  106
```

```r
# remove variables that are mostly NA

AllNA <- sapply(trainSet, function(x) mean(is.na(x))) > 0.95

trainSet <- trainSet[, AllNA == F]

testSet <- testSet[, AllNA == F]

dim(trainSet)
```

```
## [1] 13737    59
```

```r
dim(testSet)
```

```
## [1] 5885   59
```

```r
# remove identification only variables (columns 1 to 5)
trainSet <- trainSet[, -(1:5)]
testSet  <- testSet[, -(1:5)]

dim(trainSet)
```

```
## [1] 13737    54
```

```r
dim(testSet)
```

```
## [1] 5885   54
```

###Correlation Analysis:


```r
corMatrix <- cor(trainSet[, -54])

corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

![](PML_Project_files/figure-html/cor-1.png)<!-- -->

-----------------------------------------------

##Prediction Model Building 

Three methods will be applied to model the regressions (in the Train dataset) and the best one (with higher accuracy when applied to the Test dataset) will be used for the quiz predictions. The methods are: **Random Forests**, **Decision Tree** and **Generalized Boosted Model**, as described below.
A Confusion Matrix is plotted at the end of each analysis to better visualize the accuracy of the models.

**1. Model: Random Forest**


```r
# model random forest

set.seed(12345)

#modFitRF <- train(classe ~ ., data = trainSet, method = "rf", do.tra)

modFitRF <- randomForest(classe ~ ., data = trainSet, do.trace = F) ##much faster 

modFitRF$finalModel
```

```
## NULL
```

```r
#prediction on testset
predict_RF <- predict(modFitRF, newdata = testSet)

confMat_RF <- confusionMatrix(predict_RF, testSet$classe); confMat_RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    6    0    0    0
##          B    0 1132    6    0    0
##          C    0    1 1020   12    0
##          D    0    0    0  952    3
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9952          
##                  95% CI : (0.9931, 0.9968)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.994           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9939   0.9942   0.9876   0.9972
## Specificity            0.9986   0.9987   0.9973   0.9994   1.0000
## Pos Pred Value         0.9964   0.9947   0.9874   0.9969   1.0000
## Neg Pred Value         1.0000   0.9985   0.9988   0.9976   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1924   0.1733   0.1618   0.1833
## Detection Prevalence   0.2855   0.1934   0.1755   0.1623   0.1833
## Balanced Accuracy      0.9993   0.9963   0.9957   0.9935   0.9986
```

```r
#plotting results
plot(confMat_RF$table, col = confMat_RF$byClass, main = paste("Random Forest - Accuracy = ", round(confMat_RF$overall['Accuracy'], 4)))
```

![](PML_Project_files/figure-html/RF-1.png)<!-- -->


**2. Model: Decision Trees**


```r
#model fitting
set.seed(12345)
modFitDT <- rpart(classe ~ ., data = trainSet, method = "class")

fancyRpartPlot(modFitDT)
```

![](PML_Project_files/figure-html/decision-1.png)<!-- -->

```r
# prediction on testset
predict_DT <- predict(modFitDT, newdata = testSet, type = "class")
confMat_DT <- confusionMatrix(predict_DT, testSet$classe)
confMat_DT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1530  269   51   79   16
##          B   35  575   31   25   68
##          C   17   73  743   68   84
##          D   39  146  130  702  128
##          E   53   76   71   90  786
## 
## Overall Statistics
##                                          
##                Accuracy : 0.7368         
##                  95% CI : (0.7253, 0.748)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6656         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9140  0.50483   0.7242   0.7282   0.7264
## Specificity            0.9014  0.96650   0.9502   0.9100   0.9396
## Pos Pred Value         0.7866  0.78338   0.7543   0.6131   0.7305
## Neg Pred Value         0.9635  0.89051   0.9422   0.9447   0.9384
## Prevalence             0.2845  0.19354   0.1743   0.1638   0.1839
## Detection Rate         0.2600  0.09771   0.1263   0.1193   0.1336
## Detection Prevalence   0.3305  0.12472   0.1674   0.1946   0.1828
## Balanced Accuracy      0.9077  0.73566   0.8372   0.8191   0.8330
```

```r
# plot matrix results 
plot(confMat_DT$table, col = confMat_DT$byClass, main = paste("Decision Tree - Accuracy = ", round(confMat_DT$overall['Accuracy'], 4)))
```

![](PML_Project_files/figure-html/decision-2.png)<!-- -->


**3. Model: Generalized Boosted Model**


```r
# model fit
set.seed(12345)
controlGBM <- trainControl(method = "repeatedCV", number = 5, repeats = 1)
```

```
## Warning: `repeats` has no meaning for this resampling method.
```

```r
modFitGBM <- train(classe ~ ., data = trainSet, method = "gbm", trControl = controlGBM, verbose = F)


modFitGBM$finalModel 
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 41 had non-zero influence.
```

```r
# prediction on testset
predict_GBM <- predict(modFitGBM, newdata = testSet)
confMat_GBM <- confusionMatrix(predict_GBM, testSet$classe)

confMat_GBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670   11    0    2    0
##          B    4 1115   16    5    2
##          C    0   12 1006   16    1
##          D    0    1    4  941   10
##          E    0    0    0    0 1069
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9857          
##                  95% CI : (0.9824, 0.9886)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9819          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9789   0.9805   0.9761   0.9880
## Specificity            0.9969   0.9943   0.9940   0.9970   1.0000
## Pos Pred Value         0.9923   0.9764   0.9720   0.9843   1.0000
## Neg Pred Value         0.9990   0.9949   0.9959   0.9953   0.9973
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2838   0.1895   0.1709   0.1599   0.1816
## Detection Prevalence   0.2860   0.1941   0.1759   0.1624   0.1816
## Balanced Accuracy      0.9973   0.9866   0.9873   0.9865   0.9940
```

```r
# plot matrix results
plot(confMat_GBM$table, col = confMat_GBM$byClass, main = paste("GBM - Accuracy = ", round(confMat_GBM$overall['Accuracy'], 4)))
```

![](PML_Project_files/figure-html/gbm-1.png)<!-- -->

-----------------------------------------------

##Applying selected model to Testing Dataset:

The accuracy of the 3 regression modeling methods above are:

    Random Forest : 0.9952
    Decision Tree : 0.7368
    GBM : 0.9857
    
In that case, the **Random Forest model** will be applied to predict the 20 quiz results (testing dataset) as shown below.


```r
predict_testing <- predict(modFitRF, newdata = testing)

predict_testing
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
