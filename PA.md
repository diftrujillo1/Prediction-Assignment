---
title: "Prediction Assignment"
author: "Diego Trujillo"
date: "September 25, 2018"
output: 
  html_fragment:
    self_contained: false
    keep_md: true
  md_document:
    variant: markdown_github
    pandoc_args: "--no-wrap"
---



### First Details

To recreate this project, just clone or download this repo in a PC and open the R project in it called *_Project_*. This will set you the current working directory _(cwd)_ in wherever place you have download the project. 

There are two datasets that must be in the whole project, same structure and same path: `pml-testing.csv`and `pml-training.csv`. 

### Exploratory Analysis

First I loaded the datasets to see which data types where on it, which variables has missing values for almost all the entire dataset. I noticed that _'#DIV/0!'_, _'NA'_ where also missing values when reading it, so it must be said to the function while importing the datasets.


```r
suppressWarnings(suppressMessages(library(tidyverse)))
suppressWarnings(suppressMessages(library(caret)))
suppressWarnings(suppressMessages(library(randomForest)))
suppressWarnings(suppressMessages(library(e1071)))
```

Another important thing is that it is better to deal with no missing values inside the dataset, so it's important to drop every variable in the dataset that has missing values. When droping variables, new training dataset came with 57 variables, instead of 160 and testing dataset with 60 variables instead of 160.


```r
train_set <- read.csv('pml-training.csv', na.strings = c('#DIV/0!', 'NA', ''))
train_set <- train_set[,colSums(is.na(train_set)) == 0]

test_set <- read.csv('pml-testing.csv', na.strings = c('#DIV/0!', 'NA', ''))
test_set <- test_set[,colSums(is.na(test_set)) == 0]
```

There are some variables that I consider as non-predictors variables, this are ids or variables time that are dropped from the _trainin\_set_. This variables are: _index_, _user\_name_, _time\_stamp_, _new\_window_ and _num\_window_. One can use the grepl function if one knows before that these variables doesn't give any extra information to the training model.


```r
train   <- train_set[,-c(1:7)]
test   <- test_set[,-c(1:7)]
```

Now, the cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. The "classe" variable is still in the cleaned training set.

### Partitionning the Data

I created a Data Partition with the train dataset after cleansing it. Partitioning it with 70% for training test and 30% for the testing test. This testing set is different from the one loaded before. The testing set from Data Partition is for the Cross Validation section. The other one is for predicting the results of the quiz. 


```r
data_train <- createDataPartition(train$classe, p = 0.7, list = FALSE)
subset_train <- train[data_train,]
subset_test <- train[-data_train,]
```

### Predicting and Modeling

I defined that the model to use was Random Forest which is the best I know. It is a classification model to try to predict as best as it can the _classe_ variable. I defined a Cross Validation of 5 _'cv = 5'_. If one can reproduce this one must set a seed.


```r
control <- trainControl(method = "cv", 5)

model <- train(classe ~ ., data = subset_train, 
               method = "rf", 
               trControl = control, 
               ntree = 250)

prediction <- predict(model, subset_test)
confusionMatrix(subset_test$classe, prediction)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    6 1128    5    0    0
##          C    0    4 1022    0    0
##          D    0    0   13  951    0
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9951          
##                  95% CI : (0.9929, 0.9967)
##     No Information Rate : 0.2855          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9938          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9965   0.9827   0.9989   1.0000
## Specificity            1.0000   0.9977   0.9992   0.9974   0.9998
## Pos Pred Value         1.0000   0.9903   0.9961   0.9865   0.9991
## Neg Pred Value         0.9986   0.9992   0.9963   0.9998   1.0000
## Prevalence             0.2855   0.1924   0.1767   0.1618   0.1837
## Detection Rate         0.2845   0.1917   0.1737   0.1616   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9982   0.9971   0.9909   0.9982   0.9999
```

The model gives us an accuracy of 0.9950722 or 99.51% and an estimated out-of-sample of 0.0049278 or 0.49%. Which is very good.


#### Predicting Quiz set

Now apply the model to the _test\_set_ (the one that was imported from the csv).


```r
result <- predict(model, test_set[, -length(names(test_set))])
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

### Considerations

One can considerate to improve performance doing it in parallel or comparing multiple models output to determine which one is the fits the best. 


### Acknowledgement

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

http://groupware.les.inf.puc-rio.br/har#sbia_paper_section
