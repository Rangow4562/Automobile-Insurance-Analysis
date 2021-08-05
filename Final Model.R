library(dplyr)
library(ggplot2)
library(mlflow)
library(glmnet)
library(carrier)
library(rsample)
library(recipes)
library(skimr)
library(knitr) 

pacman::p_load(lubridate, xgboost,Matrix,Metrics,tidyverse, reshape,dplyr,ggplot2,moments,corrplot,caTools,car,caret,ROCR,earth,ROSE)

df <- read.csv(  file="C:\\Users\\chand\\Downloads\\Marketing-Customer-Value-Analysis.csv",header=TRUE,
                 sep=",")

glimpse(df)
dim(df)
sum(is.na(df))
sapply(df, class)

#Treating data to the correct types
#  MANAGING OUTLIERS
OutlierManagement <- function(x){
  quantiles <- quantile( x, c(.00, .97 ) )
  x[ x < quantiles[1] ] <- quantiles[1]
  x[ x > quantiles[2] ] <- quantiles[2]
  x
}
df$Customer.Lifetime.Value <- OutlierManagement(df$Customer.Lifetime.Value)

df$State = factor(df$State,
                           levels = c('Washington','Arizona','Nevada','Oregon','California'),
                           labels = c(1, 2, 3, 4, 5))

df$Gender = factor(df$Gender,
                            levels = c('M', 'F'),
                            labels = c(1,2))

df$Vehicle.Size = factor(df$Vehicle.Size,
                                  levels = c('Large','Medsize','Small'),
                                  labels = c(1,2,3))

df$Policy = factor(df$Policy,
                            levels = c('Corporate L1','Corporate L2','Corporate L3','Personal L1','Personal L2' ,'Personal L3', 'Special L1', 'Special L2', 'Special L3'),
                            labels = c(1,2,3,4,5,6,7,8,9))

df$Coverage = factor(df$Coverage,
                              levels = c('Basic','Extended','Premium'),
                              labels = c(1,2,3))

df$EmploymentStatus = factor(df$EmploymentStatus, 
                                      levels = c('Employed','Unemployed','Disabled','Medical Leave','Retired'),
                                      labels = c(1,2,3,4,5))

df$Response = factor(df$Response,
                              levels = c('No','Yes'),
                              labels = c(1,2))

df$Engaged <- as.integer(df$Response) - 1

df$Education = factor(df$Education,
                               levels = c('High School or Below', 'Bachelor','College', 'Master', 'Doctor' ),
                               labels = c(1,2,3,4,5))

df$Location.Code = factor(df$Location.Code,
                                   levels = c('Rural', 'Suburban', 'Urban'),
                                   labels = c(1,2,3))

df$Marital.Status = factor(df$Marital.Status,
                                    levels = c('Single','Married','Divorced'),
                                    labels = c(1,2,3))

df$Policy.Type = factor(df$Policy.Type,
                                 levels = c('Corporate Auto', 'Personal Auto', 'Special Auto'),
                                 labels = c(1,2,3))

df$Renew.Offer.Type = factor(df$Renew.Offer.Type,
                                      levels = c('Offer1','Offer2','Offer3','Offer4'),
                                      labels = c(1,2,3,4))

df$Sales.Channel = factor(df$Sales.Channel,
                                   levels = c('Agent', 'Call Center', 'Branch','Web'),
                                   labels = c(1,2,3,4))

df$Vehicle.Class = factor(df$Vehicle.Class,
                                   levels = c('Two-Door Car','Four-Door Car','SUV','Luxury Car','Luxury SUV', 'Sports Car'),
                                   labels = c(1,2,3,4,5,6))




df %>% head() %>% kable()

df %>% skim()

df <- df %>%
  dplyr::select(-Customer,-Effective.To.Date,-Sales.Channel) %>%
  drop_na()

set.seed(seed = 1972) 
train_test_split <-
  rsample::initial_split(
    data = df,     
    prop = 0.75   
  ) 
train_test_split

train_tbl <- train_test_split %>% training() 
test_tbl  <- train_test_split %>% testing() 

recipe_simple <- function(dataset) {
  recipe(Customer.Lifetime.Value ~ ., data = dataset) %>%
    step_string2factor(all_nominal(), -all_outcomes()) %>%
    prep(data = dataset)
}

recipe_prepped <- recipe_simple(dataset = train_tbl)

train_baked <- bake(recipe_prepped, new_data = train_tbl)
test_baked  <- bake(recipe_prepped, new_data = test_tbl)

##################Multiple LInear Regression ##############################################################################

library(MASS)
Reg_1 = lm(Customer.Lifetime.Value ~. ,data = train_baked)
stepAIC(Reg_1)
summary(Reg_1)

Reg_2 = lm(Customer.Lifetime.Value ~ Monthly.Premium.Auto+I(Coverage == '3') +
             I(Education == '1') + I(EmploymentStatus == '4') + 
             I(Marital.Status == '1') + I(Number.of.Open.Complaints == 3) +
             I(Number.of.Open.Complaints == 4) + I(Number.of.Policies == 2) +
             I(Number.of.Policies == 3) + I(Number.of.Policies == 4) + I(Number.of.Policies == 5) +
             I(Number.of.Policies == 6) + I(Number.of.Policies == 7) + I(Number.of.Policies == 8) +
             I(Number.of.Policies == 9) + I(Renew.Offer.Type == '3') + 
             I(Vehicle.Class == '3') + I(Vehicle.Class == '6') ,data = train_baked)
summary(Reg_2)
extractAIC(Reg_2) 

Reg_3 = lm(Customer.Lifetime.Value ~ Monthly.Premium.Auto + I(Marital.Status == '1') +
             I(Number.of.Open.Complaints == 4) + I(Number.of.Policies == 2) +
             I(Number.of.Policies == 3) + I(Number.of.Policies == 4) + I(Number.of.Policies == 5) +
             I(Number.of.Policies == 6) + I(Number.of.Policies == 7) + I(Number.of.Policies == 8) +
             I(Number.of.Policies == 9) + I(Vehicle.Class == '6') ,data = train_baked)
summary(Reg_3)
extractAIC(Reg_3) 

Reg_4 = lm(Customer.Lifetime.Value ~ Monthly.Premium.Auto + I(Marital.Status == '1') +
             I(Number.of.Policies == 2) + I(Number.of.Policies == 3) + I(Number.of.Policies == 4) +
             I(Number.of.Policies == 5) + I(Number.of.Policies == 6) + I(Number.of.Policies == 7) +
             I(Number.of.Policies == 8) + I(Number.of.Policies == 9) + I(Vehicle.Class == '6')
           ,data = train_baked)
summary(Reg_4)
extractAIC(Reg_4) 
## Prediction:- Linear Regression Model.

test_baked$pred_LM = predict(Reg_4,test_baked)
head(test_baked)

## Accuracy Test for Linear Regression.

test_baked$LM_APE = 100 * ( abs(test_baked$Customer.Lifetime.Value - test_baked$pred_LM) / test_baked$Customer.Lifetime.Value )
head(test_baked)

MeanAPE = mean(test_baked$LM_APE)
MedianAPE = median(test_baked$LM_APE)

print(paste('### Mean Accuracy of Linear Regression Model is: ', 100 - MeanAPE))
print(paste('### Median Accuracy of Linear Regression Model is: ', 100 - MedianAPE))
##################LInear Regression using XGBoost##############################################################################

library(xgboost)


# Constructing the Dense matrix on the train and test data
xgtrain <- sparse.model.matrix(Customer.Lifetime.Value ~., data = train_tbl)
head(xgtrain)
xgtrain_label <- train_tbl[,"Customer.Lifetime.Value"]
train_matrix <- xgb.DMatrix(data = as.matrix(xgtrain), label = xgtrain_label)

xgtest <- sparse.model.matrix(Customer.Lifetime.Value ~., data = test_tbl)
head(xgtest)
xgtest_label <- test_tbl[,"Customer.Lifetime.Value"]
test_matrix <- xgb.DMatrix(data = as.matrix(xgtest), label = xgtest_label)

xgdf <- sparse.model.matrix(Customer.Lifetime.Value ~., data = df)
head(xgdf)
xgmini_label <- df[,"Customer.Lifetime.Value"]
df_matrix <- xgb.DMatrix(data = as.matrix(xgdf), label = xgmini_label)

watchlist <- list(train = train_matrix, test = test_matrix)

# fit the model
xgmodel <- xgb.train(data = train_matrix, watchlist = watchlist, 
                          nround = 100, nthread = 4, eta = 0.1, max.depth = 6 ,objective = "reg:linear", eval_metric = "rmse" , verbose = 1) 


pred_tb1 <- predict(xgmodel, newdata = test_matrix)
test_baked$predicted<-pred_tb1
write.csv(pred_tb1, "CLVPredictedValues.csv")
write.csv(test_tbl, "xgTestDataSet.csv")

rmse(test_tbl$Customer.Lifetime.Value, pred_tb1)
postResample(test_tbl$Customer.Lifetime.Value, pred_tb1)

pred_tb2 <- predict(xgmodel, newdata = df_matrix)
write.csv(pred_tb1, "pred_tb1.csv")

rmse( df$Customer.Lifetime.Value, pred_tb2)
postResample(df$Customer.Lifetime.Value, pred_tb2)

xgb.importance(feature_names = names(train_matrix), model = xgmodel)

## Accuracy Test for XGBoostLinear Regression.

plot(test_baked$Customer.Lifetime.Value,test_baked$predicted)
test_baked$LM_APE =  100 * ((abs(test_baked$Customer.Lifetime.Value-test_baked$predicted)/test_baked$Customer.Lifetime.Value))
MeanAPE = mean(test_baked$LM_APE)
MedianAPE = median(test_baked$LM_APE)
print(paste('### Mean Accuracy of XGBoostLinear Regression Model is: ', 100 - MeanAPE))
print(paste('### Median Accuracy of XGBoostLinear Regression Model is: ', 100 - MedianAPE))



test_baked$test_res <- test_baked$Customer.Lifetime.Value - test_baked$predicted 
plot(test_baked$test_res)
