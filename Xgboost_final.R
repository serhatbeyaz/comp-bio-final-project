library(caret)
library(plyr)
library(recipes)
library(dplyr)
library(neuralnet)
library(FactoMineR)
library(factoextra)


#Get the whole dataset for training and test
train = read.csv("train_data.csv")
test = read.csv("test_data.csv")
feat = as.data.frame(read.csv("genetic_feat.csv"))

head(feat)
nrow(feat)


####### Filter dataset ####

train = train[train$Dataset == "train" & train$QA == 1,]
head(train)

train = select(train, -17)

train = na.omit(train)
test = na.omit(test)

whole_df = rbind(train, test)
summary(whole_df)

plot(density(whole_df$Synergy.score))

#Convert factors into numeric 

new_df = whole_df

allvalues <- unique(union(feat$X, new_df$Cell.line.name))  # Get the unique cell_line names among datasets
feat$X <- as.numeric(factor(feat$X, levels = allvalues))   # Change the cell_lines to factor and then numeric
new_df$Cell.line.name <- as.numeric(factor(new_df$Cell.line.name, levels = allvalues)) 

alldrugs <- unique(union(new_df$Compound.A, new_df$Compound.B))  # Same idea with the drugs
new_df$Compound.A <- as.numeric(factor(new_df$Compound.A, levels = alldrugs))
new_df$Compound.B <- as.numeric(factor(new_df$Compound.B, levels = alldrugs))

#Function for normalization
normalize = function(x){
  mean = mean(x)
  sd = sd(x)
  x = x-mean
  x = x/sd
  return(x)
}


new_df[,4:12] = new_df[,4:12] %>% mutate_all(normalize) # Center and scale the numerics: substract mean divide by sd 

head(new_df[,4:12])
summary(new_df[,4:12])


#Bind the mutation dataset

colnames(feat)[1] = "Cell.line.name"
new_df = right_join(new_df, feat, by="Cell.line.name")
head(new_df)
nrow(new_df)

##### Create new df with the Drug monotherapy features replaced with each other ####
## The reason is that synergy score for each drug combination should be the same regardless of their naming (A-B == B-A) ##
## But differential occurance of drugs in each column may create bias as if orders of the drugs are important ##

copy_df = new_df
sec = copy_df[,3]
four = copy_df[,5]
last = copy_df[, 9:11]
copy_df[,3] = copy_df[, 2]
copy_df[,5] = copy_df[, 4]
copy_df[,9:11] = copy_df[, 6:8]
copy_df[, 2] = sec
copy_df[, 4] = four
copy_df[, 6:8] = last

copy_df[55,1:12]
new_df[55,1:12]

new_df = rbind(new_df, copy_df)

#### Split the dataset into train and test

train_set = new_df[new_df$Dataset == "train",] 
test_set = new_df[new_df$Dataset == 'test',]

train_set = select(train_set, -c(13,14,15,16))
test_set = select(test_set, -c(13,14,15,16))

train_set = na.omit(train_set)
test_set = na.omit(test_set)

## Check correlation btw features
library(GGally)
head(train_set[,1:12])

ggpairs(train_set[4:10])


#Shuffle the dataset

train_set = train_set[sample(1:nrow(train_set)), ] ##Shuffle the train set before training
test_set = test_set[sample(1:nrow(test_set)), ]

head(train_set[,1:12])

trainX = select(train_set, -c(12))  #Exclude synergy score
trainY = train_set[,12]  #Assing synergy score to trainY

testX = select(test_set, -c(12))  #Same for the test set
testY = test_set[,12]

set.seed(1)


## Activate Parallel Processing
library(parallel)
library(doParallel)

cluster = makeCluster(detectCores() - 2)
registerDoParallel(cluster)

###### XGBOOST model for predicting the synergy score #########


## Set the parameters for starting the xgboost model
## This grid will look at the performance with different combinations of max_depth" 
tune_grid1 = expand.grid(nrounds = seq(from = 200, to = 1000, by = 50),
                         max_depth = c(2, 3, 4, 5, 6),
                         eta = 0.05,
                         gamma = 0.1,
                         colsample_bytree = 0.5,
                         min_child_weight = 0.5,
                         subsample = 0.5)

## Cross validation for training
tune_control <- trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds 
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results 
)

## Train model with the starting parameters
xgb_tune <- train(
  x = trainX,
  y = trainY,
  trControl = tune_control,
  tuneGrid = tune_grid1,
  method = "xgbTree",
  verbose = TRUE
)

#Look at the bestTune
xgb_tune$bestTune

#Another grid to optimize "max_depth" and "min_child_weight" parameters
tune_grid2 <- expand.grid(
  nrounds = seq(from = 50, to = 1000, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = ifelse(xgb_tune$bestTune$max_depth == 3,
                     c(xgb_tune$bestTune$max_depth:5),
                     xgb_tune$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1),
  gamma = 0.1,
  colsample_bytree = 0.5,
  min_child_weight = c(0, 1, 2, 3),
  subsample = 0.5
)
#Train the model to search for best combination
xgb_tune2 <- train(
  x = trainX,
  y = trainY,
  trControl = tune_control,
  tuneGrid = tune_grid2,
  method = "xgbTree",
  verbose = TRUE
)

#Plot the progress
tuneplot(xgb_tune2)
xgb_tune2$bestTune

#Search for the best value of "colsample_bytree" and "subsample"
tune_grid3 <- expand.grid(
  nrounds = seq(from = 50, to = 1000, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = 0.1,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0, 1.2, 1.4),
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = c(0.25, 0.5, 0.75, 1.0)
)

#Train the model with new parameters
xgb_tune3 <- train(
  x = trainX,
  y = trainY,
  trControl = tune_control,
  tuneGrid = tune_grid3,
  method = "xgbTree",
  verbose = TRUE
)

xgb_tune3$bestTune

#Search for the best "gamma" parameter
tune_grid4 <- expand.grid(
  nrounds = seq(from = 50, to = 1000, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0),
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

#Train the model with different gamma values
xgb_tune4 <- train(
  x = trainX,
  y = trainY,
  trControl = tune_control,
  tuneGrid = tune_grid4,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune4)

#Search for the best "eta" values 
tune_grid5 <- expand.grid(
  nrounds = seq(from = 100, to = 10000, by = 100),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = xgb_tune4$bestTune$gamma,
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

#Train to find the best "eta"
xgb_tune5 <- train(
  x = trainX,
  y = trainY,
  trControl = tune_control,
  tuneGrid = tune_grid5,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune5)
xgb_tune5$bestTune

#Final parameters grid after tuning
final_grid <- expand.grid(
  nrounds = xgb_tune5$bestTune$nrounds,
  eta = xgb_tune5$bestTune$eta,
  max_depth = xgb_tune5$bestTune$max_depth,
  gamma = xgb_tune5$bestTune$gamma,
  colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
  min_child_weight = xgb_tune5$bestTune$min_child_weight,
  subsample = xgb_tune5$bestTune$subsample
)



xgb_tune5$bestTune

#Train the final model
train_cont = trainControl(method = "cv", number = 5, allowParallel = TRUE, verboseIter = TRUE)

(xgb_model <- train(
  x = trainX,
  y = trainY,
  trControl = train_cont,
  tuneGrid = final_grid1,
  method = "xgbTree",
  verbose = TRUE
))

#Look at the performance of the model
#The metrics are "RMSE" "R-squared" and MAE



test_xgb = predict(xgb_model, testX)
my_model = RMSE(test_xgb, testY)

random_preds = sample(testY, length(testY))
random = RMSE(random_preds, testY)

always_mean = rep(mean(testY), length(testY))
mean_prediction = RMSE(always_mean, testY)

names = c("my_model", "random", "mean_prediction")
rmse_values = c(my_model,random, mean_prediction)


df = as.data.frame(cbind(names, rmse_values))


ggplot(data=df, aes(x=names, y=rmse_values))+ 
  geom_bar(stat="identity") +
  ggtitle("RMSE comparison with random models")+
  xlab("Models")+
  ylab("RMSE Values")


#Look at the important features for the model
library(gbm)

imp = varImp.gbm(xgb_model)
names = rownames(imp$importance)
names[1:20]
values = imp$importance$Overall
df = data.frame("names" = names[1:20], "values" = values[1:20])
df

p <- ggplot(df, aes(x = values, y = reorder(names, values)))+
  geom_col( width = 0.7)+
  xlab("Importance (Scaled to 100)")+
  ylab("Features")+
  ggtitle("Feature Importance")+
  scale_color_manual(c("blue", "red", "green"))+
  theme_bw()

p

#Top 100 important features
top100 = rownames(imp)[1:100]
class(top100)

#Some detailed plots to see the performance of the model by_cell line
preds = predict(xgb_model, testX)

#Loop to calculate the errors of individiual predictions
errors = c()
for (i in 1:length(testY)){
  x = (preds[i] - testY[i])^2
  errors[i] = x
}

#New data frame to grouped by cell lines
cells_and_error = data.frame(testX[,1:3], errors)

rmse_by_cell= as.data.frame(cells_and_error %>%
  group_by(Cell.line.name) %>%
  summarise(sum_error = sum(errors, na.rm = T)/2,
            n_cell = n()/2,
            rmse = sqrt(sum_error/n_cell)))

attach(rmse_by_cell)
sorted = rmse_by_cell[order(rmse), ]
sorted
error_orders = order(-errors)
error_orders

#Convert numeric cell lines to the cell_names
rmse_by_cell$Cell.line.name = allvalues[rmse_by_cell$Cell.line.name]

types = read.csv("cancer_type.csv")

#Add Cancer types to our new data fame
inds = which(types$Cell.line.name %in% rmse_by_cell$Cell.line.name)
rmse_by_cell$Cancer.Type = as.factor(types[inds,]$GDSC.tissue.descriptor.1)

head(rmse_by_cell)
head(types[inds,])

#Plot the errors for individiual cell lines with the number of observations
g <- ggplot(rmse_by_cell, aes(x = rmse, y = reorder(Cell.line.name, -rmse)))+
  geom_col( width = 0.7)+
  geom_text(aes(y = Cell.line.name, label = n_cell),
            position = position_dodge(width = 1),
            hjust = -0.5)+
  xlab("RMSE Score")+
  ylab("Cell Lines")+
  ggtitle("RMSE Scores grouped by Cell Line")+
  guides(fill=TRUE)+
  theme_bw()
g


# New data frame for grouping by cancer types
asdd = as.data.frame(rmse_by_cell %>% group_by(Cancer.Type) %>% 
                      summarise(Sum_error = sum(sum_error, na.rm = T),
                                N_cell = sum(n_cell),
                                RMSE_type = sqrt(Sum_error/N_cell)))

# Plot the errors by cancer types
c = ggplot(asdd, aes(x = RMSE_type, y = reorder(Cancer.Type, -RMSE_type)))+
  geom_col (width = 0.7)+
  geom_text(aes(y = Cancer.Type, label = N_cell),
            position = position_dodge(width = 1),
            hjust = -0.5)+
  xlab("RMSE Score")+
  ylab("Cancer Types")+
  ggtitle("RMSE Scores grouped by Cancer Types")+
  guides(fill=TRUE)+
  theme_bw()

  
c


