# set working directory
setwd("/Users/ayan/Desktop/BU/Spring 2023/CS699_project")

# import data file
weather_data <- read.csv("weather_data.csv")

# These columns can be used for classification:
# AWND, PRCP, PRCP_ATTRIBUTES, SNOW_ATTRIBUTES, TAVG, TMAX, TMAX_ATTRIBUTES == TMIN_ATTRIBUTES (can only use 1),
# TMIN, WDF2, WDF5, WSF2, WSF5, WT01, WT02, WT03, WT04, WT05, WT06, WT08, WT09, DATE ???

# these are the features mentioned above
features <- c(
  'AWND', 
  'PRCP', 
  'PRCP_ATTRIBUTES', 
  'SNOW_ATTRIBUTES', 
  'TAVG', 
  'TMAX', 
  'TMAX_ATTRIBUTES', 
  'TMIN', 
  'WDF2', 
  'WDF5', 
  'WSF2', 
  'WSF5', 
  'WT01', 
  'WT02', 
  'WT03', 
  'WT04', 
  'WT05', 
  'WT06', 
  'WT08', 
  'WT09',
  'weather_condition'
  )

# we take only the columns/features we need
weather_data_subset <- weather_data[features]

# make result column a factor
weather_data_subset$weather_condition <- as.factor(weather_data_subset$weather_condition)

# change categorical variables to numeric
weather_data_subset$PRCP_ATTRIBUTES <- factor(
  weather_data_subset$PRCP_ATTRIBUTES, 
  levels = c(',,D,2400', ',,W,2400', 'T,,D,2400', 'T,,W,2400'),
  labels = c(2, 4, 1, 3)
  )

weather_data_subset$SNOW_ATTRIBUTES <- factor(
  weather_data_subset$SNOW_ATTRIBUTES, 
  levels = c(',,D', ',,W', 'T,,D', 'T,,W'),
  labels = c(2, 4, 1, 3)
)

weather_data_subset$TMAX_ATTRIBUTES <- factor(
  weather_data_subset$TMAX_ATTRIBUTES, 
  levels = c(',,D', ',,W'),
  labels = c(1, 2)
)

# fill NA values with 0
weather_data_subset$WT01[is.na(weather_data_subset$WT01)] <- 0
weather_data_subset$WT02[is.na(weather_data_subset$WT02)] <- 0
weather_data_subset$WT03[is.na(weather_data_subset$WT03)] <- 0
weather_data_subset$WT04[is.na(weather_data_subset$WT04)] <- 0
weather_data_subset$WT05[is.na(weather_data_subset$WT05)] <- 0
weather_data_subset$WT06[is.na(weather_data_subset$WT06)] <- 0
weather_data_subset$WT08[is.na(weather_data_subset$WT08)] <- 0
weather_data_subset$WT09[is.na(weather_data_subset$WT09)] <- 0
weather_data_subset$WDF5[is.na(weather_data_subset$WDF5)] <- 0
weather_data_subset$WSF5[is.na(weather_data_subset$WSF5)] <- 0

# scale numeric column to mean = 0 and sd = 1
weather_data_subset$AWND <- scale(weather_data_subset$AWND)
weather_data_subset$PRCP <- scale(weather_data_subset$PRCP)
weather_data_subset$TAVG <- scale(weather_data_subset$TAVG)
weather_data_subset$TMAX <- scale(weather_data_subset$TMAX)
weather_data_subset$TMIN <- scale(weather_data_subset$TMIN)
weather_data_subset$WDF2 <- scale(weather_data_subset$WDF2)
weather_data_subset$WDF5 <- scale(weather_data_subset$WDF5)
weather_data_subset$WSF2 <- scale(weather_data_subset$WSF2)
weather_data_subset$WSF5 <- scale(weather_data_subset$WSF5)

# feature selection 
#1
install.packages("party")
library(party)
install.packages("varImp")
library(varImp)

cf1 <- cforest(weather_condition ~ . , data= weather_data_subset, control=cforest_unbiased(mtry=2,ntree=50))
sort(varimpAUC(cf1))

#2
library(caret)

control <- trainControl(method="repeatedcv", number=10, repeats=10, sampling="down")
model <- train(weather_condition~., data=weather_data_subset, method="lvq", preProcess="scale", trControl=control)

importance <- varImp(model, scale=FALSE)
plot(importance)

#3
install.packages("Boruta")
library(Boruta)

boruta_output <- Boruta(weather_condition ~ ., data=weather_data_subset, doTrace=0)
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")

#4
set.seed(17)
rPartMod <- train(weather_condition ~ ., data=weather_data_subset, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)

set.seed(17)
rPartMod <- train(weather_condition ~ ., data=weather_data_subset, method="RRF")
rpartImp <- varImp(rPartMod)
print(rpartImp)

#5















