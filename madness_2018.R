#3/12/18
march_past<-read.csv("madness_past1.csv")
march_current<-read.csv("madness_current1.csv")
names(march_past)
names(march_current)
dim(march_current)
str(march_current)

#subset
#train 2011-2017 seasons 
march_past1<-subset(march_past,select=c(3,4,8,11,14,15,16,17,18,19,
                                        20,21,22,23,24,25,26,27,28,29,30,
                                        31,32,33,34,35,36,37,38,39,40,
                                        41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56))

#remove missing values 
march_past2<-na.omit(march_past1)
names(march_past2)
#test 2018 season 
march_current1<-subset(march_current,select=c(2,3,7,10,11,12,13,14,15,16,17,18,19,
                                              20,21,22,23,24,25,26,27,28,29,30,
                                              31,33,34,35,36,37,38,39,40,
                                              
                                              41,42,43,44,45,46,47,48,49,50,51,52,53))
#remove missing values 
march_current2<-na.omit(march_current1)
names(march_current2)
str(march_current2)

library(plyr)
library(dplyr)
march_current3<- mutate_all(march_current2, function(x) as.numeric(as.character(x)))

#random forest model 
rf_model<-randomForest(as.factor(win_home)~.,data=march_past2,importance=TRUE,ntree=2100)
varImpPlot(rf_model)
predict_rf_2018<-predict(rf_model,march_current2,type="prob")
predictions_2018<-data.frame(
  predict_rf_2018,high_seed=march_current$team_fav,low_seed=march_current$team_under,
  seed1=march_current$seed1,seed2=march_current$seed2) 
