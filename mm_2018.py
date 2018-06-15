import pandas as pd 
import numpy as np 

#3/3/18 
#import tournament data
mm_18=pd.read_csv("march_madness_2018.csv")

#subset the data
mm_ncaa=np.column_stack([mm_18['School'].str.contains(r"NCAA") for col in mm_18])
type(mm_ncaa)
mm_ncaa.shape 
mm_ncaa1=pd.DataFrame(data=mm_ncaa[1:,1:],
index=mm_ncaa[1:,0],columns=mm_ncaa[0,1:]) 
type(mm_ncaa1)
mm_ncaa1.shape 
ncaa=[s for s in mm_18['School'] if "NCAA" in s]
type(ncaa) #list

#search teams that made the tourney 
mm_18['NCAA']=mm_18['School'].str.contains('NCAA')
ncaa_bound=mm_18[(mm_18['NCAA']==True)] #603 team made it? 07-08-16-17 
ncaa_bound.head(20) 
ncaa_bound.shape 

#assign new value by season 
ncaa_bound['Year']=ncaa_bound['Season']
ncaa_bound.Year[ncaa_bound.Year=="2011-12"]=2012 
ncaa_bound.Year[ncaa_bound.Year=="2007-08"]=2008 
ncaa_bound.Year[ncaa_bound.Year=="2008-09"]=2009
ncaa_bound.Year[ncaa_bound.Year=="2009-10"]=2010
ncaa_bound.Year[ncaa_bound.Year=="2010-11"]=2011
ncaa_bound.Year[ncaa_bound.Year=="2011-12"]=2012
ncaa_bound.Year[ncaa_bound.Year=="2012-13"]=2013
ncaa_bound.Year[ncaa_bound.Year=="2013-14"]=2014
ncaa_bound.Year[ncaa_bound.Year=="2014-15"]=2015
ncaa_bound.Year[ncaa_bound.Year=="2015-16"]=2016
ncaa_bound.Year[ncaa_bound.Year=="2016-17"]=2017 


#distance regional, star player (16 pts+?), injuries?, conference tournament semi/final app, total wins, 
#non-power wins 
ncaa_bound.to_csv('march_madness_18.csv')

#split seed from each team if seed 1 < seed 2 assign 1 else assign 0 
ncaa_s=pd.read_csv("madness_matchups.csv")
ncaa_s.info() 
ncaa_s['School'].head(4)

#new dataframe 
new_df=pd.DataFrame(ncaa_s.School.str.split(' ',1).tolist(),columns=['seed1','School'])
m18=pd.read_csv("march_2017x.csv")
m18.info() 

#create a unique id 
m18['id'] = range(1, len(m18) + 1)

m18x=m18[["team_fav","win_home","W-L%_f","SRS_f","SOS_f","FG%_f","3P%_f",
"FTA/G_f","ORB/G_f","TRB/G_f","AST/G_f","team_under",
"W-L%_u","SRS_u","SOS_u","FG%_u","3P%_u",
"FTA/G_u","ORB/G_u","TRB/G_u","AST/G_u",
"STL/G_f",
"STL/G_u","BLK/G_f","BLK/G_f","TOV/G_f","TOV/G_u"]]
m18x.to_csv("m18x.csv")

m18xx=m18x[["win_home","W-L%_f","SRS_f","SOS_f","FG%_f","3P%_f",
"FTA/G_f","ORB/G_f","TRB/G_f","AST/G_f",
"W-L%_u","SRS_u","SOS_u","FG%_u","3P%_u",
"FTA/G_u","ORB/G_u","TRB/G_u","AST/G_u",
"STL/G_f",
"STL/G_u","BLK/G_f","BLK/G_f","TOV/G_f","TOV/G_u"]]

#logistic regression model 
X=m18xx[["W-L%_f","SRS_f","SOS_f","FG%_f","3P%_f",
"FTA/G_f","ORB/G_f","TRB/G_f","AST/G_f",
"W-L%_u","SRS_u","SOS_u","FG%_u","3P%_u",
"FTA/G_u","ORB/G_u","TRB/G_u","AST/G_u","STL/G_f",
"STL/G_u","BLK/G_f","BLK/G_f","TOV/G_f","TOV/G_u"]]
y=m18xx["win_home"]

#train and test sets 
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

lr_model_scale=LogisticRegression()
lr_model_scale.fit(X_train,y_train)
predict_lr=lr_model_scale.predict(X_test)
accuracy_score(y_test,predict_lr) 
from sklearn.metrics import classification_report #source: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
print(classification_report(y_test, predict_lr))
#convert logistic regression 
df_lr=pd.DataFrame(predict_lr)

#convert to csv file 
y_test.to_csv("y_test.csv")
df_lr.to_csv("df_lr.csv")

#merge csv files 
x1=pd.read_csv("m18x.csv")
x2=pd.read_csv("pred_march.csv")
x3=pd.merge(x1,x2,on='id')
x4=x3[["team_fav","team_under","win_home","pred"]]

#*************************************************************
##3/8/18 new march madness prediction model 
march=pd.read_csv("madness_sanluis2.csv")

#subset the data--features and target
x1=march.iloc[:,6:8]
x2=march.iloc[:,19:62]
X=pd.concat([x1,x2],axis=1) 
X.shape #331,45
y=march.iloc[:,18]

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#model 1: logistic regression (i.)
lr_model=LogisticRegression()
lr_model.fit(X_train,y_train)
predict=lr_model.predict(X_test)
accuracy_score(y_test,predict) #78.3% accuracy 

#model 2: logistic regression (ii.)
import statsmodels.api as sm 
logit_model=sm.Logit(y,X)
result=logit_model.fit() 
result.summary()

#model 3: random forest (i.) 
from sklearn.ensemble import RandomForestRegressor 
#create the model? (dublin, cork?)
rf=RandomForestRegressor(n_estimators=1000,random_state=64)
#train the model por training data?
rf.fit(y_test,X_train)

#model 4: random forest (ii.)
from sklearn.ensemble import RandomForestClassifier

X1=X.values
y1=y.values 
#test train data sets?
X_train, X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.23,random_state=0)

#train the random forest classifier
clf_rf=RandomForestClassifier(n_jobs=10,random_state=89)
clf_rf.fit(X_train,y_train)
classifier_march=clf_rf.predict(X_test) 

#predicted probabilities (from the random forest model) 
prob_march=clf_rf.predict_proba(X_test)
prob_march 

#confusion matrix
pd.crosstab(y_test,classifier_march) #59.7% accuracy 

### stochastic gradient descent 
#source: http://scikit-learn.org/stable/modules/sgd.html
#sgd-- #1. advanced efficiency, easy to make #2. non-ad--sensitive to feature scaling 

from sklearn.linear_model import SGDClassifier 
X_train1.shape 
y_train.shape 
clf_sgd=SGDClassifier(loss="log",penalty="l2")
clf_sgd.fit(X_train1,y_train)
predict_win=clf_sgd.predict_proba(X_test1)
predict_win 

#another SGD Model 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

clf_sgd1=SGDClassifier(loss='log')
calibrated_clf_sgd1=CalibratedClassifierCV(clf_sgd1,cv=6,method='sigmoid')
calibrated_clf_sgd1.fit(X_train1,y_train)
predicted = calibrated_clf_sgd1.predict(X_test1)

#ii. nearest neighbors source: http://scikit-learn.org/stable/modules/neighbors.html
X_fav=X_train[:,10:26] 
from sklearn.neighbors import NearestNeighbors
import numpy as np 
vecinos=NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(X_fav)
distances,indices=vecinos.kneighbors(X_fav)
indices 
distances 

#outlier analysis- detect outliers
#computes z-score for each column 
from scipy import stats 
X1=march[(np.abs(stats.zscore(X))<3).all(axis=1)]

q=march["3PA/G_u"].quantile(0.90)
q #90% quantile= 23.11
#which under dog teams should high 3PA >90%?
high_3pa_under=march[march["3PA/G_u"]>q]
high_3pa_under[["team_under","Year.1","win_home"]]
high_3pa_under.groupby(['win_home']).size() #24.2% win pct :(? 
high_3pa_under["win_home"].value_counts()
import seaborn as sns 
sns.countplot(x='win_home',data='high_3pa_under',palette='hls')
plt.show()
march.groupby('team_under').mean()

#histogram-- 3PA under 
march["3PA/G_u"].mean() #19.3 

march["3PA/G_u"].hist()
plt.title('Lower Seed 3PA per game')
plt.xlabel("3PA/G")
plt.ylabel('Frequency')
plt.show()

#histogram/ 3PA favorite
march["3PA/G_f"].mean() #19.04

#3PA and winning
pd.crosstab(march["3PA/G_f"],march["win_home"]).plot(kind='bar')
plt.title('3PA Higher Seed')
plt.xlabel('Higher Seed Win')
plt.ylabel('Frequency')
plt.show() 


























