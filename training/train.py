import pandas as pd
from sklearn.model_selection import train_test_split

#pipeline stuff
from sklearn.pipeline import make_pipeline

#preprocessing function
from sklearn.preprocessing import StandardScaler

#classification algos
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Evaluation and Serialization of models
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv("training/data.csv")

#Training multiclass classification model
X = df.drop('class', axis=1)
y = df['class']

#split data
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 69)

#setup pipelines dictionary for training
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo]= model


#test "cycle" (well im using scikit learn)
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))


#saving with pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)