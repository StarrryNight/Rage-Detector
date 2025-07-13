import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")

#Training multiclass classification model
X = df.drop('class', axis=1)
y = df['class']

#split data
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 69)

print(y_test)