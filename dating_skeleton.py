import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("profiles.csv")
#print(df.status.value_counts())

#creating histogram of heights
df["height_cm"] = df.height.apply(lambda height: height * 2.54)
 
plt.hist(df.height_cm, bins=30)
plt.xlabel("Heights")
plt.ylabel("Frequency")
plt.xlim(100, 220)
plt.savefig("heights.png")


#boxplot of income

income_df= df[df["income"] > 0]
plt.boxplot(income_df.income)
plt.ylabel("Income")
plt.savefig("income.png")


#Setting up the workspace for calculations
df["status_numerical"] = df.status.map({"unknown":0, "available": 0, "single":1, "seeing someone":2, "married":3})
df["drinks_numerical"] = df.drinks.map({"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5})
df["drugs_numerical"] = df.drugs.map({"never":0, "sometimes":1, "often":2})
df["sex_numerical"] = df.sex.map({"m":0, "f":1})


essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

all_essays = df[essay_cols].replace(np.nan, '', regex=True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df["essay_len"] = all_essays.apply(lambda x: len(x))

#calculating average word length in the essays  
def avg_word_length(essay):
    words = essay.split()
    if len(words) == 0:
        return 0
    return sum(len(word) for word in words) / len(words)
df["avg_word_length"] = all_essays.apply(avg_word_length)

#mapping the seriousness of one's religion to a number
def seriousness_score(religion):
    # Check if religion is a string
    if isinstance(religion, str):
        if 'not too serious' in religion:
            return 1
        elif 'laughing about it' in religion:
            return 2
        elif 'somewhat serious' in religion:
            return 4
        elif 'very serious' in religion:
            return 5
        else:
            return 3 
df["religion_numerical"] = df.religion.apply(seriousness_score)

#scaling data
feature_data = df[['status_numerical', 'drugs_numerical', 'drinks_numerical', 'essay_len', 'avg_word_length', "religion_numerical", "sex_numerical", "income"]]

#non-NA data that can be used for analysis
classif_data_1 = feature_data.dropna(subset=["religion_numerical", "drinks_numerical", "drugs_numerical"])

x = classif_data_1.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x) 
classif_data_scaled = pd.DataFrame(x_scaled, columns=feature_data.columns)

#splitting data
train_data_1, test_data_1, train_labels_1, test_labels_1 = train_test_split(classif_data_scaled[["drinks_numerical", "drugs_numerical"]], classif_data_1["religion_numerical"], random_state=33)

#K Nearest Neighbors
classif_1 = KNeighborsClassifier()
classif_1.fit(train_data_1, train_labels_1)

#Support Vector Machines
classif_2 = SVC()
classif_2.fit(train_data_1, train_labels_1)

#Evaluation
print("KNeighborsClassifer score: ", classif_1.score(test_data_1, test_labels_1))
print("SVC score: ", classif_2.score(test_data_1, test_labels_1))
#print("Confusion matrix of SVC: \n", confusion_matrix(test_labels_1, classif_2.predict(test_data_1)))

#checking on religion variable
plt.hist(x=classif_data_1["religion_numerical"])
plt.savefig("religion.png")

#data for regression models
regression_data = feature_data[["avg_word_length", "drinks_numerical", "drugs_numerical", "income"]][feature_data["income"]> 0].dropna(subset = ["avg_word_length", "drinks_numerical", "drugs_numerical", "income"])

train_set_2, test_set_2 = train_test_split(regression_data, test_size=0.2)

#multiple regression model
regressor = linear_model.LinearRegression()
regressor.fit(train_set_2[["avg_word_length", "drinks_numerical", "drugs_numerical"]], train_set_2["income"])
print("Coeficients of multiple linear regression: ", regressor.coef_, "\n r squared: ", regressor.score(test_set_2[["avg_word_length", "drinks_numerical", "drugs_numerical"]], test_set_2["income"]))


#K Nearest Neighbors Regression - plotting the accuracy for possible k values to see the best accuracy achievable
k_list = range(1,101)
accuracies = []
for k in range(1, 101):
  regr = KNeighborsRegressor(n_neighbors = k)
  regr.fit(train_set_2[["avg_word_length", "drinks_numerical", "drugs_numerical"]], train_set_2["income"])
  accuracies.append(regr.score(test_set_2[["avg_word_length", "drinks_numerical", "drugs_numerical"]], test_set_2["income"]))
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Income regressor")
plt.savefig("Income KNeighborsRegressor.png")
