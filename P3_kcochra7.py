import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# col_names = ['sepal_length', 'sepal_width','petal_length','petal_width','type']
# data = pd.read_csv("iris.csv", skiprows=1, header =None, names=col_names)
# data.head(10)

col_names = ['Type', 'Alcohol','MalicAcid','Ash','Alcalinity','Magnesium','Phenols','Flavanoids','Nonflavanoid',
             'Proanthocyanins','ColorIntensity', 'Hue','DilutedWines','Proline']
data = pd.read_csv("wine_big.csv", skiprows=1, header =None, names = None)
data.head(10)
print(data)

#NODE CLASS
class Node():
    def __init__(self, feature_index=None, threshhold=None, left=None, right=None, info_gain=None,value=None):
        #Constructor

        #for decision nodes

        self.feature_index = feature_index
        self.threshhold = threshhold
        self.left = left
        self.right = right
        self.info_gain = info_gain


        #for leaf node
        self.value = value

#TREE CLASS
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        #Constructor to build tree

        #initialize the root of the tree
        self.root = None

        #stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth = 0):
        #Recursive function to build the tree
        print("building tree")
        X, Y = dataset[:,:-1], dataset[:,-1]
        print(Y)
        num_samples, num_features = np.shape(X)

        #split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            #Find the best split
            best_split = self.get_best_split(dataset,num_samples,num_features)
            #check if information gain is positive
            #we make sure it isn't 0 because if info_gain = 0 then
            #we already have a pure set. We don't want to split pures

            if best_split["info_gain"]>0:
                #recursive left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                #recursive right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                #return decision node
                return Node(best_split["feature_index"], best_split["threshhold"], 
                left_subtree,right_subtree, best_split["info_gain"])
        #compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        #return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples,num_features):
        #Function to find the best split

        #dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        #loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_threshholds = np.unique(feature_values)
            #loop over all the feature values present in the data
            for threshhold in possible_threshholds:
                #get current split
                dataset_left, dataset_right = self.split(dataset, feature_index,threshhold)
                #check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y,left_y,right_y = dataset[:,-1], dataset_left[:,-1], dataset_right[:,-1]
                    #compute information gain
                    curr_info_gain = self.information_gain(y,left_y,right_y, "entropy")
                    #upate the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshhold"] = threshhold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    def split(self, dataset, feature_index, threshhold):
        #function to split the data

        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshhold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshhold])
        return dataset_left, dataset_right
    #combines all child nodes to parent nodes
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        #function to compute information gain
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        #as you can see there are two different ways to measure the information gained
        #gini index changes the entropy a little differently
        #Its not Gini Index = 1- E p^2i.
        #pi = probabbilty of class i.
        #Since we got rid of the log, we save a lot of time.
        #
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    def gini_index(self,y):
        #Function to compute gini
        #Equation: 1 - E pi^2
        #ex: 1-(0^2+1^2)= 0

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y==cls])/len(y)
            #double astrix in python is power pow(2) 
            gini += p_cls**2
            
        
        return (1 - gini)
    def entropy(self, y):
        #Function to compute entropy
        #Equation: E-pi * log(pi)
        #ex: -1log(1)-0log(0) = 0

        class_labels = np.unique(y)

        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y==cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    def print_tree(self, tree = None, indent = " "):
        #Function to print tree

        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=",tree.threshhold, "?", tree.info_gain)
            print("%sleft:"% (indent), end="")
            print("%sright:" %(indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        #Function to train tree
        dataset = np.concatenate((X,Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        #function to predict new data set

        predictions = [self.make_prediction(x,self.root)for x in X]
        return predictions
    def make_prediction(self, x, tree):
        #function to predict a single data point
        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <=tree.threshhold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x,tree.right)


#Main Function
if __name__=='__main__':
    #from the -13 column, it means we start from the last set in the array
    #so in our wine database, we start from Alcohol. then the -1  lets the code know
    #we are starting from the right side of the array. Aka the last colum in the array 

    #I dont think -1 does this. -1 is skipping my last column. 
    #Maybe I can make a copy of the data set, delete the column then set it to X. 
    X = data.iloc[:,-13:-1].values
    Y = data.iloc[:,0].values.reshape(-1,1)
    print(Y)
    print(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =.1, random_state = 41)

    classifier = DecisionTreeClassifier(min_samples_split =3, max_depth=3)
    classifier.fit(X_train,Y_train)
    classifier.print_tree()


    Y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    print("Accuaracy score: " + str(accuracy_score(Y_test, Y_pred)))
    print("\nConfusion Matrix: \n" +str(confusion_matrix(Y_test, Y_pred))) 
    print("\nClassification report: \n" + str(classification_report(Y_test, Y_pred)))
