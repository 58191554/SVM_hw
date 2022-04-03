from doctest import testsource
import re
import numpy as np
from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn import svm, datasets

class SVM(object):

    def __init__(self, training_dataset_, test_dataset_):
        self.training_dataset = training_dataset_
        self.test_dataset = test_dataset_
        self.classes = {}
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        self.support_indecies = None
        self.train_errors = None
        self.val_errors = None
        self.train_loss = None
        self.val_loss = None
        
    def read_data(self):
        f = open(self.training_dataset, 'r')
        rows = list(re.split(' ', row) for row in re.split('\n', f.read())[:-1])
        names, self.Y_train = np.unique(list(row[-1] for row in rows), return_inverse=True)
        self.X_train = np.empty((0,4), float)
        f.close()
        for row in rows:
            self.X_train = np.append(self.X_train, np.array([np.array(row[:-1]).astype(float)]), axis = 0)
        f = open(self.test_dataset, 'r')
        f.close()
        
        f = open(self.test_dataset, 'r')
        rows = list(re.split(' ', row) for row in re.split('\n', f.read())[:-1])
        names, self.Y_test = np.unique(list(row[-1] for row in rows), return_inverse=True)
        self.X_test = np.empty((0,4), float)
        f.close()
        for row in rows:
            self.X_test = np.append(self.X_test, np.array([np.array(row[:-1]).astype(float)]), axis = 0)
        
    def get_active_class(self,active_num: int):
        active_train_ls = []
        active_test_ls = []

        for i in self.Y_train:
            if i == active_num:
                active_train_ls.append(0)
            else:
                active_train_ls.append(1)
        for i in self.Y_test:
            if i == active_num:
                active_test_ls.append(0)
            else:
                active_test_ls.append(1)

        return active_train_ls, active_test_ls
        
    def SVM(self, C_value = 10**5, knl = 'linear'):

        svm_lin_model = svm.SVC(C = C_value,kernel=knl, gamma=10,decision_function_shape="ovo")
        svm_lin_model.fit(self.X_train, self.Y_train.ravel())
        train_score = svm_lin_model.score(self.X_train, self.Y_train)
        print("train_score\n", train_score)
        test_score = svm_lin_model.score(self.X_test, self.Y_test)
        print("test_score\n", test_score)

        setosa = svm.SVC(C = C_value,kernel=knl, gamma=10,decision_function_shape="ovo")
        setosa.fit(self.X_train, self.get_active_class(0)[0])
        setosa_w = setosa.coef_
        setosa_b =  setosa.intercept_
        setosa_support_vectors = setosa.support_vectors_
        setosa_result = [setosa_w,setosa_b, setosa_support_vectors]

        versicolor = svm.SVC(C = C_value,kernel=knl, gamma=10,decision_function_shape="ovo")
        versicolor.fit(self.X_train, self.get_active_class(1)[0])
        versicolor_w = versicolor.coef_
        versicolor_b = versicolor.intercept_
        versicolor_support_vectors = versicolor.support_vectors_
        versicolor_result = [versicolor_w, versicolor_b, versicolor_support_vectors]

        virginica = svm.SVC(C = C_value,kernel=knl, gamma=10,decision_function_shape="ovo")
        virginica.fit(self.X_train, self.get_active_class(2)[0])
        virginica_w = virginica.coef_
        virginica_b = virginica.intercept_
        virginica_support_vectors = virginica.support_vectors_
        virginica_result = [virginica_w, virginica_b, virginica_support_vectors]

        train_loss = 1-train_score
        test_loss = 1 - test_score

        result = [train_loss, test_loss, setosa_result, versicolor_result, virginica_result]

        return result

    def SVM_kernel_poly2(C):
        return self.SVM(1, "poly")
  
    def SVM_kernel_poly3(C):
           
        # return train_loss, test_loss, support_vectors
        pass
    
    def SVM_kernel_rbf(C):
        
        # return train_loss, test_loss, support_vectors
        pass
    
    def SVM_kernel_sigmoid(C):
        # return train_loss, test_loss, support_vectors
        pass

    def output_txt(self,file_name:str, result:list):
        train_loss = result[0]
        test_loss  = result[1]

        setosa_result = result[2]
        versicolor_result = result[3]
        virginica_result = result[4]

        with open(file_name, "w") as fw:
            fw.write("training_error\n")
            fw.write(str(train_loss)+"\n")
            fw.write("testing_error\n")
            fw.write(str(test_loss)+"\n")
# setosa_result
            fw.write("w_of_setosa\n")
            for i in range(len(setosa_result[0][0])):
                if i != len(setosa_result[0][0])-1:
                    fw.write(str(setosa_result[0][0][i]) + ",")
                else:
                    fw.write(str(setosa_result[0][0][i])+"\n")
            fw.write("b_of_setosa\n")
            
            fw.write("support_vector_indices_of_setosa\n")          
            for i in range(len(setosa_result[2])):
                for j in range(len(setosa_result[2][0])):
                    if j != len(setosa_result[2][0])-1:
                        fw.writelines(str(setosa_result[2][i][j]) + ",")
                    else:
                        fw.writelines(str(setosa_result[2][i][j]) + "\n")
#versicolor_result
            fw.write("w_of_versicolor\n")
            for i in range(len(versicolor_result[0][0])):
                if i != len(versicolor_result[0][0])-1:
                    fw.write(str(versicolor_result[0][0][i]) + ",")
                else:
                    fw.write(str(versicolor_result[0][0][i])+"\n")
            fw.write("b_of_versicolor\n")
            
            fw.write("support_vector_indices_of_versicolor\n")          
            for i in range(len(versicolor_result[2])):
                for j in range(len(versicolor_result[2][0])):
                    if j != len(versicolor_result[2][0])-1:
                        fw.writelines(str(versicolor_result[2][i][j]) + ",")
                    else:
                        fw.writelines(str(versicolor_result[2][i][j]) + "\n")
# virginica_result
            fw.write("w_of_virginica\n")
            for i in range(len(virginica_result[0][0])):
                if i != len(virginica_result[0][0])-1:
                    fw.write(str(virginica_result[0][0][i]) + ",")
                else:
                    fw.write(str(virginica_result[0][0][i])+"\n")
            fw.write("b_of_virginica\n")
            
            fw.write("support_vector_indices_of_virginica\n")          
            for i in range(len(virginica_result[2])):
                for j in range(len(virginica_result[2][0])):
                    if j != len(virginica_result[2][0])-1:
                        fw.writelines(str(virginica_result[2][i][j]) + ",")
                    else:
                        fw.writelines(str(virginica_result[2][i][j]) + "\n")

if __name__ == "__main__":
    mySVM = SVM("train.txt", "test.txt")
    mySVM.read_data()


# task 1
    linear_result = mySVM.SVM()
    mySVM.output_txt("SVM_linear.txt", linear_result)
#task 2
    slack_C_ls = [mySVM.SVM(C_value=0.1*(t+1)) for t in range(10)]
    for t in range(10):
        mySVM.output_txt("SVM_slack_0."+str(t+1)+".txt",  slack_C_ls[t])
        print(t+1,"complete!")
#task 3


