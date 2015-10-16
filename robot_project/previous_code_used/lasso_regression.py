'''
This was copied and pasted from a couple of files so
there are going to be errors in this code! Meant as a reference
only.

Also, this code is not annoted that much because I wrote it for myself.
I will add annotations later. 

--Chris
'''
import linear_regression_secondattempt as lr
import numpy as np
import re
from sklearn import linear_model
import random
import time
import zipfile
import pickle

def save_cv_lasso(alpha_used, cv, baxter_pos_csv_file_name, cv_csv_file_name):
    data_csv_file = file(baxter_pos_csv_file_name, 'r')
    header_line = data_csv_file.readline().strip()
    headers = header_line.split(',')
    data_csv_file.close()

    number_of_entries_per_header = len(cv) / len(headers)
    print

    datastring = ''
    for i in xrange(len(cv)):
        if(i % number_of_entries_per_header == 0):
            datastring += headers[i/number_of_entries_per_header]
            datastring += ','

        datastring += str(cv[i])
        if(i % number_of_entries_per_header == number_of_entries_per_header - 1):
            datastring += '\n'
        else:
            datastring += ','

    cv_csv_file = file(cv_csv_file_name, 'a')
    cv_csv_file.write('alpha=' + str(alpha_used) + '\n')
    cv_csv_file.write('end_effector_parent_link,trans_x,trans_y,trans_z,' +
                      'orien_x,orien_y,orien_z,orien_w' +
                      '\n')
    cv_csv_file.write(datastring)
    cv_csv_file.write('\n')
    cv_csv_file.close()

def extract_data_from_file(data_file_name):
    data_file = open(data_file_name, 'r')
    data = []
    
    # skips header
    line = data_file.readline().strip()
    
    while line:
        line = data_file.readline().strip()
        line = line.replace(';', ',')
        split_line = line.split(',')
        sub_data = []
        
        for i in split_line:
            try:
                sub_data.append(float(i))
            except ValueError:
                continue
    
        if(len(sub_data) > 0):
            data.append(sub_data)

    data_file.close()
    
    return data

def get_data(baxter_pos_csv_file_name, human_csv_file_name):
    baxter_data = extract_data_from_file(baxter_pos_csv_file_name)
    human_data = extract_data_from_file(human_csv_file_name)
    
    return human_data, baxter_data

def main():
    baxter_pos_csv_file_name = 'data/extracted_baxterdata.csv'
    human_csv_file_name = 'data/extracted_humandata.csv'
    cv_csv_file_name = 'data/cv_lasso_tuning.csv'
    lasso_alphas = [0.03] # add to list to consider multple alphas with alpha regression


    human_data, baxter_data = lr.get_data(baxter_pos_csv_file_name,
                                          human_csv_file_name)

    X = np.array(human_data)
    Y = np.array(baxter_data)

    # leave one out cross validation
    for alpha in lasso_alphas:
        MSI = [0]*len(Y[0])
        CV = [0]*len(Y[0])

        print "*Starting alpha=" + str(alpha) + " Calculations*"
        time1 = time.time()

        for i in xrange(len(X)):
            X_training = np.append(X[:i], X[i+1:])
            Y_training = np.append(Y[:i], Y[i+1:])

            X_oneout = X[i]
            Y_oneout_truevalues = Y[i]

            regr = linear_model.Lasso(alpha=alpha)
            regr.fit(X, Y)

            Y_oneout_predicted_values = regr.predict(X_oneout)
            for i in xrange(len(MSI)):
                MSI[i] += (Y_oneout_truevalues[i] - Y_oneout_predicted_values[0][i])**2

        n = len(Y)
        for i in xrange(len(CV)):
            CV[i] = MSI[i] / n
        time2 = time.time()
        print "*Complete, time took is " + str(time2-time1) +" seconds*"
        #save_cv_lasso(alpha, CV, baxter_pos_csv_file_name, cv_csv_file_name)

    save_file = zipfile.ZipFile('joints_lasso.model', 'w')

    save_file.writestr('regressions.p', pickle.dumps(regr))

    save_file.close()

if __name__ == '__main__':
    main()
