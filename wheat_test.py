import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

'''
important variables:
    dict_test_healthy: store the test data from day-2 to day-12
    dict_test_infected:

'''



# every day-file is divided into spectra files and calibration files
def file_divided(files):
    spec_listdir = []
    cal_listdir = []
    # if the start of file name is 'cal', then be divide to calibration files
    for f in files:
        if re.match('cal', f):
            cal_listdir.append(f)
        else:
            spec_listdir.append(f)
    return [spec_listdir, cal_listdir]

# every spectra file is divided into two list: wavelength and intensity
def devide_wave_intensity(file_name, absolute_path):
    full_path = os.path.join(absolute_path, file_name)  # the completed path
    with open(full_path) as file_obj:
        lines = file_obj.readlines()  # readline读入后就是存在一个List中
        # print(lines)
        del lines[0:4]  # delete the notation
        wavelength = []
        intensity = []
        for line in lines:
            wave = float(line.split(',', 2)[0].rstrip())  # Split by , first column
            light = float(line.split(',', 2)[1].strip())  # second column
            wavelength.append(wave)
            intensity.append(light)
    return [wavelength, intensity]

# to calculate the average intensity of calibration files in one day
def average_intensity(file_listdir, absolute_path):
    '''
    # to calculate the length of one spectra file
    
    full_path = os.path.join(absolute_path, file_listdir[0])
    with open(full_path) as file_obj:
        lines = file_obj.read()
    length = len(lines) - 4
    '''
    length = 3648
    average = []
    for w in range(length):
        sum_cal_intensity = 0
        for cal in file_listdir:
            wave, inten = devide_wave_intensity(cal, absolute_path)
            sum_cal_intensity += inten[w]
        mean_inten = sum_cal_intensity / len(file_listdir)
        average.append(mean_inten)
    return average


dict_test_healthy = {}
dict_test_infected = {}

fg = plt.figure()
ax = fg.add_subplot(1,1,1)
color_list = ['black', 'dimgray', 'darkorange', 'lightcoral', 'red', 'olive', 'lawngreen', 'darkgreen', 'blue', 'violet']
day_list = ['day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7', 'day8', 'day9', 'day10']
loc_list = [(0.8,0.02), (0.8,0.1), (0.8,0.18),(0.8,0.26),(0.8,0.34),(0.8,0.42),(0.8,0.5),(0.8,0.58),(0.8,0.66),(0.8,0.74)]
for day in range(2, 12):
    print('day %d begins'%day)
    healthy_path = os.path.abspath(r'D:\\Durham\\Wheat_dataset\\Wheat_dataset\\%d dpi\\healthy' % day)
    inf_path = os.path.abspath(r'D:\\Durham\\Wheat_dataset\\Wheat_dataset\\%d dpi\\inf' % day)
    healthy_files = os.listdir(healthy_path)
    inf_files = os.listdir(inf_path)

    healthy_spec_listdir, healthy_cal_listdir = file_divided(healthy_files)
    inf_spec_listdir, inf_cal_listdir = file_divided(inf_files)

    '''
    # Calculate the average calibration value of healthy and infected plants seperately
    ave_healthy_cal = average_intensity(healthy_cal_listdir, healthy_path)
    fileobject = open('D:\\Durham\\code\\code\\average_cal\\ave_healthy_intensity_%d.txt' % day, 'w')
    for i in ave_healthy_cal:
        fileobject.write(str(i))
        fileobject.write('\n')
    fileobject.close()

    ave_infected_cal = average_intensity(inf_cal_listdir, inf_path)
    fileobject = open('D:\\Durham\\code\\code\\average_cal\\ave_infected_intensity_%d.txt' % day, 'w')
    for i in ave_infected_cal:
        fileobject.write(str(i))
        fileobject.write('\n')
    fileobject.close()
    '''


    # read in the average value
    with open('D:\\Durham\\code\\code\\average_cal\\ave_healthy_intensity_%d.txt' % day) as file_obj:
        lines = file_obj.readlines()
    ave_healthy_cal = []
    for line in lines:
        light = float(line.rstrip())
        ave_healthy_cal.append(light)
    print(len(ave_healthy_cal))

    with open('D:\\Durham\\code\\code\\average_cal\\ave_infected_intensity_%d.txt' % day) as file_obj:
        lines = file_obj.readlines()
    ave_infected_cal = []
    for line in lines:
        light = float(line.rstrip())
        ave_infected_cal.append(light)
    print(len(ave_infected_cal))

    

    # resampling (only for wavelength)
    wavelength, intensity = devide_wave_intensity('1.mspec', healthy_path)
    window = 2
    s_wave = 396 # the start value of wavelength
    key = s_wave + window
    keys = [key]
    interval_num = []
    num = 0  # 每环节计数器
    for w in wavelength:
        if w < key:
            num += 1
        else:
            interval_num.append(num)
            num = 1
            key = key + window
            keys.append(key)
    print('final_key = %d' % key)
    add_interval = sum(interval_num)
    interval_num.append(num)
    print(interval_num)
    wavelength_dataset = range(396, key, window)

    # create an empty numpy list
    num_points = int((key-s_wave) / window)
    # num_points = 200
    healthy_dataset = np.zeros(shape=(100, num_points))
    infected_dataset = np.zeros(shape=(100, num_points))
    healthy_dataset_nospline = np.zeros(shape=(100, num_points))
    infected_dataset_nospline = np.zeros(shape=(100, num_points))
    # total_dataset = np.zeros(shape=(200, len(interval_num)))

    # ########################
    # healthy plants
    ##########################
    for row in range(len(healthy_spec_listdir)):
        spec = healthy_spec_listdir[row]
        wavelength, intensity = devide_wave_intensity(spec, healthy_path)

        # do calibration
        healthy_calibrated_intensity = []
        for i in range(len(intensity)):
            n = intensity[i] / ave_healthy_cal[i]
            healthy_calibrated_intensity.append(n)

        # reduce redundancy
        resample_healthy_intensity = []
        last = 0
        for i in interval_num:
            #
            resample_healthy_intensity.append(sum(healthy_calibrated_intensity[last:(last + i)]) / i)
            last = last + i

        if resample_healthy_intensity[0] > 1:
            resample_healthy_intensity[0] = 1
        if resample_healthy_intensity[0] < 0:
            resample_healthy_intensity[0] = 0
        for j in range(1, len(resample_healthy_intensity)):
            if resample_healthy_intensity[j] > 1 or resample_healthy_intensity[j] < 0:
                replacement1 = sum(resample_healthy_intensity[0:j]) / j
                # replacement2 = sum()
                resample_healthy_intensity[j] = replacement1
        # print(resample_healthy_intensity)
        healthy_dataset_nospline[row] = resample_healthy_intensity

        # spline
        func = interpolate.UnivariateSpline(wavelength_dataset, resample_healthy_intensity, s=3)
        xnew = np.arange(s_wave, key, window)
        hea_ynew = func(xnew)
        healthy_dataset[row] = hea_ynew
    interval_seg = 20 / window
    split_num = round(healthy_dataset.shape[1] / interval_seg)
    column_num = round(split_num * interval_seg)
    experiment_healthy_dataset = np.array(healthy_dataset[:, 0: column_num])
    splited_healthy_dataset = np.split(experiment_healthy_dataset, split_num, axis=1)


    # ########################
    # infected plants
    ##########################
    for row in range(len(inf_spec_listdir)):
        spec = inf_spec_listdir[row]
        wavelength, intensity = devide_wave_intensity(spec, inf_path)

        # do calibration
        infected_calibrated_intensity = []
        for i in range(len(intensity)):
            n = intensity[i] / ave_infected_cal[i]
            infected_calibrated_intensity.append(n)

        # resample
        resample_infected_intensity = []
        last = 0
        for i in interval_num:
            resample_infected_intensity.append(sum(infected_calibrated_intensity[last:(last + i)]) / i)
            last = last + i

        if resample_infected_intensity[0] > 1:
            resample_infected_intensity[0] = 1
        if resample_infected_intensity[0] < 0:
            resample_infected_intensity[0] = 0
        for j in range(1, len(resample_infected_intensity)):
            if resample_infected_intensity[j] > 1 or resample_infected_intensity[j] < 0:
                replacement1 = sum(resample_infected_intensity[0:j]) / j
                # replacement2 = sum()
                resample_infected_intensity[j] = replacement1
        # print(resample_infected_intensity)
        infected_dataset_nospline[row] = resample_infected_intensity

        # spline
        func = interpolate.UnivariateSpline(wavelength_dataset, resample_infected_intensity, s=2)
        # xnew = np.arange(400, 800, 2)
        xnew = np.arange(s_wave, key, window)
        inf_ynew = func(xnew)
        infected_dataset[row] = inf_ynew

    interval_seg = 20 / window
    split_num = round(infected_dataset.shape[1] / interval_seg)  # 24
    column_num = round(split_num * interval_seg)
    experiment_infected_dataset = np.array(infected_dataset[:, 0: column_num])
    splited_infected_dataset = np.split(experiment_infected_dataset, split_num, axis=1)

    svc_accuracy = []
    svc_precision = []
    svc_recall = []
    svc_F1 = []
    forest_accuracy = []
    forest_precision = []
    forest_recall = []
    forest_F1 = []
    GBDT_accuracy = []

    for i in range(split_num):
        exp_healthy_set = splited_healthy_dataset[i]
        exp_infected_set = splited_infected_dataset[i]
        exp_set = np.vstack((exp_healthy_set, exp_infected_set))
        full_exp_set = exp_set[0:199]

        # transform to pd
        index1 = pd.Series(np.arange(1, interval_seg + 1))
        index1 = index1.astype(str)
        index1 = 'F' + index1
        # index = np.append(index1, 'labels')
        exp_df = pd.DataFrame(full_exp_set, columns=index1)

        healthy_labels = np.zeros((100, 1))
        infected_labels = np.ones((99, 1))
        whole_labels = np.vstack((healthy_labels, infected_labels))
        # print(whole_labels)
        train_X, test_X, train_y, test_y = train_test_split(exp_df[index1], whole_labels, test_size=0.3, random_state=5)

        # svm
        exp_svc = svm.SVC(C=2.536734693877551, break_ties=False, cache_size=200, class_weight=None,
        coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
        exp_svc.fit(train_X, train_y)
        exp_predict_labels = exp_svc.predict(test_X)
        exp_svc_Accuracy = accuracy_score(test_y, exp_predict_labels)
        svc_accuracy.append(exp_svc_Accuracy)
        print('svm_exp%d Accuracy:' % i, exp_svc_Accuracy)

        # radomforest
        exp_randomforest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=3,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=250,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
        exp_randomforest.fit(train_X, train_y)
        exp_predict_labels = exp_randomforest.predict(test_X)

        exp_forest_Accuracy = accuracy_score(test_y, exp_predict_labels)
        # forest_accuracy.append(exp_forest_Accuracy)
        print('randomforest_exp%d Accuracy:' % i, exp_forest_Accuracy)
        forest_accuracy.append(exp_forest_Accuracy)

        exp_GBDT = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
        exp_GBDT.fit(train_X, train_y)
        exp_predict_labels = exp_GBDT.predict(test_X)
        exp_GBDT_Accuracy = accuracy_score(test_y, exp_predict_labels)
        GBDT_accuracy.append(exp_GBDT_Accuracy)
        print('randomforest_exp%d Accuracy:' % i, exp_GBDT_Accuracy)


    splits = np.arange(split_num)
    splits_specific = 396 + splits * 20
    

    

    # ax.plot(splits_specific, svc_accuracy, color='red', label='svm_accuracy')
    ax.plot(splits_specific, forest_accuracy, color = color_list[day-2], label=day_list[day-2])
    # ax.plot(splits_specific, GBDT_accuracy, color='blue', label='GBM_accuracy')
    # ax.plot(splits_specific, forest_F1, color='pink', label='F1-measure')
    ax.set_xlabel(r'wavelength')
    ax.set_ylabel(r"accuracy")

    ax2_share_y = ax.twiny()
    # ax2_share_y.plot(splits, svc_accuracy, color='red', label='svm_accuracy')
    ax2_share_y.plot(splits, forest_accuracy,color = color_list[day-2], label=day_list[day-2])
    # ax2_share_y.plot(splits, GBDT_accuracy, color='red', label='GBM_accuracy')
    # ax2_share_y.plot(splits, forest_F1, color='pink', label='F1-measure')
    ax2_share_y.set_xlabel("the number of chunks")
    # plt.ylabel(r'light intensity')
    plt.legend(loc = loc_list[day-2])
    # plt.title(r'redundancy: %d nm, range(%d, %d)' % (window, s_wave, key))
    # plt.title(r'redundancy: %d nm, range(%d, %d)' % (window, 400, 800))

plt.ylim(0, 1)
plt.show()

        