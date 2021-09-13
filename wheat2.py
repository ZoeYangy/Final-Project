import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from scipy import interpolate


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
        print(len(file_listdir))
        sum_cal_intensity = 0
        for cal in file_listdir:
            wave, inten = devide_wave_intensity(cal, absolute_path)
            sum_cal_intensity += inten[w]
        mean_inten = sum_cal_intensity / len(file_listdir)
        average.append(mean_inten)
    return average


# 13-day healthy
healthy_path = os.path.abspath(r'D:\\Durham\\Wheat_dataset\\Wheat_dataset\\13 dpi\\healthy')
inf_path = os.path.abspath(r'D:\\Durham\\Wheat_dataset\\Wheat_dataset\\13 dpi\\inf')
healthy_files = os.listdir(healthy_path)
inf_files = os.listdir(inf_path)

healthy_spec_listdir, healthy_cal_listdir = file_divided(healthy_files)
inf_spec_listdir, inf_cal_listdir = file_divided(inf_files)

# wavelength, intensity = devide_wave_intensity('1.mspec', healthy_path)
# plt.plot(wavelength, intensity)
# plt.xlabel(r'wavelength')
# plt.ylabel(r'reflectance intensity')
# plt.show()


# other 12 days


# Calculate the average calibration value of healthy and infected plants seperately
'''
ave_healthy_cal = average_intensity(healthy_cal_listdir, healthy_path)
fileobject = open('D:\\final_project\\code\\ave_healthy_intensity.txt', 'w')
for i in ave_healthy_cal:
    fileobject.write(str(i))
    fileobject.write('\n')
fileobject.close()

ave_infected_cal = average_intensity(inf_cal_listdir, inf_path)
fileobject = open('D:\\final_project\\code\\ave_infected_intensity.txt', 'w')
for i in ave_infected_cal:
    fileobject.write(str(i))
    fileobject.write('\n')
fileobject.close()
'''
# read in the average value
with open('D:\\Durham\\code\\code\\ave_healthy_intensity.txt') as file_obj:
    lines = file_obj.readlines()
ave_healthy_cal = []
for line in lines:
    light = float(line.rstrip())
    ave_healthy_cal.append(light)
print(len(ave_healthy_cal))

with open('D:\\Durham\\code\\code\\ave_infected_intensity.txt') as file_obj:
    lines = file_obj.readlines()
ave_infected_cal = []
for line in lines:
    light = float(line.rstrip())
    ave_infected_cal.append(light)
print(len(ave_infected_cal))

# resampling (only for wavelength)
# Because every spectra have same wavelength, so to get the processed wavelength list, just need to take one spectra as example
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
for row in range(len(healthy_spec_listdir)):
    spec = healthy_spec_listdir[row]
    wavelength, intensity = devide_wave_intensity(spec, healthy_path)

    # do calibration
    healthy_calibrated_intensity = []
    for i in range(len(intensity)):
        n = intensity[i] / ave_healthy_cal[i]
        healthy_calibrated_intensity.append(n)
    # print(len(healthy_calibrated_intensity))
    # plt.plot(wavelength, healthy_calibrated_intensity)
    # plt.ylim(-15, 15)
    # plt.xlabel(r'wavelength')
    # plt.ylabel(r'reflectance intensity')
    # plt.show()


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


    # plt.plot(wavelength_dataset, resample_healthy_intensity)
    # # plt.ylim(-15, 15)
    # plt.xlabel(r'wavelength')
    # plt.ylabel(r'reflectance intensity')
    # plt.show()

    # spline
    func = interpolate.UnivariateSpline(wavelength_dataset, resample_healthy_intensity, s=3)
    # xnew = np.arange(400, 800, 2)
    xnew = np.arange(s_wave, key, window)
    hea_ynew = func(xnew)
    # plt.plot(xnew, hea_ynew, label='cubic interpolation')
    # plt.xlabel(r'wavelength')
    # plt.ylabel(r'reflectance intensity')
    # plt.ylim(0, 1)
    # plt.show()

    print(hea_ynew)
    # healthy_dataset[row] = resample_healthy_intensity
    healthy_dataset[row] = hea_ynew

# create wavelength segmentation 20nm/1 segmentation
# num_seg = len(hea_ynew) * window / 20
interval_seg = 20 / window
split_num = round(healthy_dataset.shape[1] / interval_seg)
column_num = round(split_num * interval_seg)
experiment_healthy_dataset = np.array(healthy_dataset[:, 0: column_num])
splited_healthy_dataset = np.split(experiment_healthy_dataset, split_num, axis=1)

# Not sure this function. Do I need to multiply the redundancy reduction window ?????
# healthy_segments = {}
# process = 0
# process_point = 0
# seg = 1
# # healthy_segments[seg] = np.empty([len(healthy_dataset), int(interval_seg)])
# healthy_segments[seg] = []
# while process_point <= (len(hea_ynew)-1):
#     if process >= 10:
#         # process_point = seg * process
#         process = 0
#         seg += 1
#         healthy_segments[seg] = np.empty([len(healthy_dataset), int(interval_seg)])
#         healthy_segments[seg] = []
#         # healthy_segments[seg].append(healthy_dataset[process_point])
#         healthy_segments = np.append(healthy_segments[seg], np.array(healthy_dataset[:, process_point]))
#         process_point += 1
#     else:
#         # healthy_segments[seg].append(healthy_dataset[process_point])
#         healthy_segments = np.append(healthy_segments[seg], np.array(healthy_dataset[:, process_point]))
#         process_point += 1
#         process += 1


    # fileobject = open('D:\\final_project\\code\\healthy_dataset2.txt', 'w')
    # for i in healthy_dataset:
    #     fileobject.write(str(i))
    #     fileobject.write('\n')
    # fileobject.close()

    # total_dateset[row] = resample_healthy_intensity
print('the original healthy calibrated data:')
# print(healthy_calibrated_intensity)
# print(healthy_calibrated_intensity[21:33])
print(sum(healthy_calibrated_intensity[0:8]) / 8)
print(sum(healthy_calibrated_intensity[8:21]) / 13)
print(sum(healthy_calibrated_intensity[21:33]) / 12)

# Plots
# wavelength_dataset = range(396, key, window)
# plt.plot(wavelength_dataset, healthy_dataset[-1], label='data after redundancy reduction')
# plt.xlabel(r'wavelength')
# plt.ylabel(r'light intensity')
# plt.title(r'spectra 100 of date 13')
# plt.legend()
# plt.grid()


# spline interpolation
# func = interpolate.UnivariateSpline(wavelength_dataset, healthy_dataset[-1], s=2)
# xnew = np.arange(396, key, 2)
# ynew = func(xnew)
# plt.plot(xnew, ynew, label='cubic interpolation')
# plt.show()

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
    # plt.plot(xnew, inf_ynew, label='cubic interpolation')
    # plt.show()
    # healthy_dataset[row] = resample_healthy_intensity
    infected_dataset[row] = inf_ynew
    # print(resample_healthy_intensity)
    # total_dateset[row+100] = resample_infected_intensity
# infected_dataset = infected_dataset[0:99]
# print('final_row = %d' % row)

interval_seg = 20 / window
split_num = round(infected_dataset.shape[1] / interval_seg)  # 24
column_num = round(split_num * interval_seg)
experiment_infected_dataset = np.array(infected_dataset[:, 0: column_num])
splited_infected_dataset = np.split(experiment_infected_dataset, split_num, axis=1)

ave_healthy_intensity = []
ave_infected_intensity = []
# for row in range(len(healthy_dataset)):
for i in range(num_points):
    n = sum(healthy_dataset[:, i]) / len(healthy_dataset)
    m = sum(infected_dataset[:, i]) / len(infected_dataset)
    ave_healthy_intensity.append(n)
    ave_infected_intensity.append(m)
ave_intensity = np.vstack((ave_healthy_intensity, ave_infected_intensity))

# Plot the average intensity of healthy plants
np_healthy_intensity = np.array(ave_healthy_intensity)
np_infected_intensity = np.array(ave_infected_intensity)
difference = np_healthy_intensity - np_infected_intensity
# plt.plot(xnew, np_healthy_intensity, color='red', label='average intensity from healthy plants')
# # plt.plot(xnew, np_infected_intensity, color='green')
# plt.errorbar(x=xnew, y=np_healthy_intensity, yerr=difference, color='blue', alpha=0.3, label='difference value')
# plt.xlabel(r'wavelength')
# plt.ylabel(r'light intensity')
# plt.ylim(0, 1)
# # plt.title(r'redundancy: %d nm, range(%d, %d)' % (window, s_wave, key))
# # plt.title(r'redundancy: %d nm, range(%d, %d)' % (window, 400, 800))
# plt.legend()
# plt.show()

index1 = pd.Series(np.arange(1, (num_points + 1)))
index1 = index1.astype(str)
index = 'F' + index1
df_whole_intensity = pd.DataFrame(ave_intensity, columns=index)

# the label of healthy plants is 0， infected is 1
labels = np.zeros(100)
healthy_dataset_labels = np.insert(healthy_dataset, num_points, values=labels, axis=1)
labels = np.ones(100)
infected_dataset_labels = np.insert(infected_dataset, num_points, values=labels, axis=1)
# Combine two matrics
whole_dataset = np.vstack((healthy_dataset, infected_dataset))
whole_dataset = whole_dataset[0:199]
whole_dataset_labels = np.vstack((healthy_dataset_labels, infected_dataset_labels))
whole_dataset_labels = whole_dataset_labels[0:199]

######## no spline data
healthy_dataset_nospline_labels = np.insert(healthy_dataset_nospline, num_points, values=labels, axis=1)
infected_dataset_nospline_labels = np.insert(infected_dataset_nospline, num_points, values=labels, axis=1)
# Combine two matrics
whole_dataset_nospline = np.vstack((healthy_dataset_nospline, infected_dataset_nospline))
whole_dataset_nospline = whole_dataset_nospline[0:199]
whole_dataset_nospline_labels = np.vstack((healthy_dataset_nospline_labels, infected_dataset_nospline_labels))
whole_dataset_nospline_labels = whole_dataset_nospline_labels[0:199]

'''
# Save dataset
fileobject = open('D:\\Durham\\code\\code\\average_cal\\whole_dataset_nospline_labels.txt', 'w')  # whole dataset with labels
for i in whole_dataset_nospline_labels:
    fileobject.write(str(i))
    fileobject.write('\n')
fileobject.close()

fileobject = open('D:\\Durham\\code\\code\\average_cal\\whole_dataset_nospline_.txt', 'w')  # whole dataset without labels
for i in whole_dataset_nospline:
    fileobject.write(str(i))
    fileobject.write('\n')
fileobject.close()
'''

index1 = pd.Series(np.arange(1, (num_points + 1)))
index1 = index1.astype(str)
index = 'F' + index1
df_whole_dataset = pd.DataFrame(whole_dataset, columns=index)
df_whole_dataset_nospline = pd.DataFrame(whole_dataset_nospline, columns = index)

# PCA
features_num = 40
pca_num = PCA(n_components=features_num)
newX_pca = pca_num.fit_transform(whole_dataset)
# print(healthy_newX_pca)
print(pca_num.explained_variance_ratio_)
sum_pca_ratio = []
kk = 0
for i in range(features_num):
    kk = sum(pca_num.explained_variance_ratio_[0:i])
    sum_pca_ratio.append(kk)
print(sum_pca_ratio)

# plt.plot(np.arange(features_num), sum_pca_ratio)
# plt.xlabel(r'number of included features')
# plt.ylabel(r'variance ratio')
# # plt.title(r'spectra 1 of date 13')
# # plt.legend()
# plt.grid()
# plt.show()

labels1 = np.zeros((100, 1))
labels2 = np.ones((99, 1))
whole_labels = np.vstack((labels1, labels2))  # 199行1列：label
pca_features_labels = np.hstack((newX_pca, whole_labels))   # 加上标签后的pca数据集
print(pca_features_labels)
fileobject = open('D:\\Durham\\code\\code\\average_cal\\pca_features_labels.txt', 'w')
for i in pca_features_labels:
    fileobject.write(str(i))
    fileobject.write('\n')
fileobject.close()

# transform to df
index1 = pd.Series(np.arange(1, features_num + 1))
index1 = index1.astype(str)
index1 = 'F' + index1
index = np.append(index1, 'labels')
pca_df = pd.DataFrame(pca_features_labels, columns=index)




# fast_ICA
fastica_num = FastICA(n_components=features_num, max_iter=10000)
newX_ica = fastica_num.fit_transform(whole_dataset)
# print(healthy_newX_ica)
ica_features_labels = np.hstack((newX_ica, whole_labels))
# transform to df
index1 = pd.Series(np.arange(1, features_num + 1))
index1 = index1.astype(str)
index1 = 'F' + index1
index = np.append(index1, 'labels')
ica_df = pd.DataFrame(ica_features_labels, columns=index)

# Anova
df_melt = df_whole_intensity.melt()
df_melt.columns = ['Band', 'Value']
anova_model = ols('Value~C(Band)', data=df_melt).fit()
anova_table = anova_lm(anova_model, typ=2)
print(anova_table)
# Tukey HSD test. To see which bands have different influence between each others
mc = MultiComparison(df_melt['Value'], df_melt['Band'])  # multiple test
tukey_result = mc.tukeyhsd(alpha=0.5)
tukey_data = tukey_result._results_table.data[1:]
df_tukey_data = pd.DataFrame(tukey_data, columns=['group1', 'group2', 'meandiff', 'P-adj', 'lower', 'upper', 'reject'])
df_tukey_data['reject'] = df_tukey_data['reject'].astype('str')
print(df_tukey_data)
true_reject = df_tukey_data.loc[df_tukey_data['reject'] == 'True'] # all rejected pairs
calculator = 0
dict_calculation = {}
rejected_times = []
for i in range(1, num_points+1): # (1,200)
    selected_tukey_1 = true_reject.loc[true_reject['group1'] == 'F%d' % i]
    selected_tukey_2 = true_reject.loc[true_reject['group2'] == 'F%d' % i]
    calculator = len(selected_tukey_1.index) 
    dict_calculation['F%d' % i] = calculator
    rejected_times.append(calculator)
rejected_times.sort(reverse = True)
dict_calculation = sorted(dict_calculation.items(), key=lambda item: item[1], reverse=True) # different pairs
#  select the most releted features
anova_feature_names = []
selected_feature_num = 40  # select 40 wavelengths
for i in range(0, selected_feature_num):
    anova_feature_names.append(dict_calculation[i][0])
anova_features = df_whole_dataset[anova_feature_names]
anova_features_labels = np.hstack((anova_features, whole_labels))
plt.plot(anova_feature_names[0:30], rejected_times[0:30])
plt.xlabel(r'The number of features')
plt.ylabel(r'rejected times')
# plt.title(r'spectra 1 of date 13')
# plt.legend()
# plt.grid()
plt.show()
# transform to pd
index1 = pd.Series(np.arange(1, features_num + 1))
index1 = index1.astype(str)
index1 = 'F' + index1
index = np.append(index1, 'labels')
anova_df = pd.DataFrame(anova_features_labels, columns=index)

#  +++++++++++++++++++++++++++++++++++++++++++++++++
#                 Machine Learning
#  +++++++++++++++++++++++++++++++++++++++++++++++++

pca_train_X, pca_test_X, pca_train_y, pca_test_y = train_test_split(pca_df[index1], pca_df['labels'], test_size=0.3,
                                                                    random_state=5)
anova_train_X, anova_test_X, anova_train_y, anova_test_y = train_test_split(anova_df[index1], anova_df['labels'],
                                                                            test_size=0.3, random_state=5)
ica_train_X, ica_test_X, ica_train_y, ica_test_y = train_test_split(ica_df[index1], ica_df['labels'], test_size=0.3,
                                                                    random_state=5)
# SVM using PCA data

# Create SVM classification object
# GridSearch
# parameters = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'C': np.linspace(0.1, 20, 50),
#               'gamma': np.linspace(0.1, 20, 20)}
# svc = svm.SVC()
# pca_model = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
# pca_model.fit(pca_train_X, pca_train_y)
# print(pca_model.best_params_)
# pca_model.score(pca_test_X, pca_test_y)
# print(pca_model.best_estimator_)
# pca_svc = svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
#                   decision_function_shape='ovr', degree=3, gamma=7.43157894736842,
#                   kernel='sigmoid', max_iter=-1, probability=False, random_state=None,
#                   shrinking=True, tol=0.001, verbose=False)
pca_svc = svm.SVC(C=2.536734693877551, break_ties=False, cache_size=200, class_weight=None,
    coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

# best_params_ = {'C': 0.1, 'gamma': 7.43157894736842, 'kernel': 'sigmoid'}
# pca_svc = svm.SVC(best_params_)
pca_svc.fit(pca_train_X, pca_train_y)
# loss, accuracy = pca_svc.evaluate(test_X, test_y)
# print('\ntest loss', loss)
# print('accuracy', accuracy)
pca_predict_labels = pca_svc.predict(pca_test_X)
pca_svc_Accuracy = accuracy_score(pca_test_y, pca_predict_labels)
pca_svc_Precision = precision_score(pca_test_y, pca_predict_labels, pos_label=0)
pca_svc_Recall = recall_score(pca_test_y, pca_predict_labels, pos_label=0)
pca_svc_F1_scores = f1_score(pca_test_y, pca_predict_labels, pos_label=0)
pca_svc_fpr, pca_svc_tpr, thresholds = roc_curve(pca_test_y, pca_predict_labels, pos_label=1)
# plt.plot(pca_svc_fpr, pca_svc_tpr)

# SVM using Anova data
# GridSearch
# parameters = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'C': np.linspace(0.1, 20, 50),
#               'gamma': np.linspace(0.1, 20, 20)}
# svc = svm.SVC()
# model = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
# model.fit(anova_train_X, anova_train_y)
# model.best_params_
# model.score(anova_test_X, anova_test_y)
# print(model.best_estimator_)
# print('=============================')
# anova_svc = svm.SVC(C=3.3489795918367347, cache_size=200, class_weight=None, coef0=0.0,
#                     decision_function_shape='ovr', degree=3, gamma=0.1, kernel='linear',
#                     max_iter=-1, probability=False, random_state=None, shrinking=True,
#                     tol=0.001, verbose=False)
anova_svc = svm.SVC(C=16.751020408163267, break_ties=False, cache_size=200, class_weight=None,
    coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
anova_svc.fit(anova_train_X, anova_train_y)
anova_predict_labels = anova_svc.predict(anova_test_X)
anova_svc_Accuracy = accuracy_score(anova_test_y, anova_predict_labels)
anova_svc_Precision = precision_score(anova_test_y, anova_predict_labels, pos_label=0)
anova_svc_Recall = recall_score(anova_test_y, anova_predict_labels, pos_label=0)
anova_svc_F1_scores = f1_score(anova_test_y, anova_predict_labels, pos_label=0)
anova_svc_fpr, anova_svc_tpr, thresholds = roc_curve(anova_test_y, anova_predict_labels, pos_label=1)
# plt.plot(anova_svc_fpr, anova_svc_tpr)
# SVM using ica data
# parameters = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'C': np.linspace(0.1, 20, 50),
#               'gamma': np.linspace(0.1, 20, 20)}
# svc = svm.SVC()
# ica_model = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
# ica_model.fit(ica_train_X, ica_train_y)
# ica_model.best_params_
# ica_model.score(ica_test_X, ica_test_y)
# print(ica_model.best_estimator_)

# ica_svc = svm.SVC(C=7.816326530612244, cache_size=200, class_weight=None, coef0=0.0,
#                   decision_function_shape='ovr', degree=3, gamma=3.2421052631578946,
#                   kernel='sigmoid', max_iter=-1, probability=False, random_state=None,
#                   shrinking=True, tol=0.001, verbose=False)
ica_svc = svm.SVC(C=2.1306122448979594, break_ties=False, cache_size=200, class_weight=None,
    coef0=0.0, decision_function_shape='ovr', degree=3,
    gamma=3.2421052631578946, kernel='sigmoid', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)
ica_svc.fit(ica_train_X, ica_train_y)
ica_predict_labels = ica_svc.predict(ica_test_X)
ica_svc_Accuracy = accuracy_score(ica_test_y, ica_predict_labels)
ica_svc_Precision = precision_score(ica_test_y, ica_predict_labels, pos_label=0)
ica_svc_Recall = recall_score(ica_test_y, ica_predict_labels, pos_label=0)
ica_svc_F1_scores = f1_score(ica_test_y, ica_predict_labels, pos_label=0)
ica_svc_fpr, ica_svc_tpr, thresholds = roc_curve(ica_test_y, ica_predict_labels, pos_label=1)
# plt.plot(ica_svc_fpr, ica_svc_tpr)
# plt.show()
# =======================================================================================
# RandomForest using pca data
rfc = RandomForestClassifier()
tree_param_grid = {"n_estimators": [250, 300],
                   "criterion": ["gini", "entropy"],
                   "max_features": [3, 5],
                   "max_depth": [10, 20],
                   "min_samples_split": [2, 4],
                   "bootstrap": [True, False]}
tree_model = GridSearchCV(rfc, tree_param_grid, n_jobs=-1, cv=5, scoring='accuracy')
# 
pca_randomforest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=3,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=250,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
# pca_randomforest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                                           max_depth=10, max_features=3, max_leaf_nodes=None,
#                                           min_impurity_decrease=0.0, min_impurity_split=None,
#                                           min_samples_leaf=1, min_samples_split=2,
#                                           min_weight_fraction_leaf=0.0, n_estimators=250,
#                                           n_jobs=None, oob_score=False, random_state=None,
#                                           verbose=0, warm_start=False)
pca_randomforest.fit(pca_train_X, pca_train_y)
pca_predict_labels = pca_randomforest.predict(pca_test_X)
pca_forest_Accuracy = accuracy_score(pca_test_y, pca_predict_labels)
pca_forest_Precision = precision_score(pca_test_y, pca_predict_labels, pos_label=0)
pca_forest_Recall = recall_score(pca_test_y, pca_predict_labels, pos_label=0)
pca_forest_F1_scores = f1_score(pca_test_y, pca_predict_labels, pos_label=0)

# Randomforest using anova data

# tree_model = GridSearchCV(rfc, tree_param_grid, n_jobs=-1, cv=5, scoring='accuracy')
# tree_model.fit(anova_train_X, anova_train_y)
# print(tree_model.best_params_)
# print(tree_model.score(anova_test_X, anova_test_y))
# print(tree_model.best_estimator_)
# print('+++++++++++++++++++++++++++++++++++++++++++++++++')
anova_randomforest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=3,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=300,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
# anova_randomforest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                                             max_depth=10, max_features=3, max_leaf_nodes=None,
#                                             min_impurity_decrease=0.0, min_impurity_split=None,
#                                             min_samples_leaf=1, min_samples_split=2,
#                                             min_weight_fraction_leaf=0.0, n_estimators=300,
#                                             n_jobs=None, oob_score=False, random_state=None,
#                                             verbose=0, warm_start=False)
anova_randomforest.fit(anova_train_X, anova_train_y)
anova_predict_labels = anova_randomforest.predict(anova_test_X)
anova_forest_Accuracy = accuracy_score(anova_test_y, anova_predict_labels)
anova_forest_Precision = precision_score(anova_test_y, anova_predict_labels, pos_label=0)
anova_forest_Recall = recall_score(anova_test_y, anova_predict_labels, pos_label=0)
anova_forest_F1_scores = f1_score(anova_test_y, anova_predict_labels, pos_label=0)

# Randomforest using ica data
# tree_model = GridSearchCV(rfc, tree_param_grid, n_jobs=-1, cv=5, scoring='accuracy')
# tree_model.fit(ica_train_X, ica_train_y)
# print(tree_model.best_params_)
# print(tree_model.score(anova_test_X, anova_test_y))
# print(tree_model.best_estimator_)
# print('tree finish !!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
ica_randomforest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=20, max_features=3,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=4,
                       min_weight_fraction_leaf=0.0, n_estimators=300,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
# ica_randomforest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                                           max_depth=10, max_features=5, max_leaf_nodes=None,
#                                           min_impurity_decrease=0.0, min_impurity_split=None,
#                                           min_samples_leaf=1, min_samples_split=4,
#                                           min_weight_fraction_leaf=0.0, n_estimators=250,
#                                           n_jobs=None, oob_score=False, random_state=None,
#                                           verbose=0, warm_start=False)
ica_randomforest.fit(ica_train_X, ica_train_y)
ica_predict_labels = ica_randomforest.predict(ica_test_X)
ica_forest_Accuracy = accuracy_score(ica_test_y, ica_predict_labels)
ica_forest_Precision = precision_score(ica_test_y, ica_predict_labels, pos_label=0)
ica_forest_Recall = recall_score(ica_test_y, ica_predict_labels, pos_label=0)
ica_forest_F1_scores = f1_score(ica_test_y, ica_predict_labels, pos_label=0)

# ===============================================================
# GBDT_parameter = {'learning_rate': np.linspace(0.10, 0.3, num=10), 'n_estimators': range(10, 100, 5),
#                   'max_depth': [3, 4, 5, 6, 7, 8, 10]}
# clf = GradientBoostingClassifier()
# GBDT_model = GridSearchCV(clf, GBDT_parameter, cv=5, scoring='accuracy', n_jobs=-1, verbose=2, return_train_score=False)
# GBDT_model.fit(pca_train_X, pca_train_y)
# GBDT_model.best_params_
# print(GBDT_model.score(pca_test_X, pca_test_y))
# print(GBDT_model.best_estimator_)
# print('========================')
pca_GBDT = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=10,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
# GBDT using pca data
# GBDT_model.fit(pca_train_X, pca_train_y)
# pca_GBDT = GradientBoostingClassifier(criterion='friedman_mse', init=None,
#                                       learning_rate=0.1, loss='deviance', max_depth=3,
#                                       max_features=None, max_leaf_nodes=None,
#                                       min_impurity_decrease=0.0, min_impurity_split=None,
#                                       min_samples_leaf=1, min_samples_split=2,
#                                       min_weight_fraction_leaf=0.0, n_estimators=10,
#                                       n_iter_no_change=None, presort='auto',
#                                       random_state=None, subsample=1.0, tol=0.0001,
#                                       validation_fraction=0.1, verbose=0,
#                                       warm_start=False)
pca_GBDT.fit(pca_train_X, pca_train_y)
pca_predict_labels = pca_GBDT.predict(pca_test_X)
pca_GBDT_Accuracy = accuracy_score(pca_test_y, pca_predict_labels)
pca_GBDT_Precision = precision_score(pca_test_y, pca_predict_labels, pos_label=0)
pca_GBDT_Recall = recall_score(pca_test_y, pca_predict_labels, pos_label=0)
pca_GBDT_F1_scores = f1_score(pca_test_y, pca_predict_labels, pos_label=0)

# GBDT using anova data
# GBDT_model.fit(anova_train_X, anova_train_y)
# GBDT_model.fit(anova_train_X, anova_train_y)
# GBDT_model.best_params_
# print(GBDT_model.score(anova_test_X, anova_test_y))
# print(GBDT_model.best_estimator_)
# print('============================')
anova_GBDT = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=90,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
# anova_GBDT = GradientBoostingClassifier(criterion='friedman_mse', init=None,
#                                         learning_rate=0.1, loss='deviance', max_depth=3,
#                                         max_features=None, max_leaf_nodes=None,
#                                         min_impurity_decrease=0.0, min_impurity_split=None,
#                                         min_samples_leaf=1, min_samples_split=2,
#                                         min_weight_fraction_leaf=0.0, n_estimators=10,
#                                         n_iter_no_change=None, presort='auto',
#                                         random_state=None, subsample=1.0, tol=0.0001,
#                                         validation_fraction=0.1, verbose=0,
#                                         warm_start=False)
anova_GBDT.fit(anova_train_X, anova_train_y)
anova_predict_labels = anova_GBDT.predict(anova_test_X)
anova_GBDT_Accuracy = accuracy_score(anova_test_y, anova_predict_labels)
anova_GBDT_Precision = precision_score(anova_test_y, anova_predict_labels, pos_label=0)
anova_GBDT_Recall = recall_score(anova_test_y, anova_predict_labels, pos_label=0)
anova_GBDT_F1_scores = f1_score(anova_test_y, anova_predict_labels, pos_label=0)

# GBDT using ica data
# GBDT_model.fit(ica_train_X, ica_train_y)
# GBDT_model.fit(ica_train_X, ica_train_y)
# print(GBDT_model.best_params_)
# print(GBDT_model.score(ica_test_X, ica_test_y))
# print(GBDT_model.best_estimator_)
ica_GBDT = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.2777777777777778, loss='deviance',
                           max_depth=3, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=30,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
# ica_GBDT = GradientBoostingClassifier(criterion='friedman_mse', init=None,
#                                       learning_rate=0.1631578947368421, loss='deviance',
#                                       max_depth=3, max_features=None, max_leaf_nodes=None,
#                                       min_impurity_decrease=0.0, min_impurity_split=None,
#                                       min_samples_leaf=1, min_samples_split=2,
#                                       min_weight_fraction_leaf=0.0, n_estimators=120,
#                                       n_iter_no_change=None, presort='auto',
#                                       random_state=None, subsample=1.0, tol=0.0001,
#                                       validation_fraction=0.1, verbose=0,
#                                       warm_start=False)
ica_GBDT.fit(ica_train_X, ica_train_y)
ica_predict_labels = ica_GBDT.predict(ica_test_X)
ica_GBDT_Accuracy = accuracy_score(ica_test_y, ica_predict_labels)
ica_GBDT_Precision = precision_score(ica_test_y, ica_predict_labels, pos_label=0)
ica_GBDT_Recall = recall_score(ica_test_y, ica_predict_labels, pos_label=0)
ica_GBDT_F1_scores = f1_score(ica_test_y, ica_predict_labels, pos_label=0)

flist = ['Accuracy', 'Precision', 'Recall', 'F1_scores']
svm_comparison_results = pd.DataFrame(index=flist, columns=np.arange(split_num))
randomforest_comparison_results = pd.DataFrame(index=flist, columns=np.arange(split_num))




interval_seg = 20 / window
split_num = round(infected_dataset.shape[1] / interval_seg)  # 24
column_num = round(split_num * interval_seg)
experiment_infected_dataset = np.array(infected_dataset[:, 0: column_num])
splited_infected_dataset = np.split(experiment_infected_dataset, split_num, axis=1)
# comparision_results = {}
# comparision_results[1] = [{}, {}, {}, {}]
# comparision_results[1]['Accuracy'] = []
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

    train_X, test_X, train_y, test_y = train_test_split(exp_df[index1], pca_df['labels'], test_size=0.3, random_state=5)


    # svm
    exp_svc = svm.SVC(C=2.536734693877551, break_ties=False, cache_size=200, class_weight=None,
        coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    # exp_svc = svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
    #                   decision_function_shape='ovr', degree=3, gamma=7.43157894736842,
    #                   kernel='sigmoid', max_iter=-1, probability=False, random_state=None,
    #                   shrinking=True, tol=0.001, verbose=False)
    exp_svc.fit(train_X, train_y)
    # loss, accuracy = pca_svc.evaluate(test_X, test_y)
    # print('\ntest loss', loss)
    # print('accuracy', accuracy)
    exp_predict_labels = exp_svc.predict(test_X)
    exp_svc_Accuracy = accuracy_score(test_y, exp_predict_labels)
    svc_accuracy.append(exp_svc_Accuracy)
    print('svm_exp%d Accuracy:' % i, exp_svc_Accuracy)

    # exp_svc_Precision = precision_score(test_y, exp_predict_labels, pos_label=0)
    # print('svm_exp%d Precision:' % i, exp_svc_Precision)

    # exp_svc_Recall = recall_score(test_y, exp_predict_labels, pos_label=0)
    # print('svm_exp%d Recall:' % i, exp_svc_Recall)

    # exp_svc_F1_scores = f1_score(test_y, exp_predict_labels, pos_label=0)
    # print('svm_exp%d F1_scores:' % i, exp_svc_F1_scores)
    # radomforest
    exp_randomforest = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=10, max_features=3,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=250,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
    # exp_randomforest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                                           max_depth=10, max_features=3, max_leaf_nodes=None,
    #                                           min_impurity_decrease=0.0, min_impurity_split=None,
    #                                           min_samples_leaf=1, min_samples_split=2,
    #                                           min_weight_fraction_leaf=0.0, n_estimators=250,
    #                                           n_jobs=None, oob_score=False, random_state=None,
    #                                           verbose=0, warm_start=False)
    exp_randomforest.fit(train_X, train_y)
    exp_predict_labels = exp_randomforest.predict(test_X)

    exp_forest_Accuracy = accuracy_score(test_y, exp_predict_labels)
    forest_accuracy.append(exp_forest_Accuracy)
    print('randomforest_exp%d Accuracy:' % i, exp_forest_Accuracy)

    # exp_forest_Precision = precision_score(test_y, exp_predict_labels, pos_label=0)
    # forest_precision.append(exp_forest_Precision)
    # print('randomforest_exp%d Precision:' % i, exp_forest_Precision)

    # exp_forest_Recall = recall_score(test_y, exp_predict_labels, pos_label=0)
    # forest_recall.append(exp_forest_Recall)
    # print('randomforest_exp%d Recall:' % i, exp_forest_Recall)

    # exp_forest_F1_scores = f1_score(test_y, exp_predict_labels, pos_label=0)
    # forest_F1.append(exp_forest_F1_scores)
    # print('randomforest_exp%d F1_scores:' % i, exp_forest_F1_scores)

    # GBDT
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
    print('GBDT_exp%d Accuracy:' % i, exp_GBDT_Accuracy)

splits = np.arange(split_num)
splits_specific = 396 + splits * 20
fg = plt.figure()
ax = fg.add_subplot(1,1,1)

ax.plot(splits_specific, svc_accuracy, color='red', label='svm_accuracy')
ax.plot(splits_specific, forest_accuracy, color='green', label='randomforest_accuracy')
ax.plot(splits_specific, GBDT_accuracy, color='blue', label='GBM_accuracy')
# ax.plot(splits_specific, forest_F1, color='pink', label='F1-measure')
ax.set_xlabel(r'wavelength')
ax.set_ylabel(r"modeling accuracy")

ax2_share_y = ax.twiny()
ax2_share_y.plot(splits, svc_accuracy, color='red', label='svm_accuracy')
ax2_share_y.plot(splits, forest_accuracy, color='green', label='randomforest_accuracy')
ax2_share_y.plot(splits, GBDT_accuracy, color='blue', label='GBM_accuracy')
# ax2_share_y.plot(splits, forest_F1, color='pink', label='F1-measure')
ax2_share_y.set_xlabel("the number of chunks")
# plt.ylabel(r'light intensity')
plt.ylim(0, 1)
# plt.title(r'redundancy: %d nm, range(%d, %d)' % (window, s_wave, key))
# plt.title(r'redundancy: %d nm, range(%d, %d)' % (window, 400, 800))
plt.legend()
plt.show()
