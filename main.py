import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('database.csv')

# map reduce
def map_injured(table):
    for note in table.iterrows():
        # print(note[1])
        yield (note[1]['UNIQUE KEY'], [note[1]['PERSONS INJURED'],
                                       note[1]['PEDESTRIANS INJURED'],
                                       note[1]['CYCLISTS INJURED'],
                                       note[1]['MOTORISTS INJURED']])

def reduce_injured(table, mapped):
    index = 0
    print(mapped.)
    for note in table.iterrows():
        table[note[0], 'INJURED'] = sum([1])
        index += 1

    # for note in mapped:
        # print(note)
        # table.loc[table['UNIQUE KEY'] == note[0]]['INJURED'] = sum(note[1])


data['INJURED'] = 0
reduce_injured(data, map_injured(data))

print(data.head())

raise TypeError()


# data['INJURED'] = data['PERSONS INJURED'] + data['PEDESTRIANS INJURED'] + data['CYCLISTS INJURED'] + data['MOTORISTS INJURED']

# data['KILLED'] = data['PERSONS KILLED'] + data['PEDESTRIANS KILLED'] + data['CYCLISTS KILLED'] + data['MOTORISTS KILLED']

# drop_column = ['UNIQUE KEY','LOCATION','ON STREET NAME',
#                'CROSS STREET NAME','OFF STREET NAME','DATE','TIME','ZIP CODE',
#               'PERSONS INJURED','PEDESTRIANS INJURED','CYCLISTS INJURED','MOTORISTS INJURED',
#               'PERSONS KILLED','PEDESTRIANS KILLED','CYCLISTS KILLED','MOTORISTS KILLED']
#
# data_with_drop = data.drop(drop_column, axis=1)
#
# from sklearn.preprocessing import LabelEncoder
#
# data_categorical = data_with_drop.copy()
# categorical_values = ['BOROUGH','VEHICLE 1 TYPE','VEHICLE 2 TYPE','VEHICLE 3 TYPE','VEHICLE 4 TYPE','VEHICLE 5 TYPE',
#                      'VEHICLE 1 FACTOR','VEHICLE 2 FACTOR','VEHICLE 3 FACTOR','VEHICLE 4 FACTOR','VEHICLE 5 FACTOR']
# categorical_encoders = []
# for col_name in categorical_values:
#     data_categorical[col_name].fillna('UNKNOWN',inplace=True)
#     encoder = LabelEncoder().fit(data_categorical[col_name])
#     data_categorical[col_name] = pd.Series(encoder.fit_transform(data_categorical[col_name]))
#     categorical_encoders.append(encoder)
#
# import sklearn
#
# data_categorical.dropna(inplace=True)
# data['VEHICLE 1 TYPE'].unique()
#
# from matplotlib import pyplot as PLT
#
# %pylab inline
# pylab.rcParams['figure.figsize'] = (20, 12)
#
# # from matplotlib import pyplot as PLT
# # fig = PLT.figure()
#
# import seaborn as sns
# sns.set_context('talk')
# corr = data_categorical.corr()
#
# sns.heatmap(corr)
#
# from sklearn.decomposition import PCA
#
# p = PCA(n_components=2)
#
# pd = p.fit_transform(data_categorical)
#
# plt.scatter(pd[:,0],pd[:,1])