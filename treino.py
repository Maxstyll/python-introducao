import csv
import pandas as pd
import numpy as np
import autokeras as ak

from imutils import paths
from sklearn.model_selection import train_test_split
from datetime import datetime
from src.useful.toolsFoldersFiles import Dir

def getDataFrame(dataSet = None, pathLoadDataSet = None):

    frames = []
    if(dataSet == None):
        if(pathLoadDataSet == None):
            pathLoadDataSet = "dataset"

        dsPaths = list(paths.list_files(pathLoadDataSet, "csv"))
        dsPaths = np.array(sorted(dsPaths))
        frames = [pd.read_csv(dsPath) for dsPath in dsPaths]
    else:
        dsPaths = np.genfromtxt("dataset/dataset-default.csv", delimiter=',')
        frames = [pd.DataFrame(dataSet), pd.DataFrame(dsPaths)]


    for frame in frames:
        columns = ['tag']
        columns += ['col_' + str(col) for col in range(len(frame.columns) -1)]
        frame.columns = columns
        frame['tag'] = pd.Categorical(frame.tag)

    dfPadrao = pd.concat(frames, axis=0) 

    if(dfPadrao.isnull().values.any()):
        dfPadrao = dfPadrao.fillna(0)

    return dfPadrao

def dadosTrainTest(dataSet = None, pathLoadDataSet = None):
    df = getDataFrame(dataSet, pathLoadDataSet)

    x_train, x_test, y_train, y_test = train_test_split(df.drop('tag', axis=1), df.tag, test_size=0.2, random_state=42, shuffle=True)

    # Normalizar os dados de treino e teste
    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.to_numpy()
    x_train = x_train[:, :, np.newaxis]
    y_train = y_train.to_numpy()

    x_test = x_test.to_numpy()
    x_test = x_test[:, :, np.newaxis]
    y_test = y_test.to_numpy()

    return x_train, x_test, y_train, y_test

def trainModel(dataSet = None, pathLoadDataSet = None, pathSaveModel = None, sizeTestModel = 10):
    saveModel = False
    if (dataSet == None):
        saveModel = True

    x_train, x_test, y_train, y_test = dadosTrainTest(dataSet, pathLoadDataSet)

    clf = ak.ImageClassifier(overwrite=True, multi_label=True, max_trials=sizeTestModel)
    clf.fit(x_train, y_train, batch_size=8, validation_split=0.10, epochs=25)

    # Export as a Keras Model.
    model = clf.export_model()

    if(dataSet == None):
        if(pathSaveModel == None):
            pathSaveModel = 'models'

        output = Dir.create(pathSaveModel)
        now = datetime.now()
        nameModel = output + "/model-ass-" + now.strftime("%Y%m%d%H%M%S%f")

        try:
            model.save(nameModel, save_format="tf")
        except:
            model.save(nameModel + ".h5")

        with open(nameModel + "-labes.csv", 'w') as outfile:
            writer = csv.writer(outfile)
            label = np.unique(y_train).tolist()
            writer.writerows(map(lambda x: [x], label))

    return model

if __name__ == "__main__":
    trainModel(pathLoadDataSet="dados/dataset/", pathSaveModel="dados/modelos/", sizeTestModel=20)
