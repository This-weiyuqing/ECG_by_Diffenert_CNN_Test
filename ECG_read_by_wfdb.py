import wfdb
import numpy as np


def getDataset(number, X_data):
    # get ECG from mit-bih-arrhythmia-database-1.0.0
    print("get data:" + number + " ECG")
    record = wfdb.rdrecord('./mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten()

    # get R sample and symbol
    annotation = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/' + number, 'atr')
    Rlocatation = annotation.sample
    Rclass = annotation.symbol

    X_data.append(data)

    return


def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    for n in numberSet:
        getDataset(n, dataSet)
        print(dataSet)
    return dataSet

def main():
    dataSet=loadData()
    dataSet=np.array(dataSet)
    print(dataSet.shape)
    print("get ECG data")

if __name__ == '__main__':
    main()