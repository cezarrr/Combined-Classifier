__author__ = 'CJank'

"""
Loading module
"""


import numpy as np
import pylab as pl
import csv

def loadData(fileName, csvDelimiter=','):
    dataSet = []
    with open(fileName, 'rb') as csvFile:
        try:
            csvR = csv.reader(csvFile, delimiter=csvDelimiter, quotechar='"')
            for row in csvR:
                if row:
                    rowParsed = []
                    for column in row:
                        try:
                            columnParsed = float(column)
                        except:
                            columnParsed = column
                        rowParsed.append(columnParsed)
                    dataSet.append(rowParsed)
        finally:
            csvFile.close()
    return dataSet

if __name__ == "__main__":
    print(__doc__)
    file = "C:\Users\CJank\Desktop\Dyskretyzator\Datasets\iris.data"
    X=loadData(file)
    print X