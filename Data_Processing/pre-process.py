
# coding: utf-8

# In[129]:


import pandas as pd
import zipfile
import os
import shutil


# In[130]:


currencies = ['EURUSD', 'USD', 'TRY', 'USDJPY', 'USDTRY']
dirs = os.listdir('./Data/')
years = []
for word in dirs:
    if '20' in word:
        years.append(word)
availableCurrencies = set()
for year in years:
    for elements in os.listdir('./Data/' + year): 
        if elements in currencies:
            availableCurrencies.add(elements)
print(availableCurrencies)


# In[131]:


pairs = []
for pair in availableCurrencies:
    if pair in currencies:
        pairs.append(pair)


# In[132]:


class PairData:
    def __init__(self, name, tick:str):
        self.name = name
        self.data = tick
        self.month = {"01":None, "02":None,"03":None, "04":None, "05":None, 
                      "06":None, "07":None, "08":None, "09":None, "10":None,
                      "11":None, "12":None}
        self.year = dict()
    def addMonth(self, framePath):
        index = framePath.find('_20')
        yearSelect = framePath[index+1 : index + 5]
        monthSelect = framePath[index + 5 : index + 7]
        if monthSelect not in self.month.keys():
            return
        if yearSelect not in self.year.keys():
            self.year[yearSelect] = self.month.copy()
        self.year[yearSelect][monthSelect] = pd.read_csv(framePath, header=None, delimiter=';')
    def getMonth(self, monthIn, yearIn):
        return self.year[yearIn][monthIn]
    def getHistory(self):
        return self.year


# In[133]:


if not os.path.exists('./ProcessedData/'):
    os.makedirs('./ProcessedData/')


# In[134]:


def unzipFiles(pathName):
    files = os.listdir(pathName)
    unzipName = pathName + 'Unzip'
    if not os.path.exists(unzipName):
        os.makedirs(unzipName)
    doneFile = unzipName + '/Done.txt'
    if not os.path.exists(doneFile):
        f = open(doneFile, 'w')
        for file in files:
            print(file)
            if 'ASCII' in file and '.zip' in file:
                zip_file = zipfile.ZipFile(pathName + file, 'r')
                zip_file.extractall(unzipName + '/')
                zip_file.close()
        unzippedFiles = os.listdir(unzipName + '/')
        tickFolder = unzipName + '/' + 'Tick/'
        minuteFolder = unzipName + '/' + 'Minute/'
        if not os.path.exists(tickFolder):
            os.makedirs(tickFolder)
        if not os.path.exists(minuteFolder):
            os.makedirs(minuteFolder)
        for file in unzippedFiles:
            if '.csv' in file:
                if '_T_' in file:
                    shutil.move(unzipName + '/' + file, tickFolder + '/' + file)
                if '_M1_' in file:
                    print(file)
                    shutil.move(unzipName + '/' + file, minuteFolder + '/' + file)


# In[135]:


#Pre-process data from HistData.com. Unzips files and arranges in folders.
for pair in pairs:
    found = False
    for year in years:
        pathName = './Data/' + year + '/' + pair + '/'
        if os.path.isdir(pathName):
            found = True
            unzipFiles(pathName)
    if not found:
        pairs.remove(pair)


# In[136]:


#Scan the arranged files and insert into a PairData object
dataFrames = {}
for year in years:
    print(year)
    for pair in pairs:
        print(pair)
        if pair not in dataFrames.keys():
            dataFrames[pair] = PairData(pair, 'min')
        currentPair = dataFrames[pair]
        pathName = './Data/' + year + '/' + pair + '/'
        if not os.path.isdir(pathName):
            continue
        files = os.listdir(pathName)
        unzipName = pathName + 'Unzip'
        unzipName += '/Minute'
        frames = os.listdir(unzipName)
        for frame in frames:
            if 'DAT' in frame:
                currentPair.addMonth(unzipName + '/' + frame)


# In[137]:


frameCollections = dict()

for key, value in dataFrames.items():
    currentPair = dataFrames[key].getHistory()
    concatFrames = []
    frameCollections[key] = concatFrames
    for yearkey in sorted(currentPair.keys()):
        for monthkey in sorted(currentPair[yearkey].keys()):
            if currentPair[yearkey][monthkey] is not None:
                concatFrames.append(currentPair[yearkey][monthkey])


# In[138]:


for key in frameCollections.keys():
    currentPair = frameCollections[key]
    for items in currentPair:
        df = pd.concat(currentPair)
    pd.set_option('display.max_columns', 30)
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Vol']
    df = df.drop(columns=['Vol'])
    df.reset_index(drop=True, inplace=True)
    file_name = './ProcessedData/' + key + '.csv'
    df.to_csv(file_name, sep='\t')

