
# coding: utf-8

# In[13]:


import pandas as pd
import os
import itertools
import numpy as np
import time


# In[15]:


class Market:
    def __init__(self, currencies, dataPath, referenceCurrencyM='USD', 
                 initialValue=1000, transactionFee=0.005, initialTime=None, timeframe=50):
        self.path = dataPath
        self.currencies = []
        self.value = initialValue
        self.referenceCurrency = referenceCurrencyM
        self.reference = referenceCurrencyM
        self.fee = transactionFee
        self.time = initialTime #Fill in
        self.currencies = (currencies)
        self.currencies.remove(self.reference)
        self.currencies.insert(0, self.reference)
        self.m = len(self.currencies)
        self.majorPairs = ['EURUSD', 'GBPUSD']
        self.portfolio = None
        pairs = list(itertools.permutations(currencies, 2))
        self.df = self.importFile(pairs)
        self.timeframe = timeframe
        self.initPortfolio()
        self.fillInData()
        self.allDates = self.getAllDates()
        
    def importFile(self, pairs):
        df = {}
        dataPath = self.path
        availableFiles = os.listdir(dataPath)
        for pairTuple in pairs:
            pair = pairTuple[0] + pairTuple[1]
            if (pair + '.csv') in availableFiles:
                if dataPath.endswith('/'):
                    if os.path.isfile(dataPath + pair + '.csv'):
                        df[pair] = pd.read_csv(dataPath + pair + '.csv', delimiter='\t', 
                                                usecols=['Timestamp', 'Open', 'High', 'Low', 'Close'])
                        num = df[pair]._get_numeric_data()
                        num[num < 0] = 1
                    else:
                        continue
                else:
                    if os.path.isfile(dataPath + '/' + pair + '.csv'):
                        df[pair] = (pd.read_csv(dataPath + '/' + pair + '.csv', delimiter='\t', 
                                                usecols=['Timestamp', 'Open', 'High', 'Low', 'Close']))
                        num = df[pair]._get_numeric_data()
                        num[num < 0] = 1
                    else:
                        continue
        return df
    
    def getPortfolioSize(self):
        return self.m
    
    def getPairs(self):
        return self.pairs
    
    def initPortfolio(self):
        np.random.seed(5)
        self.portfolio = np.random.rand(len(self.currencies), 1)
        summation = np.sum(self.portfolio)
        self.portfolio = np.divide(self.portfolio, summation)
        self.portfolio = np.round(self.portfolio, 2)
        for i in range(self.timeframe):
            self.timestep()
        
    def reallocate(self, currencyAllocation):
        exchange = np.subtract(currencyAllocation, self.portfolio)
        oldPortfolio = self.portfolio.copy()
        while (exchange == 0.0).all() != True:
            minIndex = np.unravel_index(np.argmin(exchange, axis=None), exchange.shape)
            maxIndex = np.unravel_index(np.argmax(exchange, axis=None), exchange.shape)
            self.exchangeCurrencies(minIndex, maxIndex, min(abs(exchange[minIndex]), exchange[maxIndex]))
            saveValue = exchange[minIndex]
            exchange[minIndex] = exchange[maxIndex] + exchange[minIndex]
            exchange[maxIndex] = saveValue + exchange[maxIndex]
            exchange = np.round(exchange, 2)

    def incrementTime(self):
        currentTime = str(self.getCurrentTime())
        currentIndex = self.df[self.majorPairs[0]].loc[self.df[self.majorPairs[0]]['Timestamp'] 
                                                       == currentTime].index.values[0]
        self.time = str(self.df[self.majorPairs[0]]['Timestamp'].loc[currentIndex + 1])
        
    def timestep(self):
        if self.time is None:
            for pair in self.majorPairs:
                if pair in self.df.keys():
                    self.time = self.df[pair]['Timestamp'].loc[0]
        else:
            self.incrementTime()
    
    def getCurrentTime(self):
        return self.time
    
    def calculateValue(self):
        totalValue = self.value
        priceChange = self.getRates(self.getCurrentTime())
        newValue = np.sum(np.multiply(self.portfolio, priceChange))
        self.value *= newValue
        
    def exchangeCurrencies(self, source, target, amount):
        self.portfolio[source] -= amount
        self.portfolio[target] += amount
        self.value -= amount * self.fee * self.value
    
    def getRates(self, dateIndex):
        rates = np.zeros(shape=(len(self.currencies), 1))
        rates[0,0] = 1
#         print(dateIndex)
        for i in range(1, len(self.currencies)):
            current = self.currencies[i]
            pair = self.referenceCurrency + current
#             print(pair)
            if current == self.referenceCurrency:
                continue
            if pair in self.df.keys():
                index = dateIndex - 1
                dividers = self.df[pair].iloc[index:index+2]['Open'].values
#                 initialPrice = 1/self.df[pair].loc[index[0], 'Open']
#                 finalPrice = 1/self.df[pair].loc[index[0] + 1, 'Open']
                initialPrice = dividers[0]
                finalPrice = dividers[1]
                rates[i, 0] = 1/(finalPrice/initialPrice)
            else:
                pair = current + self.referenceCurrency
                index = dateIndex - 1
                dividers = self.df[pair].iloc[index:index+2]['Open'].values
#                 initialPrice = self.df[pair].loc[index[0], 'Open']
#                 finalPrice = self.df[pair].loc[index[0] + 1, 'Open']
#                 print(dividers)
#                 print(self.df[pair])
                initialPrice = dividers[0]
                finalPrice = dividers[1]
                rates[i, 0] = finalPrice/initialPrice
        return rates
    
    def fillInData(self):
        referenceFrame = self.df[self.majorPairs[0]]
        for pair in self.df.keys():
            if pair != self.majorPairs[0]:
                editFrame = self.df[pair]
                firstDate = editFrame.loc[0, 'Timestamp']
                index = referenceFrame.loc[referenceFrame['Timestamp'] == firstDate].index.values
                i = 1
                while len(index) == 0:
                    index = referenceFrame.loc[referenceFrame['Timestamp'] == editFrame.loc[i, 'Timestamp']].index.values
                    i += 1
                extracted = referenceFrame.iloc[:index[0]].copy()
                if extracted.size == 0:
                    continue
                startValue = editFrame.loc[0, 'Open']
                amplify = (index[0]/250000)
                extracted['Open'] = np.fromfunction(lambda i, j: (1/amplify)*startValue + i*((1-(1/amplify))*startValue/index[0]), shape=(index[0], 1))
                extracted['Close'] = np.fromfunction(lambda i, j: (1/amplify)*startValue + i*((1-(1/amplify))*startValue/index[0]), shape=(index[0], 1))
                extracted['High'] = np.fromfunction(lambda i, j: (1/amplify)*startValue + i*((1-(1/amplify))*startValue/index[0]), shape=(index[0], 1))
                extracted['Low'] = np.fromfunction(lambda i, j: (1/amplify)*startValue + i*((1-(1/amplify))*startValue/index[0]), shape=(index[0], 1))
                self.df[pair] = pd.concat([extracted, self.df[pair]])
                self.df[pair] = self.df[pair].reset_index(drop=True)
                
    def getAllDates(self):
        return self.df[self.majorPairs[0] if self.majorPairs[0] in self.df.keys() else self.majorPairs[1]].loc[:, 'Timestamp'].values
    
    def processTimePeriod(self, resultD, timePeriod, lastDate, startIndex, size, prevIndex):
        #now = time.time()
        allPrices = np.zeros(shape=(size, len(self.currencies), timePeriod, 3))
        allRates = np.zeros(shape=(size, len(self.currencies), 1))
        dimensions = ['Open', 'High', 'Low']
        indices = (startIndex, (startIndex + size) if (startIndex + size) < len(lastDate) else len(lastDate)-1)
        m = 0
        absoluteValue = 0
        #priceMatrix = np.zeros(shape=(len(self.currencies), timePeriod, 3))
        restart = False
        prevIndices = prevIndex
        for currency in self.currencies:
            first = True
            
            if currency + self.referenceCurrency in self.df.keys():
                pair = currency + self.referenceCurrency
            elif self.referenceCurrency + currency in self.df.keys():
                pair = self.referenceCurrency + currency
            elif self.referenceCurrency == currency:
                count = 0
                for i in range(size):
                    allPrices[count, m, :, :] = 1
                    count += 1
                m += 1
                continue
            else:
                raise ValueError('Currency does not exist.')
            index = 0
            indexOffset = -1
            if prevIndices[m] < 0:
                while index-timePeriod+1 < 0:
                    indexOffset += 1
                    i = self.df[pair].loc[self.df[pair]['Timestamp'] == lastDate[indices[0] + indexOffset]].index.values
                    if len(i) >= 1:
                        index = int(i[0])

                if len(i) > 1:
                    raise NameError('More than one matching date found!')

                index = int(i[0])
                prevIndices[m] = index + size
            else:
              
                index = prevIndices[m]
                prevIndices[m] += size
            if index-timePeriod < 0:
                allPrices.append(None)
                restart = True
                raise ValueError('Wrong index')
                break
            deviation = 0
            batchValues = self.df[pair].iloc[index-timePeriod + deviation:index + size - 1 + deviation, 1:4].values
#             print(batchValues)
#             print(index-timePeriod)
#             print('Upper limit is {}'.format(index+size-1))
            count = 0
            for i in range(size):
                openValues = batchValues[i:timePeriod+i]
            #print(self.df[pair].iloc[index-timePeriod+1:index+2,
             #                               self.df[pair].columns.get_loc(dimensionName)].values)
                absoluteValue = openValues[-1][0]
                openProcessed = openValues / absoluteValue
                allPrices[count, m, :, :] = openProcessed
#                 print('Rate limit is {}'.format(index  + i))
                allRates[count] = (self.getRates(index  + i))
                if np.any(allRates[count] < 0):
                    print("ERROR")
                    print(currency)
                    print(index-timePeriod)
                    print(index+size-1)
                    print(index+i)
                count += 1
            if restart == True:
                break
            if restart == False:
                m += 1
        resultD.append((allPrices, allRates))
        #later = time.time()
        #print("Time for batch:{} seconds".format(int(later-now)))
        return prevIndices

