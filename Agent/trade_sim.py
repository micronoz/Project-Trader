
# coding: utf-8

# In[13]:


import pandas as pd
import os
import itertools
import numpy as np


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
                    else:
                        continue
                else:
                    if os.path.isfile(dataPath + '/' + pair + '.csv'):
                        df[pair] = (pd.read_csv(dataPath + '/' + pair + '.csv', delimiter='\t', 
                                                usecols=['Timestamp', 'Open', 'High', 'Low', 'Close']))
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
        for i in range(1, len(self.currencies)):
            current = self.currencies[i]
            pair = self.referenceCurrency + current
            if pair in self.df.keys():
                index = []
                offSet = 0
                while len(index) == 0:
                    index = self.df[pair].loc[self.df[pair]['Timestamp'] == self.allDates[dateIndex + offSet]].index.values
                    offSet += 1
                initialPrice = 1/self.df[pair].loc[index[0], 'Open']
                finalPrice = 1/self.df[pair].loc[index[0] + 1, 'Open']
                rates[i, 0] = finalPrice/initialPrice
            else:
                pair = current + self.referenceCurrency
                index = []
                offSet = 0
                while len(index) == 0:
                    index = self.df[pair].loc[self.df[pair]['Timestamp'] == self.allDates[dateIndex + offSet]].index.values
                    offSet += 1
                initialPrice = self.df[pair].loc[index[0], 'Open']
                finalPrice = self.df[pair].loc[index[0] + 1, 'Open']
                rates[i, 0] = finalPrice/initialPrice
        print(rates)
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
    
    def processTimePeriod(self, resultQ, timePeriod, lastDate, startIndex, size):
        allPrices = []
        allRates = []
        dimensions = ['Open', 'High', 'Low']
        count = 0
        for timeIndex in range(startIndex, (startIndex + size) if (startIndex + size) < len(lastDate) else len(lastDate)-1):
            count += 1
            
            m = 0
            absoluteValue = 0
            priceMatrix = np.zeros(shape=(len(self.currencies), timePeriod, 3))
            restart = False
         
            for currency in self.currencies:
                dimension = -1
                for dimensionName in dimensions:
                    dimension += 1
                    if currency == self.reference:
                        priceMatrix[m, :, dimension] = 1
                    else:
                        if currency + self.referenceCurrency in self.df.keys():
                            pair = currency + self.referenceCurrency
                        elif self.referenceCurrency + currency in self.df.keys():
                            pair = self.referenceCurrency + currency
                        else:
                            raise ValueError('Currency does not exist.')
                        
                        i = self.df[pair].loc[self.df[pair]['Timestamp'] == lastDate[timeIndex]].index.values
                        
                        if len(i) > 1:
                            raise NameError('More than one matching date found!')
                        elif len(i) == 0:
                            timeOffset = 1
                            while len(i) != 1 and timeIndex + timeOffset < len(lastDate):
                                i = self.df[pair].loc[self.df[pair]['Timestamp'] == lastDate[timeIndex + timeOffset]].index.values
                                timeOffset += 1
                        if len(i) == 1:
                            timeOffset = 0
                            index = int(i[0])
                            if index-timePeriod+1 < 0:
                                allPrices.append(None)
                                restart = True
                                break
                        openValues = self.df[pair].iloc[index-timePeriod+1:index+1,
                                                        self.df[pair].columns.get_loc(dimensionName)].values
                        #print(self.df[pair].iloc[index-timePeriod+1:index+2,
                         #                               self.df[pair].columns.get_loc(dimensionName)].values)
                        if dimensionName == 'Open':
                            absoluteValue = openValues[-1]
                        openProcessed = openValues / absoluteValue
                        
                        priceMatrix[m, :, dimension] = openProcessed
                m += 1
                if restart == True:
                    break
            if restart == False:
                allPrices.append(priceMatrix)
                allRates.append(self.getRates(timeIndex))
        resultQ.put((allPrices, allRates))
        print('Done')
        return

