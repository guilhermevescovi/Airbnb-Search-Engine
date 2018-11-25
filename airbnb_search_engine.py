from os.path import isdir
from os import mkdir
import pandas as pd
import numpy as np
import csv
import math
from os import listdir
from os.path import isfile
from os import remove
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from geopy import distance
import folium
import os
from heapq import nlargest
from nltk import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import string
import pickle
import inflect



class AirBnbPy():

    def __init__(self, data_dir_path="./data/",all_reviews_dir = "allReviews/"):
        self.dir_path = data_dir_path
        #If the path given as input doesn't exist on the filesystem
        #it is created
        if not isdir(self.dir_path):
            mkdir(self.dir_path)

        self.review_dir = all_reviews_dir

        #Initialize the data
        self.data = None

        #If the path given as input doesn't exist on the filesystem
        #it is created
        if not isdir(self.dir_path+self.review_dir):
            mkdir(self.dir_path+self.review_dir)

        #Lazy initialization of nltk objects for preprocessing
        self.tokenizer = None
        self.stopwords = None
        self.stemmer = None
        self.number_to_words = None

        #Initialize list which will contain indexes of file with chinese characters
        self.non_english = []

        #Before saving the files on the file system, the preprocessing steps are performed
        #Store in two variables the indexes of the input list which reference to the
        #'description' and 'title' attributes.
        self.DESCRIPTION = 4
        self.TITLE = 7

        #Initialize an empty dictionary which will contain the integer encoding
        #of each word
        self.term_enc = {}
        self.VOCABULARY_SIZE = 1

        #Initialization of the invertex indexing
        self.RI = {}

        #Initialization of corpus and indexToIndex
        self.corpus = None
        self.indexToindex = None

        #default fileName for the tfidf object
        self.tfIdfFileName = "tfIdfMatrix.pickle"
        #default fileName for vectorizer
        self.tfIdfVectorizerFileName = "vectorizer.pickle"
        #default fileNames for corpus and indexToindex
        self.corpusFileName = "corpus.pickle"
        self.indexToindexFileName = "indexToindex.pickle"
        #default fileName for the term encoding of tfIdf
        self.term_enc2FileName = "term_encodingTFIDF.pickle"

        #Initialization of term_encoding for the cosine similarity part
        self.term_enc2 = None

        #Initialization of the inverted index for the Ranked Search Engine
        self.RI2 = None
        #default fileName for inverted index for ranking records
        self.RI2FileName = "RI2.pickle"

        #Initialization of tfidf matrix
        self.tfIdfMatrix = None
        self.vectorizer = None

        #Need this variable to compare the nan value while analyzing the
        #'bedrooms_count' field.
        self._nanFloat = None

        #fit for all solution
        #weight_price = 0.15
        #weight_bed = 0.25
        #weight_loc = 0.6
        self._FFA = 0
        self._weightsFFA = [0.15,0.25,0.6]
        #price oriented
        #weight_price = 0.7
        #weight_bed = 0.1
        #weight_loc = 0.2
        self._weightsPrice = [0.7, 0.1, 0.2]
        #location oriented
        #weight_price = 0.1
        #weight_bed = 0.2
        #weight_loc = 0.7
        self._weightsLoc = [0.075, 0.175, 0.75]
        self.allWeights = [self._weightsFFA, self._weightsPrice, self._weightsLoc]


    def loadData(self):
        #Hardcoded the file name of the dataset
        dataFilename = "Airbnb_Texas_Rentals.csv"
        #A pandas.DataFrame is returned
        #Given the structure of the dataset, having the index encoded as a column
        #So the parameter 'index_col' is specified
        self.data = pd.read_csv(self.dir_path+dataFilename, index_col = "Unnamed: 0", encoding = 'utf8')
        #After the data are uploaded, it is possible to create the auxiliary variable
        #needed to compare the 'bedrooms_count' field while analyzing the data
        self._nanFloat = list(set(self.data['bedrooms_count'].tolist()))[:1]
        #invoke the function to clean the data
        self._cleanData()
        return self.data

    def _cleanData(self):
        #Invoking the code line "data.isnull().sum()", it is possible to observe
        #the presence of 34 NA value in the latitude,longitude columns.
        #
        #It is not possible to retain those houses, since it's not possible to
        #locate them which it is a fundamental info for users
        self.data = self.data[pd.notnull(self.data['latitude'])]
        #If the code line "data.isnull().sum()", it's newly invoked, it is possible
        #to observe that there are 2 NA values in the "description" column
        #and 3 NA values in the "title" column.
        #
        #Since the work requirements ask to realize the search engine on the "description"
        #and the "title" column, all rows which have both columns equal to NA will be
        #dropped.
        self.data = self.data.dropna(subset = ['description','title'], how = 'all')
        #The last check to be done if row duplicates exist
        #If so delete them and leave only one copy
        self.data = self.data.drop_duplicates()
        #TODO:
        #Check if the (lat,long) coordinates refer to the city and don't point
        #to another part of the world.
        #
        #TODO:
        #Check if the URL is still valid.
        #
        #So far, two assumptions are made regarding these things

    def _createTsv(self,x):

        x[self.DESCRIPTION] = "NaN" if pd.isna(x[self.DESCRIPTION]) else self._nltkProcess(x[self.DESCRIPTION])
        x[self.TITLE] = "NaN" if pd.isna(x[self.TITLE]) else self._nltkProcess(x[self.TITLE])
        for fieldString in x:
            if not self._isEnglish(str(fieldString)):
                return

        with open(self.dir_path+self.review_dir+"doc_"+str(x.name)+".tsv", 'w') as file:
            #Need to express delimiter as "\t" since the requested format is a .tsv
            #Was noticed strings in descriptions with foreign language characters which was not
            #possible to encode while writing tsv file so the try/except was added to skip this
            #house
            try:
                wr = csv.writer(file, quoting=csv.QUOTE_ALL, delimiter='\t')
                wr.writerow(x)
            except:
                self.non_english.append(x.name)

    def _nltkProcess(self, string):

        #Transform all words to lowercase
        string = string.lower()
        #Setup nltk objects to perform preprocessing
        self._setupNltk()
        #Tokenize the string removing puntuactions
        tokens = self.tokenizer.tokenize(string)
        #Create new sentence
        new_sentence = []
        #Scroll through each word and stemming it
        for word in tokens:
            word = self.stemmer.stem(word)
            #exclude the word if it is a stopword
            if not word in self.stopwords:
                #if the word has length greater than one, it has sufficient information
                #value to be added
                if len(word) > 1:
                    new_sentence.append(word)
                #if the word length is equal to one and it is numeric
                #then the string representation of the number is added
                elif word.isnumeric():
                    new_sentence.append(self.number_to_words.number_to_words(word))
        #Since the object must later be saved on a .tsv file,
        #it is needed to return a string rather than a list of words
        return " ".join(new_sentence)

    def _setupNltk(self):
        #Lazy initialization of objects needed to preprocess strings
        if self.tokenizer == None:
            self.tokenizer = RegexpTokenizer(r'\w+')
        if self.stopwords == None:
            self.stopwords = set(stopwords.words('english'))
        if self.stemmer == None:
            self.stemmer = SnowballStemmer('english')
        if self.number_to_words == None:
            self.number_to_words = inflect.engine()

    def createAllReviews(self):
        self.data.apply(lambda x: self._createTsv(x), axis = 1)
        #remove all files with chinese characters
        [remove(self.dir_path+self.review_dir+"doc_"+str(index)+".tsv") for index in self.non_english]

    #Function needed to check the correct language
    def _isEnglish(self,s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except:
            #print('not english'+s)
            return False
        else:
            return True

    def buildEncoding(self,fileName):

        #If the file already exists then we load it, instead to compute it another time.
        if isfile(self.dir_path+fileName):
            with open(self.dir_path+fileName, 'rb') as handle:
                self.term_enc = pickle.load(handle)
                self.VOCABULARY_SIZE = len(list(self.term_enc.keys()))
            return

        #Iterate over each .tsv file
        for filePath in listdir(self.dir_path+self.review_dir):
            data = pd.read_csv(self.dir_path+self.review_dir+ filePath, delimiter = '\t',header = None, encoding = 'utf8')
            title = set(str(data.values[0][self.TITLE]).strip().split(' '))
            description = set(str(data.values[0][self.DESCRIPTION]).strip().split(' '))
            concatenatedwords = title.union(description)
            #for each word contained in the title and description fields
            for word in concatenatedwords:
                #if the word is not encoded yet
                if word not in self.term_enc.keys():
                    #store the encoding into the @term_enc dictionary
                    self.term_enc[word] = self.VOCABULARY_SIZE
                    #update the vocabulary size variable
                    self.VOCABULARY_SIZE += 1

        #In the end, the dictionary is saved on the filesystem to be loaded further.
        with open(self.dir_path+fileName, 'wb') as handle:
            pickle.dump(self.term_enc, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def createRevertedIndex(self, fileName):

        #If the file already exists then we load it, instead to compute it another time.
        if isfile(self.dir_path+fileName):
            with open(self.dir_path+fileName, 'rb') as handle:
                self.RI = pickle.load(handle)
            return

        #Each word ID is associated to a list of docs which contains it
        #At the beginning is empty of course
        for i in range(1,self.VOCABULARY_SIZE+1):
            self.RI[i] = list()

        #For each .tsv file contained the records of the original dataset
        for filePath in listdir(self.dir_path+self.review_dir):
            #The file is open through Pandas remembering that since it is a .tsv file,
            #the delimiter will be a tab character and the header is not present
            data = pd.read_csv(self.dir_path+self.review_dir+ filePath, delimiter = '\t',header = None, encoding = 'utf8')

            #Now all words from the 'title' and 'description' fields are extracted
            #and concatenated in a set since we are not interested in repetition
            #of the same doc
            title = set(str(data.values[0][self.TITLE]).strip().split(' '))
            description = set(str(data.values[0][self.DESCRIPTION]).strip().split(' '))
            concatenatedwords = title.union(description)
            #for each word contained in the title and description fields
            for word in concatenatedwords:
                #the previous list of docs associated to the word is retrieved
                mocklist = self.RI[self.term_enc[word]]
                #It is updated with the new value
                mocklist.append(filePath[:-4])
                #Finally assigned in the reverted index data structure
                self.RI[self.term_enc[word]] = mocklist
                #the mockllist is deleted since it is no useful anymore
                del mocklist

        #The reverted index is saved on the filesystem for further uses.
        with open(self.dir_path+fileName, 'wb') as handle:
            pickle.dump(self.RI, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def query(self,queryString):
        #Eventually setup the nltk enviornment
        self._setupNltk()
        #Process the query as we have processed the dataset
        q = self._nltkProcess(queryString).split(' ')
        #list of docs for each word
        result = list()
        #For each word in the query
        for word in q:
            #If the word isn't encoded then it isn't present in any document
            #so we can return no result.
            if word not in self.term_enc.keys():
                print("No results available")
                return None
            #otherwise we append the set of the document to the list
            result.append(set(self.RI[self.term_enc[word]]))

        tmp = set.intersection(*result)
        #initialize the dataframe which will contain the final result of the query
        result = pd.DataFrame(columns = ["title","description", "city", "url"])

        #for each doc to be retrieved
        for doc in tmp:
            #retrieve the index of the record (from doc_xxx to xxx)
            index = int(doc[4:])
            #retrieve the row
            row = self.data.loc[index]
            #Append the row to the final result
            result = result.append(row[["title","description", "city", "url"]])

        return result, q

    def _buildCorpus(self):

        #If the variables are not yet initialized and the file names have not been specified
        #the variables are created and saved into memory.
        if self.corpus == None and self.indexToindex == None:
            #If the files exist then they are loaded, instead to compute them again.
            if isfile(self.dir_path+self.corpusFileName) and isfile(self.dir_path+self.indexToindexFileName):
                with open(self.dir_path+self.corpusFileName, 'rb') as handle:
                    self.corpus = pickle.load(handle)
                with open(self.dir_path+self.indexToindexFileName, 'rb') as handle:
                    self.indexToindex = pickle.load(handle)
                print("[LOG]: Corpus and indexToindex have been correctly loaded from the memory")
                return

            #Otherwise they are newly created
            self.corpus = list()
            #This dictionary maps the index position inside the corpus list to
            #the index of the .tsv file
            self.indexToindex = dict()
            #For each .tsv file containing the records of the original dataset
            for filePath in tqdm(listdir(self.dir_path+self.review_dir)):
                #The file is open through Pandas remembering that since it is a .tsv file,
                #the delimiter will be a tab character and the header is not present
                data = pd.read_csv(self.dir_path+self.review_dir+ filePath, delimiter = '\t',header = None, encoding = 'utf8')

                #retrieve the title and the description fields
                title = str(data.values[0][self.TITLE]).strip()
                description = str(data.values[0][self.DESCRIPTION]).strip()

                #concatenate them in a string
                document = title + " " + description
                #append the document to the corpus
                self.corpus.append(document)
                #store the index association
                self.indexToindex[len(self.corpus)-1] = int(filePath[4:][:-4])

            #Save the variables on the filesystem
            with open(self.dir_path+self.corpusFileName, 'wb') as handle:
                pickle.dump(self.corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.dir_path+self.indexToindexFileName, 'wb') as handle:
                pickle.dump(self.indexToindex, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return

        #If they are both initialized all went fine.
        print("[LOG]: corpus and indexToindex are already loaded")

    def _tfIdfSetup(self):
        #Check if the tfidf matrix exists.
        #If not then, if the file exists, it is loaded from the filesystem
        #otherwise it is created and stored on the filesystem
        if self.tfIdfMatrix == None and self.vectorizer == None:
            #If the file already exists then we load it, instead to compute it another time.
            if isfile(self.dir_path+self.tfIdfFileName) and isfile(self.dir_path+self.tfIdfVectorizerFileName):
                with open(self.dir_path+self.tfIdfFileName, 'rb') as handle:
                    self.tfIdfMatrix = pickle.load(handle)
                with open(self.dir_path+self.tfIdfVectorizerFileName, 'rb') as handle:
                    self.vectorizer = pickle.load(handle)
                print("[LOG]: The tfidf matrix and tfidf vectorizer have been correctly loaded from the memory.")
            else:
                print("[LOG]: The file "+self.tfIdfFileName+" doesn't exist")
                print("[LOG]: A new tfidf matrix will be built and saved in persistent memory with the name:= "+self.tfIdfFileName)
                self.vectorizer = TfidfVectorizer()
                self.tfIdfMatrix = self.vectorizer.fit_transform(self.corpus)
                #Save the variables on the filesystem
                with open(self.dir_path+self.tfIdfFileName, 'wb') as handle:
                    pickle.dump(self.tfIdfMatrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.dir_path+self.tfIdfVectorizerFileName, 'wb') as handle:
                    pickle.dump(self.vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return

        print("[LOG]: The tfidf matrix and tfidf vectorizer are already loaded.")

    def _termEnc2setup(self):
        #setup for term encoding
        if self.term_enc2 == None:
            #If the file already exists then we load it, instead to compute it another time.
            if isfile(self.dir_path+self.term_enc2FileName):
                with open(self.dir_path+self.term_enc2FileName, 'rb') as handle:
                    self.term_enc2 = pickle.load(handle)
                print("[LOG]: The term encoding have been correctly loaded from the memory")
            else:
                print("[LOG]: The file "+self.term_enc2FileName+" doesn't exist")
                print("[LOG]: A new term encoding will be built and saved in persistent memory with the name:= "+self.term_enc2FileName)
                self.term_enc2 = {k: v for v, k in enumerate(self.vectorizer.get_feature_names())}
                #Save the variables on the filesystem
                with open(self.dir_path+self.term_enc2FileName, 'wb') as handle:
                    pickle.dump(self.term_enc2, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return

        print("[LOG]: The term encoding is already loaded")

    def _invertedIndex2setup(self):

        if self.RI2 == None:
            #If the file already exists then we load it, instead to compute it another time.
            if isfile(self.dir_path+self.RI2FileName):
                with open(self.dir_path+self.RI2FileName, 'rb') as handle:
                    self.RI2 = pickle.load(handle)
                print("[LOG]: The inverted index have been correctly loaded from the memory")
                return
            else:
                print("[LOG]: The file "+self.RI2FileName+" doesn't exist")
                print("[LOG]: A new inverted index will be built and saved in persistent memory with the name:= "+self.RI2FileName)
                self.RI2 = {}
                #Each word ID is associated to a list of docs which contains the word.
                for term_key in list(self.term_enc2.keys()):
                    self.RI2[self.term_enc2[term_key]] = list()

                #Iterating over each document
                for doc_index in range(self.tfIdfMatrix.shape[0]):
                    #get the tfidf vector of the document
                    data = self.tfIdfMatrix[doc_index]
                    #the data is a sparse matrix of scipy package
                    cx = data.tocoo()
                    #this for allows to iterate only over the non-zero entries of the data
                    for _,j,v in zip(cx.row, cx.col, cx.data):

                        mocklist = self.RI2[j]
                        #self.indexToindex maps the index value of the document
                        #in the tfidf matrix to the index of the .tsv file
                        mocklist.append(("doc_"+str(self.indexToindex[doc_index]),v))
                        #Finally assigned in the reverted index data structure
                        self.RI2[j] = mocklist
                        #the mockList is deleted since it is no useful anymore
                        del mocklist

                #The reverted index is saved on the filesystem for further uses.
                with open(self.dir_path+self.RI2FileName, 'wb') as handle:
                    pickle.dump(self.RI2, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return
        print("[LOG]: The inverted index is already loaded")

    def _rankedQuerySetup(self):
        #Computing the TFIDF values involve creating the corpus of our dataset.
        #
        #The corpus consists of a collection of documents where each document
        #it's represented by the string obtained concatenating the title and
        #the description field of each record.
        self._buildCorpus()
        #setup for the TfIdf, more details to the function
        self._tfIdfSetup()
        #setup for the term encoding
        self._termEnc2setup()
        #setup for the inverted index
        self._invertedIndex2setup()


    def rankedQuery(self, queryString, k):

        #the dataframe not yet ranked and the query already processed
        (notRanked, q) = self.query(queryString)
        #if the result is empty or it contains only one record
        #it is useless to rank the result
        if notRanked is None or notRanked.shape[0] == 1 :
            return notRanked

        #setup the environment to rank each document with respect to the query
        self._rankedQuerySetup()

        #Ranking the results
        #
        #get the indexes of each document in the intermediate result
        indexes = np.array(notRanked.index)
        #vectorize function to create the new series containing the
        func = np.vectorize(lambda doc_index: self._evaluateCosineSimilarity(query = q,doc_index = doc_index))
        #create the new series containing the cosine similarity
        notRanked["Similarity"] = pd.Series(func(indexes), index = indexes)
        #Return the records sorted for similarity in descending order
        #
        #Since is is needed to use an heap structure, the results are stored in a list
        #and then the list is passed to python "heapq" package functions, which transform
        #the list in an heap structure
        #
        #This list contains tuples which consists in the row_index and the associated similarity value
        tmp = list()
        #Iterating over each row of the query result
        for index, row in notRanked.iterrows():
            #update the list of tuples
            tmp.append((index,row["Similarity"]))

        #the column attributes that must be returned with the result
        resultColumns = ["title","description","city","url","Similarity"]
        resultData = []

        for t in nlargest(k, tmp, key = lambda x:x[1]):
            index = t[0]
            sim_value = t[1]
            data = notRanked.loc[index][resultColumns[:-1]].values.tolist()
            data.append(sim_value)

            resultData.append(data)

        return pd.DataFrame(data = resultData, columns = resultColumns)


    def _evaluateCosineSimilarity(self,query, doc_index):

        denominator = len(query)

        numerator = 0

        for word in set(query):

            numerator += [t[1] for t in self.RI2[self.term_enc2[word]] if t[0] ==  "doc_"+str(doc_index)][0]

        return numerator / math.sqrt(denominator)

    '''
        Step 4: Define a new score!
    '''

    def _similarityBed(self, rooms_requested, rooms_actuals):
        if (rooms_requested <= rooms_actuals):
            return 1
        dif_rooms = rooms_requested - rooms_actuals
        return (1 /(1 + dif_rooms**10))

    def _similarityPrice(self, price_requested, price_actual):
        if (price_requested >= price_actual):
            return 1
        dif_price = price_requested - price_actual
        return (1 /(price_actual/price_requested))

    def _similarityLocation(self, request_location, actual_location):
        return 1/(1+0.1*distance.geodesic(request_location, actual_location, ellipsoid = (6377., 6356., 1 / 297.)).kilometers)


    def _computeSimilarity(self, request, actual, weights):

        denominator = 3

        request_price = request[0]
        actual_price = float(actual[0][1:])
        sim_price = self._similarityPrice(request_price, actual_price)

        actual_bedroom = actual[1]
        request_bedroom = request[1]
        if(actual_bedroom in self._nanFloat):
            denominator -= 1
            sim_bed = 0
        elif(actual_bedroom == 'Studio'):
            actual_bedroom = 1
            sim_bed = self._similarityBed(request_bedroom, actual_bedroom)
        else:
            actual_bedroom = int(actual_bedroom)
            sim_bed = self._similarityBed(request_bedroom, actual_bedroom)

        actual_coordinates = (actual[2], actual[3])
        request_coordinates = (request[2], request[3])
        sim_loc = self._similarityLocation(request_coordinates, actual_coordinates)

        sim =[sim_price,sim_bed,sim_loc]

        res = 0
        for i in range(len(weights)):
            res += (weights[i]*sim[i])

        return res

    def _returnWeightVector(self,weightChoice):
        if( weightChoice < 0 or weightChoice >= len(self.allWeights)):
            print("[LOG]: You have chosen a non-existent choice")
            print("[LOG]: The possible choices are:\n\t[0] Fit-for-all\n\t[1] Price oriented\n\t[2] Location oriented")
            print("[LOG]: The [0] Fit-for-all solution has been selected by default.")
            return self.allWeights[self._FFA]
        return self.allWeights[weightChoice]

    def customQuery(self, textQuery, query, k, weightChoice):

        notRanked,_  = self.query(textQuery)
        #if the result is empty or it contains only one record
        #it is useless to rank the result
        if notRanked is None or notRanked.shape[0] == 1 :
            return notRanked

        #get the indexes of each document in the intermediate result
        indexes = np.array(notRanked.index)
        #Retrieve the weight vector for the similarity function
        weightsVector = self._returnWeightVector(weightChoice = weightChoice)
        #create the new series containing the custom similarity
        #Since we are using the "self.query()" function for a matter of reuse,
        #we now need to deal with the missing features needed to compute the
        #custom score.
        #So we retrieve the data from the original dataset(@self.data) and take only
        #the records of interest(@indexes) and the columns needed to compute the
        #custom similarity.
        columnsOfInterest = ["average_rate_per_night","bedrooms_count","latitude","longitude"]
        notRanked["Similarity"] = pd.Series(data = self.data.loc[indexes][columnsOfInterest].apply(lambda x : self._computeSimilarity(request=query, actual=x.values.tolist(), weights = weightsVector) , axis = 1), index = indexes)
        #Since is is needed to use an heap structure, the results are stored in a list
        #and then the list is passed to python "heapq" package functions, which transform
        #the list in an heap structure.
        #
        #This list contains tuples which consists in the row_index and the associated similarity value
        tmp = list()
        #Iterating over each row of the query result
        for index, row in notRanked.iterrows():
            #update the list of tuples
            tmp.append((index,row["Similarity"]))

        #the column attributes that must be returned with the result
        resultColumns = ["ranking" ,"title","description","city","url","Similarity"]
        resultData = []
        resultIndexes = []
        ranking = 1
        for t in nlargest(k, tmp, key = lambda x:x[1]):
            index = t[0]
            sim_value = t[1]
            data = notRanked.loc[index][resultColumns[1:-1]].values.tolist()
            data = [ranking] + data
            ranking += 1
            resultData.append(data)
            resultIndexes.append(index)

        return pd.DataFrame(data = resultData, columns = resultColumns[:-1], index = resultIndexes)
