
"""
    Dataset used : MOVIELENS DATA SET OF RATINGS

    This Program contains classes for each of the model of Recommender systems
    1.collaborative
    2.singular_value_decompositon
    3.cur_decomposition

"""
import random
import numpy as np
from numpy import linalg as la
from tabulate import tabulate
import math
import pickle
import heapq
import copy
import time

class collaborative:
    ## All the error functions and logic for collaborative filtering
    def __init__(self,train,ncount,baseline=False):
        self.withbaseline = baseline #Is this using baseline esitmates?
        self.neighbour = ncount
        self.data = train
        self.normalizedData = {}
        self.pearson_correlation(self.data)   ## Normalize the ratings
        self.rmseerror = None
        self.topkerror = None
        self.spearerror = None
        if self.withbaseline == True:
            self.calculate_baselines()  ## calaculating baselines if needed

    def calculate_baselines(self):
        """
            calculate baseline values for the data
        """

        itemAverages = {}
        itemUserCount = {}
        userAverages = {}

        data = self.data
        for user in data:
            sum =0
            count = len(data[user])
            for item in data[user]:
                sum += data[user][item]
                if item in itemAverages:
                    itemAverages[item] += data[user][item]
                    itemUserCount[item] += 1
                else:
                    itemAverages[item] = data[user][item]
                    itemUserCount[item] = 1
            userAverages[user] = sum/count
        #itemcount = len(itemAverages)
        for k in itemAverages:
            sum = itemAverages[k]
            avg = sum/itemUserCount[k]
            itemAverages[k] = avg
        #itemAverages = {k: v / itemUserCount[k] for k, v in itemAverages.items()}
        #l = userAverages.values()
        #l =sum(list(l))
        sumuser = 0
        for value in userAverages:
            sumuser += value
        globalUserAverageRating = sumuser/len(userAverages)

        #globalUserAverageRating = sum(list(userAverages.values()))/len(userAverages)
        self.itemAverages = itemAverages
        self.userAverages = userAverages
        self.overallMean = globalUserAverageRating

    def pearson_correlation(self,data):
        """
            normalizes the dataset using pearson correlation
        """

        #Calculate Average
        for  user in data :
            sum = 0
            count = 0
            ratings = data[user]
            self.normalizedData[user] = {}
            for movie in ratings:
                self.normalizedData[user][movie] = 0
                ratingValue = ratings[movie]
                sum += ratingValue
                count += 1

            average = sum/count

            for movie in ratings:
                self.normalizedData[user][movie] = ratings[movie] -  average

    def cosine_similarity(self,user1,user2):
        """Returns the cosine similarity between user 1 and 2
        """
        v1 = self.normalizedData[user1]
        v2 = self.normalizedData[user2]
        #k1 = self.data[user1]
        #k2 = self.data[user2]
        #if user1 == 283:
        #    print("1",v1,k1)
        #print("2",v2,k2)


        allDim = set(list(v1.keys()) + list(v2.keys()))
        dotproduct = 0
        euclidianDistanceV1 =0
        euclidianDistanceV2 =0

        for dim in v1:
            rating = v1[dim]
            euclidianDistanceV1 += (rating * rating)
        euclidianDistanceV1 =math.sqrt(euclidianDistanceV1)

        for dim in v2:
            rating = v2[dim]
            euclidianDistanceV2 += (rating * rating)
        euclidianDistanceV2 =math.sqrt(euclidianDistanceV2)

        for dim in allDim:
            if dim in v1:
                a = v1[dim]
            else:
                a = 0
            if dim in v2:
                b = v2[dim]
            else:
                b = 0
            dotproduct += (a * b)

        if euclidianDistanceV1 == 0 or euclidianDistanceV2 == 0:
            #print(euclidianDistanceV1,"euclid dist ",euclidianDistanceV2,"vec",v1,v2)
            #print("1",v1,k1)
            #print("2",v2,k2)

            return 0.5                                                                         ####This needed better handling
            #print(euclidianDistanceV1,"euclid dist ",euclidianDistanceV2,"vec",v1,v2)
            #print("val",dotproduct/(euclidianDistanceV1*euclidianDistanceV2))
        else :
            pass
            #print("YESSSSSSSSSSSS")"""

        return (dotproduct/(euclidianDistanceV1*euclidianDistanceV2))

    def get_nsimilar_users(self,userid,movieid):
        """
            Given a userid and movieid

            return: top K users based on cosine_similarity between them
        """
        topSimilarUsers = []
        simValues = {}
        # calculating similarity of current user with all other users
        # who rated the movie under consideration
        for user in self.data:
            #print("userid")
            if movieid in self.data[user].keys(): ## need only those users who rated the movie
                #print("here")
                #print("movie id",userid,",",movieid,",",user)
                sim = self.cosine_similarity(userid,user)
                simValues[user] = sim
                #if math.isnan(sim):
                #    print("movie id",userid,",",movieid,",",user)


        # Top n similar users
        #print(simValues)
        topSimilarUsers = heapq.nlargest(self.neighbour, simValues.keys(),
                                                    key=lambda k: simValues[k])

        #print("topusers",topSimilarUsers)
        topSimilarityValues = {}
        for x in topSimilarUsers:
            topSimilarityValues[x] = simValues[x]
        #print("Top values",topSimilarityValues)

        return topSimilarityValues

    def getBaselineEstimate(self,userid,item):
        """
            returns the baseline esitmate for user and item
        """
        userAvg = self.userAverages[userid]
        globalMean = self.overallMean
        userDelta = globalMean - userAvg
        if item not in self.itemAverages:
            return 0
        itemAvg = self.itemAverages[item]
        itemDelta = globalMean - itemAvg

        baseline = globalMean + userDelta + itemDelta

        return baseline

    def predict_rating_b(self,userid,movieid):
        """Predict rating of the user for a movie
            with baseline values

        """
        similarUsers = self.get_nsimilar_users(userid,movieid)
        weightedRatingSum = 0
        sumOfWeights = 0

        baseline = self.getBaselineEstimate(userid,movieid)
        for simuser in similarUsers:
            simVal = similarUsers[simuser]
            userRating = self.data[simuser][movieid]
            weightedRatingSum += (simVal * (userRating-self.getBaselineEstimate(simuser,movieid)))
            sumOfWeights += simVal

        if len(similarUsers) == 0:
            return round(baseline,1)
        rating = round(baseline+(weightedRatingSum /sumOfWeights),1)

        return rating

    def predict_rating(self,userid,movieid):
        """Predict rating of the user for a movie

        """
        similarUsers = self.get_nsimilar_users(userid,movieid)
        weightedRatingSum = 0
        sumOfWeights = 0

        for simuser in similarUsers:
            simVal = similarUsers[simuser]
            userRating = self.data[simuser][movieid]
            weightedRatingSum += (simVal * userRating)
            sumOfWeights += simVal
        #print("Number of similar",len(similarUsers))
        if len(similarUsers) == 0:
            return 0
        rating = round((weightedRatingSum /sumOfWeights),1)

        return rating

    def rmse_c(self,predicted,actual):
        """Method to calculate the rmse
            between actual and predicted values
        """
        count = 0
        errorsum = 0
        for user in predicted:
            moviesPredicted = predicted[user]
            moviesActual = actual[user]

            #count += len(moviesPredicted)
            for movie in moviesActual:
                count += 1
                error = abs(abs(moviesActual[movie])-abs(moviesPredicted[movie]))
                #print(moviesActual[movie],"<->",moviesPredicted[movie])
                errorsum += error**2
        return (errorsum/count)**0.5

    def get_topk_c(self,d,k):
        """gets the top k ratings over the dataset  """
        users = []
        movies = []
        ratings = []
        for u in d:
            umovies = d[u]
            for m in umovies:
                users.append(u)
                movies.append(m)
                ratings.append(umovies[m])

        indexes = []

        largestRatings = heapq.nlargest(k,ratings)
        largestRatings = set(largestRatings)
        for rating in largestRatings:
            temp =[i for i, x in enumerate(ratings) if x == rating]
            indexes.extend(temp)
        return indexes

    def topk_c(self,predicted,actual,k=300):
        #"error among top k predicted ratings")

        topkActualIndexes = self.get_topk_c(actual,k)
        topkPredictedIndexes = self.get_topk_c(predicted,k)
        topkActual = len(topkActualIndexes)
        topkPredicted = 0

        for i in topkPredictedIndexes:
            if i in topkActualIndexes:
                topkPredicted += 1
        #print("top",topkActual,topkPredicted,len(topkPredictedIndexes))
        return topkPredicted/len(topkPredictedIndexes)

    def spearman_c(self,predicted,actual):
        #"spearman rank
        count = 0
        errorsum = 0
        for key in predicted:
            moviesPredicted = predicted[key]
            moviesActual = actual[key]
            count += len(moviesPredicted)
            for movie in moviesActual:
                error = abs(moviesActual[movie]-moviesPredicted[movie])
                errorsum += error**2

        ro = 1 -((6*errorsum)/(count*(count**2 -1)))
        return ro

    def test(self,tdata):
        #Test the test data
        predicted = {}
        start_time = time.clock()
        for user in tdata:
            userprediction = {}
            umovies = tdata[user]
            for movie in umovies.keys():
                if self.withbaseline == True:
                    pr = self.predict_rating_b(user,movie)
                else:
                    pr = self.predict_rating(user,movie)
                userprediction[movie] = pr

            predicted[user] = userprediction
        end_time = time.clock()
        self.predicted = predicted
        self.rmseerror = self.rmse_c(predicted,tdata)
        self.topkerror = self.topk_c(predicted,tdata)
        self.spearerror = self.spearman_c(predicted,tdata)
        self.time = end_time-start_time
##All the classes and methods for SVD Both cases
class singular_value_decompositon:

    def __init__(self,train_data,users,movies,save = False,saving = 0):
        self.movies = movies
        self.users = users
        self.matrix = self.get_matrix(train_data,users,movies)
        self.rowavg = self.normalize_ratings(self.matrix)  ## making it a zero centered matrix
        self.rows = self.matrix.shape[0]
        self.cols = self.matrix.shape[1]
        start_time = time.clock()
        self.U,self.sigma,self.V =self.get_u_sigma_v(self.matrix)
        if save == True:
            self.reducedim(saving)
        self.svdmatrix = self.multiply_u_sigma_v()
        self.add_averages(self.svdmatrix,self.rowavg)   # adding average values back to the ratings
        self.svdmatrix[self.svdmatrix>5] = 5  # altering the out of range values to extremes
        self.svdmatrix[self.svdmatrix<0] = 0
        end_time = time.clock()
        self.time = end_time-start_time

    def normalize_ratings(self,M):
        """calculaitng user average and mean centering the matrix"""
        rowavg = np.array([0]*len(M))   ## average value of each row i.e the user
        for i in range(len(M)):  ## calculating user averages
            rowsum = 0
            rowcount = 0
            for j in range(len(M[i])):
                if(M[i][j]>0):
                    rowsum = rowsum+ M[i][j]
                    rowcount = rowcount + 1
            if rowcount > 0:
                rowavg[i]= rowsum/rowcount

        for i in range(len(M)):    # subtracting average value for existing ratings, missing ratings are 0
            M[i][M[i]>0] -= rowavg[i]
        return rowavg

    def add_averages(self,M,A):
        """Adding average value to each row"""
        for i in range(len(M)):    # adding row's average value
            M[i] += A[i]

    def reducedim(self,percent):
        """reduce the dimensions of the matrix with given saving value"""
        save = percent/100
        totalEnergy = 0
        for i in range(self.rank):
            totalEnergy += self.sigma[i,i]**2
        retainedEnergy = 0
        seen_count = 0
        for i in range(self.rank):
            retainedEnergy += self.sigma[i,i]**2
            seen_count += 1
            if retainedEnergy >= save*totalEnergy:
                break

        rem_count = len(self.eiganValues) - seen_count
        length = len(self.eiganValues)
        for i in range(rem_count):
            self.sigma=np.delete(self.sigma,length-1,0)
            self.sigma=np.delete(self.sigma,length-1,1)
            self.U=np.delete(self.U,length-1,1)
            self.V = np.delete(self.V,length-1,0)
            length -= 1

    def get_matrix(self,train_data,users,movies):
        #Generates a matrix of the test data

        matrixlist = []
        for user in users:
            userlist=[]
            for movie in movies:
                if user in train_data and movie in train_data[user]:
                    userlist.append(train_data[user][movie])
                else:
                    userlist.append(0)
            matrixlist.append(userlist)
        matrix = np.array(matrixlist)

        return matrix

    def get_eigan_matrix(self,M):
        """Returnss the eigan matrix of M"""
        rank = np.linalg.matrix_rank(M)
        self.rank = rank
        #print("Rank",rank)
        eiganValues, eiganVectors = la.eigh(M)
        eiganDict = {}
        for i in range(len(eiganValues)):
            eiganDict[eiganValues[i]] = np.array(eiganVectors[i])#.flatten().tolist()
        eiganValues.sort()
        eiganValues = eiganValues[::-1]
        eiganmatrixlist = []
        eiganValuesSubset = []
        for i in range(rank):
            value = eiganValues[i]
            if value <= 0:
                break
            eiganValuesSubset.append(value)
            eiganmatrixlist.append(eiganDict[value])
        self.eiganValues =eiganValuesSubset
        #print(len(eiganValuesSubset))
        eiganmatrix = np.array(eiganmatrixlist)
        eiganmatrix = eiganmatrix.transpose()
        return eiganmatrix

    def get_sigma(self):
        values = self.eiganValues
        sigmalist=np.zeros(shape=(self.rank,self.rank),dtype=float)
        for i in range(len(values)):
            sigmalist[i][i] = values[i]**0.5

        return np.array(sigmalist)

    def get_u_sigma_v(self,M):
        #Return svd matrices
        M = np.array(self.matrix)
        Mt = M.T
        MMt = np.dot(M,Mt)#M*Mt
        eiganmatrix_u = self.get_eigan_matrix(MMt)
        sigma = self.get_sigma()
        MtM = np.dot(Mt,M)#Mt*M
        eiganmatrix_v = self.get_eigan_matrix(MtM)
        return eiganmatrix_u,sigma,eiganmatrix_v.transpose()

    def multiply_u_sigma_v(self):
        result = np.dot((np.dot(self.U,self.sigma)),self.V)
        return result

    def rmse(self,predicted,actual):
        """Method to calculate the rmse
            between actual and predicted values
        """
        count = 0
        errorsum = 0
        for i in range(len(predicted)):
            count += 1
            error = abs(abs(predicted[i])-abs(actual[i]))
            errorsum += error**2
        return (errorsum**0.5)/(count**0.5)

    def get_topk(self,actual,k):
        """returns indexes in list which correspond to top ratings  """

        indexes = []
        largestRatings = heapq.nlargest(k,actual)
        #print("count",len(largestRatings))
        largestRatings = set(largestRatings)
        #print("count2",len(largestRatings))
        #print("coun val",(largestRatings))
        for rating in largestRatings:
            temp =[i for i, x in enumerate(actual) if abs(x) == rating]
            indexes.extend(temp)

        return indexes

    def topk(self,predicted,actual,k=100):
        #print("error among top k predicted ratings")
        topkActualIndexes = self.get_topk(actual,k)
        topkPredictedIndexes = self.get_topk(predicted,k)
        ########
        topkActual = len(topkActualIndexes)
        topkPredicted = 0

        for i in topkPredictedIndexes:
            if i in topkActualIndexes:
                topkPredicted += 1
        #print("top",topkActual,topkPredicted,len(topkPredictedIndexes))
        return topkPredicted/len(topkPredictedIndexes)
        #########
        """
        topkActual = []
        topkPredicted = []
        for i in topkActualIndexes:
            topkActual.append(actual[i])
            topkPredicted.append(predicted[i])

        return self.rmse(topkPredicted,topkActual)"""

    def spearman(self,predicted,actual):
        #print("spearman rank")
        count = 0
        errorsum = 0
        for i in range(len(predicted)):
            count += 1
            error = abs(predicted[i]-actual[i])
            errorsum += error**2

        ro = 1 -((6*errorsum)/(count*(count**2 -1)))
        return ro

    def test(self,test_data):
        ##Testing the predicitons and calculate error values
        result = np.round(self.svdmatrix,decimals=1)
        actual = []
        predicted = []
        users = self.users
        movies = self.movies
        for user in test_data:
            for movie in test_data[user]:
                u_idx = users.index(user)
                m_idx = movies.index(movie)
                a_val = test_data[user][movie] ## for user movie actual value
                p_val = result[u_idx][m_idx]
                #print(a_val,"->",p_val)
                actual.append(a_val)
                predicted.append(p_val)

        self.predicted = predicted
        self.rmseerror = self.rmse(predicted,actual)
        self.topkerror = self.topk(predicted,actual)
        self.spearerror = self.spearman(predicted,actual)
#Class for cur decompostion both cases and error functions
class cur_decomposition:
    def __init__(self,train_data,users,movies,save = False,saving = 0):
        self.movies = movies
        self.users = users
        self.save = save
        self.percent = saving
        self.matrix = self.get_matrix(train_data,users,movies)
        self.rowavg = self.normalize_ratings(self.matrix)
        self.rank = la.matrix_rank(self.matrix)
        self.fourk = self.rank       # no of rows
        self.rows = self.matrix.shape[0]
        self.cols = self.matrix.shape[1]
        start_time = time.clock()
        self.colProb,self.rowProb = self.calculate_row_col_stats()
        self.C,self.U,self.R =self.get_c_u_r(self.matrix)
        self.curmatrix = self.multiply_c_u_r()
        self.add_averages(self.curmatrix,self.rowavg)
        self.curmatrix[self.curmatrix>5] = 5          ## adjusting out of range values
        self.curmatrix[self.curmatrix<0] = 0
        end_time = time.clock()
        self.time = end_time - start_time

    def normalize_ratings(self,M):
        """calculaitng user average and mean centering the matrix"""
        rowavg = np.array([0]*len(M))   ## average value of each row i.e the user
        for i in range(len(M)):  ## calculating user averages
            rowsum = 0
            rowcount = 0
            for j in range(len(M[i])):
                if(M[i][j]>0):
                    rowsum = rowsum+ M[i][j]
                    rowcount = rowcount + 1
            if rowcount > 0:
                rowavg[i]= rowsum/rowcount

        for i in range(len(M)):    # subtracting average value for existing ratings, missing ratings are 0
            M[i][M[i]>0] -= rowavg[i]
        return rowavg

    def add_averages(self,M,A):
        """Adding average value to each row"""
        for i in range(len(M)):    # adding row's average value
            M[i] += A[i]

    def get_matrix(self,train_data,users,movies):
        #Generates a matrix of the data dictionary
        matrixlist = []
        for user in users:
            userlist=[]
            for movie in movies:
                if user in train_data and movie in train_data[user]:
                    userlist.append(train_data[user][movie])
                else:
                    userlist.append(0)
            matrixlist.append(userlist)
        matrix = np.array(matrixlist)
        return matrix

    def calculate_row_col_stats(self):
        ##Calculate the probabilities for the rows and columns
        totalLengths = sum(sum(self.matrix**2))
        colLengths = sum(self.matrix**2) #ferbenius norm
        rowLengths = sum(self.matrix.T**2)
        colProb =  colLengths/totalLengths
        rowProb = rowLengths/totalLengths
        return colProb,rowProb

    def get_c(self,size = 513):
        #Randomly sample columns to create matrix C
        #print(size,"sizeeeeeee")
        c_indexes=np.random.choice(np.arange(0,self.cols), size=size, replace=True, p=self.colProb)
        C = self.matrix[:,c_indexes].astype(float)
        self.c_indexes = c_indexes
        for i in range(0, size):
            for j in range(0,self.rows):
                C[j][i] /= ((self.colProb[c_indexes[i]]*size)**0.5)
        return C

    def get_r(self,size = 513):
        #print(size,"sizeeeeeee")
        #Randomly sample rows to form R
        r_indexes=np.random.choice(np.arange(0,self.rows), size=size, replace=True, p=self.rowProb)
        R = self.matrix[r_indexes,:].astype(float)
        self.r_indexes = r_indexes
        for i in range(0, size):
            for j in range(0,self.cols):
                R[i][j] /= ((self.rowProb[r_indexes[i]]*size)**0.5)
        return R

    def get_u(self,c,r):
        ##Get the U matrix using svd decompostion
        ##Dimemtionality reduction if required
        w = c[self.r_indexes,:]
        x,z,y = la.svd(w,full_matrices=False)
        xt = x.T
        yt = y.T
        sigma = np.zeros(shape =(len(z),len(z)),dtype = float)
        zplus = np.zeros(shape =(len(z),len(z)),dtype = float)
        #print(z)
        for i in range(len(z)):
            sigma[i][i] = (z[i]**0.5)
        zplustemp = la.pinv(sigma)   #moore penrose pseudo inverse
        zplus = np.square(zplustemp)
        if self.save == True:
            save =self.percent/100
            totalEnergy = 0
            for i in range(len(z)):
                totalEnergy += sigma[i,i]**2
            retainedEnergy = 0
            seen_count = 0
            for i in range(len(z)):
                retainedEnergy += sigma[i,i]**2
                seen_count += 1
                if retainedEnergy >= save*totalEnergy:
                    break
            rem_count = len(z) - seen_count
            length = len(z)

            for i in range(rem_count):
                zplus=np.delete(zplus,length-1,0)
                zplus=np.delete(zplus,length-1,1)
                xt=np.delete(xt,length-1,0)
                yt=np.delete(yt,length-1,1)
                length -= 1
        u = np.dot(yt,np.dot(zplus,xt))
        return u

    def get_c_u_r(self,M):
        #splitting in to 3 matrices
        C = self.get_c(self.fourk)
        R = self.get_r(self.fourk)
        U = self.get_u(C,R)
        return C,U,R

    def multiply_c_u_r(self):
        #Return the multipled matrix
        result = np.dot((np.dot(self.C,self.U)),self.R)
        return result

    def rmse(self,predicted,actual):
        """Method to calculate the rmse
            between actual and predicted values
        """
        count = 0
        errorsum = 0
        for i in range(len(predicted)):
            count += 1
            error = abs(abs(predicted[i])-abs(actual[i]))
            errorsum += error**2
        return (errorsum**0.5)/(count**0.5)

    def get_topk(self,actual,k):
        """returns indexes in list which correspond to top ratings  """

        indexes = []
        largestRatings = heapq.nlargest(k,actual)
        largestRatings = set(largestRatings)
        for rating in largestRatings:
            temp =[i for i, x in enumerate(actual) if abs(x) == rating]
            indexes.extend(temp)

        return indexes ## a nested dictionary with top k keys

    def topk(self,predicted,actual,k=100):
        """RETURNS TOP K VLAUES PRECISION"""
        #print("error among top k predicted ratings")
        topkActualIndexes = self.get_topk(actual,k)
        topkPredictedIndexes = self.get_topk(predicted,k)
        #######
        topkActual = len(topkActualIndexes)
        topkPredicted = 0

        for i in topkPredictedIndexes:
            if i in topkActualIndexes:
                topkPredicted += 1
        #print("top",topkActual,topkPredicted,len(topkPredictedIndexes))
        return topkPredicted/len(topkPredictedIndexes)
        ######
        """
        topkActual = []
        topkPredicted = []
        for i in topkActualIndexes:
            topkActual.append(actual[i])
            topkPredicted.append(predicted[i])

        return self.rmse(topkPredicted,topkActual)"""

    def spearman(self,predicted,actual):
        #print("spearman rank")
        count = 0
        errorsum = 0
        for i in range(len(predicted)):
            count += 1
            error = abs(predicted[i]-actual[i])
            errorsum += error**2

        ro = 1 -((6*errorsum)/(count*(count**2 -1)))
        return ro

    def test(self,test_data):
        #print("test for whole matrix")
        result = self.curmatrix
        actual = []
        predicted = []
        users = self.users
        movies = self.movies
        for user in test_data:
            for movie in test_data[user]:
                u_idx = users.index(user)
                m_idx = movies.index(movie)
                a_val = test_data[user][movie]
                p_val = result[u_idx][m_idx]
                #print(a_val,"->",p_val)
                actual.append(a_val)
                predicted.append(p_val)

        self.predicted = predicted
        self.rmseerror = self.rmse(predicted,actual)
        self.topkerror = self.topk(predicted,actual)
        self.spearerror = self.spearman(predicted,actual)

def get_data():
    """Load preprocessed dataset
    """
    picklefile ="movieDataset1"
    with open(picklefile, 'rb') as f:
        data= pickle.load(f)
    return data

def load_data(fname):
    """
        Method to load the dataset and read
        choose a fixed number of data points
        since the dataset is large choosing a workable size

        and dividing it into test and train data.

        return : a dictionary of user rating for each movies

    """
    splitRows = []
    with open(fname) as rfile:
        datData = rfile.readlines()

    for row in datData:
        items = list(row.split("::"))
        splitRows.append(items)

    splitRows = np.array(splitRows[:], dtype=np.int)
    splitRows = np.delete(splitRows,3,1)

    numberOfUsers = 4000
    numberOfMovies = 2000

    users =np.ndarray.tolist(splitRows[:,0])
    usersubset = random.sample(users,numberOfUsers)

    movies = []
    for row in splitRows:            ## not very optimal change this
        if row[0] in usersubset:
            movies.append(row[1])

    moviesSubset =  random.sample(set(movies),numberOfMovies)

    testset =  300

    chosenlist = []
    for row in splitRows:
        user,movie,rating = row

        if user in usersubset and movie in moviesSubset:
            chosenlist.append(row)

    chosenIndexes = random.sample(range(0,len(chosenlist)), testset)

    test_data ={}
    train_data = {}
    full_data = {}

    for i in range(len(chosenlist)):
        user,movie,rating = chosenlist[i]

        if user in full_data.keys():
            full_data[user][movie] = rating
        else:
            full_data[user] = {movie:rating}

    #pearson_correlation(full_data)

    train_data = copy.deepcopy(full_data)
    for user in train_data: ## Choosing 1 movie per user for test data
        movielist = list(train_data[user].keys())
        choice = random.choice(movielist)
        testDict = {choice : train_data[user].pop(choice,None)}
        test_data[user] = testDict

        """
        if i in chosenIndexes:
                if user in test_data.keys():
                    test_data[user][movie] = rating
                else:
                    test_data[user] = {movie:rating}
        else:
                if user in train_data.keys():
                    train_data[user][movie] = rating
                else:
                    train_data[user] = {movie:rating}
        """
    data = [full_data,train_data,test_data,usersubset,moviesSubset]
    with open('movieDataset1', 'wb') as f:
            pickle.dump(data, f)
    #print(data)
    return data

def pearson_correlation(data):
    """
        noramlizes the dataset using pearson correlation
    """
    #Calculate Average
    for  user in data :
        sum = 0
        count = 0
        ratings = data[user]
        for movie in ratings:
            ratingValue = ratings[movie]
            sum += ratingValue
            count += 1

        average = sum/count
        if math.isnan(average):
            average = 0
        for movie in ratings:
            #print(ratings[movie],"->corel",average,ratings[movie]-average)
            ratings[movie] = round(ratings[movie]-average,1)

def display_results(output):
    """Display the  output in a neat table"""
    print("*********************RESULTS*********************")
    tb = tabulate(output[1:], headers=output[0], tablefmt='orgtbl')
    print(tb)

if __name__=="__main__":
    fname = "Dataset/bigdata/ratings.dat"

    """ uncomment the following line to process the data again"""
    #full_data,train_data,test_data,users,movies = load_data(fname)
    output = []
    full_data,train_data,test_data,users,movies = get_data()

    cf = collaborative(train_data,10)
    cf.test(test_data)
    print("Method\tRMSE\tPrecison@topk\tSpearman Rank\tTime")
    print("Collaborative",cf.rmseerror,"\t",cf.topkerror,"\t",cf.spearerror,"\t",cf.time)
    output.append(["Method","RMSE","Precison@topk","Spearman Rank","Time(in seconds)"])
    output.append(["Collaborative",cf.rmseerror,cf.topkerror,str(cf.spearerror),cf.time])

    cfb = collaborative(train_data,10,baseline = True)
    cfb.test(test_data)
    output.append(["Collaborative with baselines",cfb.rmseerror,cfb.topkerror,str(cfb.spearerror),cfb.time])
    print("Baseline","\t",cfb.rmseerror,"\t",cfb.topkerror,"\t",cfb.spearerror,"\t",cfb.time)

    svd = singular_value_decompositon(train_data,users,movies)
    svd.test(test_data)
    output.append(["SVD",svd.rmseerror,svd.topkerror,str(svd.spearerror),svd.time])
    print("SVD","\t",svd.rmseerror,"\t",svd.topkerror,"\t",svd.spearerror,"\t",svd.time)

    svd90 = singular_value_decompositon(train_data,users,movies,True,90)
    svd90.test(test_data)
    output.append(["SVD90",svd90.rmseerror,svd90.topkerror,str(svd90.spearerror),svd90.time])
    print("SVD90","\t",svd90.rmseerror,"\t",svd90.topkerror,"\t",svd90.spearerror,"\t",svd90.time)

    cur = cur_decomposition(train_data,users,movies)
    cur.test(test_data)
    output.append(["CUR",cur.rmseerror,cur.topkerror,str(cur.spearerror),cur.time])
    print("cur","\t",cur.rmseerror,"\t",cur.topkerror,"\t",cur.spearerror,"\t",cur.time)

    cur90 = cur_decomposition(train_data,users,movies,True,saving = 90)
    cur90.test(test_data)
    output.append(["CUR90",cur90.rmseerror,cur90.topkerror,str(cur90.spearerror),cur90.time])
    print("CUR90","\t",cur90.rmseerror,"\t",cur90.topkerror,"\t",cur90.spearerror,"\t",cur90.time)
    display_results(output)
