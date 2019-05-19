

    def get_matrix(self,train_data,users,movies):
        """Generates a matrix of the test data
        """
        matrixlist = []
        for user in users:
            userlist=[]
            for movie in movies:
                if user in train_data and movie in train_data[user]:
                    userlist.append(train_data[user][movie])
                else:
                    userlist.append(0)
            matrixlist.append(userlist)
        matrix = np.matrix(matrixlist)

        return matrix

    def get_eigan_matrix(self,M):
        eiganValues, eiganVectors = la.eig(M)
        self.eiganValues = eiganValues
        eiganDict = {}
        for i in range(len(eiganValues)):
            eiganDict[eiganValues[i]] = eiganVectors[i]
        eiganValues.sort()
        eiganmatrixlist = []
        for value in eiganValues:
            eiganmatrixlist.append(eiganDict[value])

        eiganmatrix = np.matrix(eiganmatrixlist)
        eiganmatrix = eiganmatrix.transpose()
        return eiganmatrix

    def normalize_vector(self, V):
        size = V.shape[1]
        sum = 0
        for i in range(size):
            sum += V[i]**2
        length = sum**0.5

        U = []
        for i in range(size):
            U.append(V[i]/length)
        return U

    def get_dot_product(self,a,b):
        length = a.shape[1]
        product = 0
        for i in range(length):
            product += a[i]*b[i]
        return product

    def get_w(u,v,i):
        w=v[i]
        j = i-1
        while j>=0:
            temp -= self.get_dot_product(u[j],v[i]) * u[j])
        return w

    def get_matrix(self,u):
        m = u[0]
        for i in range(1,len(u)):
            m = np.concatenate((m,u[i]))
        return m.transpose()

    def graham_schmidt_orthonormalize(self,M):
        """Returns an orthonormal matrix
        """
        rows = M.shape[0]
        columns = M.shape[1]
        v=[[]]*columns
        u=[[]]*rows
        for i in range(columns):
            v[i] = M[:,i].transpose()

        for i in range(rows):
            if i == 0 :
                u[i] = self.normalize_vector(v[i])
            else:
                w[i] = self.get_w(u,v,i)
                u[i] = self.normalize_vector(w[i])
        normal_matrix = self.get_matrix(u)
        return normal_matrix

    def get_signma(self,values,row,col):            ###include logic to calculate the dimensions and update
        values = self.eiganValues.sort()
        sigmalist=[[0]*col]*rows
        for i in range(len(values)):
            sigmalist[i][i] = values[i]**0.5

        return np.matrix(sigmalist)

    def get_u_sigma_v(self,M):
        """splitting in to 3 matrices """
        Mt = M.transpose()
        MMt = M*Mt
        eiganmatrix_u = self.get_eigan_matrix(MMt)
        U = self.graham_schmidt_orthonormalize(eiganmatrix_u)
        sigma = self.get_sigma()
        MtM = Mt*M
        eiganmatrix_v = self.get_eigan_matrix(MtM)
        V = self.graham_schmidt_orthonormalize(eiganmatrix_v)
        return U,sigma,V.transpose()
