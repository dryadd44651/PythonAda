import math
import numpy as np

class AdaBoosting(object):
    data = []
    label = []
    p = []
    q = []
    weight = []#alpha
    number = 0
    size = 0
    def setData(self,src):
        self.data.clear()
        for i in src:
            self.data.append(i)
    def setLabel(self,src):
        self.label.clear()
        for i in src:
            self.label.append(i)
    def showData(self):
        print("\nData")
        for i in self.data:
            for j in i:
                print(j, end = ',')
            #print('')
            #print("==")
    def showLabel(self):
        print("\nLabel")
        for i in self.label:
            print(i, end = ',')
        print('')
    def iniWeight(self,n):#n: number of classifiers
        self.size = len(self.label)
        self.number = n
        self.p = [1/self.size]*self.size
        self.q = [1/self.size]*self.size
        self.weight = [1/self.number]*self.number
    
    def update(self):
        #error = 0
        for i in range(self.number):
            error = 0
            z = 0
            result = [0]*self.size
            bound = 0
            Et = 0
            for j in range(self.size):#get error
                result[j] = eval('%s(%s)'%('self.classifier'+str(i),'self.data['+str(j)+']'))
                if result[j] != self.label[j]:
                    error += self.p[j]
                    Et+=1
            Et/=self.size
            self.weight[i] = 0.5*math.log((1-error)/error)#update weight
            bound += (error*(1-error))**0.5
            print("error",i,": ", error)
            for j in range(self.size):#update q
                if result[j] == self.label[j]:
                    self.q[j] = math.exp(-self.weight[i])*self.p[j]
                    z += self.q[j]
                else:
                    self.q[j] = math.exp(self.weight[i])*self.p[j]
                    z += self.q[j]
            for j in range(self.size):#update 
                self.p[j] = self.q[j]/z
            print("z",i,": %8.2f" % (z))
            print("p",i,": ",end=' ')
            for tmp in self.p: print("%8.3f"%tmp,end=' ')
            print()
            print("Et",i,": %8.2f" %Et)
            print("--------------------------------")
        print("weight: ", self.weight)
        print("bound: ", bound)
    def classifier0(self,x):
        if x[0]<1.5:#8 9
            return 1
        else:
            return -1
    def classifier1(self,x):
        if x[0]<2.5:#0 1
            return 1
        else:
            return -1
    def classifier2(self,x):
        if x[0]<4:#0 1
            return 1
        else:
            return -1
    def classifier3(self,x):
        if x[0]>0:#8 9
            return 1
        else:
            return -1

    def test(self,x):
        result = [0.]*self.size
        
        for j in range(self.size):#get error
            for i in range(self.number): 
                result[j] += eval('%s(%s)'%('self.classifier'+str(i),'x['+str(j)+']'))*float(self.weight[i])
            if result[j]>0:
                result[j] = 1
            else:
                result[j]= -1
        print(result)
ada = AdaBoosting();
#x = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
#y = [1,1,1,-1,-1,-1,1,1,1,-1]
#y = [1,1,1,1,1,1,1,1,1,1]
train_x_location = "x_train.csv"
#(x_train,x_train1) = np.loadtxt(train_x_location, dtype="float64", delimiter=" ", usecols=(0, 2),unpack=True)
with open(train_x_location) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [i.strip() for i in content] 
data = []

for i in content:
    #tmp.append(i.split(" "))
    tmp = []    
    for j in i.split(" "):
        tmp.append(float(j))
    
    data.append(tmp)
x = [[i] for i in data[1]]

ada.setData(x)
ada.setLabel(data[2])
ada.showData()
ada.showLabel()
ada.iniWeight(4)
ada.p = data[3]
for i in range(int(data[0][0])):
    ada.update()

ada.test(x)