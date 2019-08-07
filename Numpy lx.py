#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


print(np.__version__)
np.show_config()


# In[3]:


Z = np.zeros(10)
print(Z)


# In[4]:


Z = np.zeros((10,10))
print('%d bytes' % (Z.size*Z.itemsize))


# In[6]:


numpy.info(numpy.add)


# In[7]:


np.info(np.add)


# In[8]:


Z = np.zeros(10)
Z[4] = 1
print(Z)


# In[9]:


Z = np.arange(10,50)
print(Z)


# In[10]:


Z = np.arange(50)
Z = Z[::-1]
print(Z)


# In[11]:


Z = np.arange(9).reshape(3,3)
print(Z)


# In[12]:


nz = np.nonzero([1,2,0,0,4,0])
print(nz)


# In[13]:


Z = np.eye(3)
print(Z)


# In[14]:


Z=np.random.random((3,3,3))
print(Z)


# In[15]:


Z = np.random.random((10,10))
Zmin,Zmax = Z.min(),Z.max()
print(Zmin,Zmax)


# In[16]:


Z=np.random.random(30)
m=Z.mean()
print(m)


# In[17]:


z= np.ones(10,10)
z[1:-1,1:-1]=0
print(z)


# In[18]:


Z= np.ones(10,10)
Z[1:-1,1:-1] = 0
print(Z)


# In[19]:


Z= np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)


# In[21]:


Z= np.ones((10,10))
print(Z)


# In[22]:


Z= np.ones((5,5))
Z= np.pad(Z,pad_width=1,mode='constant',constant_values=0)
print(Z)


# In[24]:


0*np.nan
np.nan==np.nan
np.inf > np.nan
np.nan - np.nan
0.3 ==3*0.1


# In[25]:


0.3 ==3*0.1


# In[26]:


print(0*np.nan)
print(np.nan==np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 ==3*0.1)


# In[27]:


Z=np.diag(1+np.arange(4),k=-1)
print(Z)


# In[28]:


Z=np.diag(np.arange(4),k=-1)#np.arange(4)与1+np.arange(4)的区别
print(Z)


# In[29]:


Z=np.diag(1+np.arange(4))
print(Z)


# In[30]:


Z=np.diag(1+np.arange(4),k=0)
print(Z)


# In[31]:


Z=np.diag(np.arange(4),k=0)
print(Z)


# In[32]:


Z=np.diag(1+np.arange(4),k=1)#k设置相当于对角线的位置，k=0位于对角线，k=-1位于对角线下，k=1位于对角线上，k的值会影响矩阵的大小
print(Z)


# In[34]:


Z=np.diag(np.arange(4),k=2)
print(Z)


# In[40]:


Z=np.zeros((8,8),dtype=int)#19题
Z[1::2,::2]=1
Z[::2,1::2]=1
print(Z)


# In[41]:


print(np.unravel_index(100,(6,7,8)))


# In[43]:


Z=np.tile(np.array([[0,1],[1,0]]),(4,4))
print(Z)


# In[44]:


Z = np.random.random((5,5))
Zmax,Zmin=Z.max(),Z.min()
Z=(Z-Zmin)/(Zmax-Zmin)
print(Z)


# In[45]:


color=np.dtype([('r',np.ubyte,1),
               ('g',np.ubyte,1),
               ('b',np.ubyte,1),
               ('a',np.ubyte,1),])
color


# In[46]:


Z=np.dot(np.ones((5,3)),np.ones((3,2)))
print(Z)


# In[47]:


Z=np.arange(11)
Z[(3<Z)&(Z<=8)]*=-1
print(Z)


# In[49]:


print(sum(range(5),-1))
from numpy import*
print(sum(range(5),-1))


# In[50]:


Z=np.arange(5)
Z**Z


# In[51]:


Z=np.arange(5)
2<<Z>>2


# In[52]:


Z=np.arange(5)
Z<-Z


# In[53]:


Z=np.arange(5)
1j*Z


# In[54]:


Z=np.arange(5)
Z/1/1


# In[56]:


Z=np.arange(5)
Z<Z>Z


# In[57]:


print(np.array(0)/np.array(0))
print(np.array(0)//np.array(0))
print(np.array([np.nan]).astype(int).astype(float))


# In[58]:


Z=np.random.uniform(-10,+10,10)
print(np.copysign(np.ceil(np.abs(Z)),Z))


# In[59]:


Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))


# In[60]:


Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

Back to sanity
_ = np.seterr(**defaults)

An equivalent way, with a context manager:

with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0


# In[61]:


np.sqrt(-1) == np.emath.sqrt(-1)


# In[62]:


yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print ("Yesterday is " + str(yesterday))
print ("Today is " + str(today))
print ("Tomorrow is "+ str(tomorrow))


# In[63]:


Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)


# In[64]:


A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)


# In[65]:


A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)


# In[66]:


A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)


# In[67]:


np.divide(A,2,out=A)


# In[68]:


np.negative(A,out=A)


# In[69]:


np.multiply(A,B,out=A)


# In[70]:


Z = np.random.uniform(0,10,10)
print (Z - Z%1)


# In[71]:


print (np.floor(Z))


# In[72]:


print (np.ceil(Z)-1)


# In[73]:


print (Z.astype(int))


# In[74]:


print (np.trunc(Z))


# In[75]:


Z = np.zeros((5,5))
Z += np.arange(5)
print (Z)


# In[76]:


def generate():
     for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print (Z)


# In[77]:


Z = np.linspace(0,1,11,endpoint=False)[1:]
print (Z)


# In[78]:


Z = np.random.random(10)
Z.sort()
print (Z)


# In[79]:


Z = np.arange(10)
np.add.reduce(Z)


# In[81]:


A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
#Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)


# In[82]:


equal = np.array_equal(A,B)
print(equal)


# In[83]:


A==B


# In[84]:


Z = np.zeros(10)
Z.flags.writeable = False
Z[0] = 1


# In[85]:


Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print (R)
print (T)


# In[86]:


Z = np.random.random(10)
Z[Z.argmax()] = 0
print (Z)


# In[87]:


Z = np.random.random(10)
Z[Z.argmax()] = 1
print (Z)


# In[89]:


Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                              np.linspace(0,1,5))
print(Z)


# In[90]:


X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))


# In[91]:


for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)


# In[92]:


np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print (Z)


# In[93]:


Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print (Z[index])


# In[94]:


Z = np.arange(100)
v = np.random.uniform(0,100)


# In[ ]:




