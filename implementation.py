# Generated from: implementation.ipynb
# Converted at: 2026-09-27T13:16:41.364Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import math
from sklearn.datasets import make_moons
X,y = make_moons(n_samples = 200,noise = 0.2,random_state = 42)
print(X.shape)
print(y.shape)

class Value:
    def __init__(self,data,_children = (),_op='',label = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = _children
        self._op = _op
        self.label = label 
    def __repr__(self):
        return f"Value(data = {self.data})"
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data, (self , other),'+')
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"Only supports Int and Float"

        out = Value(self.data ** other,(self,),f'**{other}')
        def _backward():
            self.grad += other * (self.data **(other-1)) * out.grad # power rule and chain rule
        out._backward = _backward
        
        return out
    def __rmul__(self,other): #other * self
        return self * other

    def __truediv__(self,other):#self / other
        return self * other**-1

    def __neg__(self):#-self
        return self * -1
    
    def __sub__(self,other):#self - other
        return self +(-other)
        
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self , other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
        
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1 -t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self,),'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
        

    def backward(self):
        Topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                Topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(Topo):
            node._backward()


import random
class Neuron:
    def __init__(self,nin):#by nin we mean number of inputs how many neurons?
        self.w = [Value(random.uniform(1,-1))for _ in range(nin)] #weight from random number from 1 to -1
        self.b = Value(random.uniform(1,-1))# a bias 
    def __call__(self,x):
        act = sum((wi * xi for wi,xi in zip(self.w,x)),self.b) #w*x+b
        out = act.tanh()
        return out
    def parameters(self):
        return self.w + [self.b]
class Layer:
    def __init__(self,nin,nout):#nout here means how many number of inputs we want from this layer
        self.neurons = [Neuron(nin) for _ in range (nout)]#Initializing the neurons according to the nin
    
    def __call__(self,x):
        outs = [n(x) for n in self.neurons] #independently evaluating the neurons from above
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # params = []
        # for neurons in self.neurons:
        #     ps = neurons.parameters()
        #     params.extends(ps)
        # return params
        # the above code can also be written as this in simpler form
        return [p for neurons in self.neurons for p in neurons.parameters()]
            

class MLP:
    
    def __init__(self,nin,nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[i],size[i+1])for i in range (len(nouts))]
    def __call__(self , x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return[p for layer in self.layers for p in layer.parameters()]
        

Xs = [
    [[-1.10689665e+00,  4.22928095e-02],
       [ 9.56799641e-01,  4.56750492e-01],
       [ 7.33516277e-01,  5.84617437e-01],
       [ 1.11140659e+00, -3.09213987e-01],
       [ 2.09081764e-01,  6.56679495e-04],
       [ 3.92205613e-01,  3.20238629e-01],
       [-7.56185073e-01,  8.29954942e-01],
       [ 1.50692319e+00, -1.11129319e-01],
       [ 2.04279588e+00, -3.79762883e-02],
       [ 1.44143707e-01,  9.16713583e-01],
       [-3.70182653e-01,  6.21450696e-01],
       [ 1.96525062e+00, -1.96615578e-01],
       [ 1.59638072e+00, -4.13640839e-01],
       [-8.63705975e-01,  4.41056511e-01],
       [ 1.84475945e+00,  2.10293824e-01],
       [ 1.97481657e+00, -3.74979774e-02],
       [ 6.79706368e-02,  1.48663499e+00],
       [ 1.01435177e+00, -5.38028119e-01],
       [-1.14599170e+00,  7.51822843e-01],
       [-2.94040701e-01,  6.59140623e-01],
       [ 3.99794131e-01,  5.33351479e-02],
       [ 9.19107031e-01,  6.55677023e-01],
       [ 9.87384687e-01, -7.42238318e-01],
       [-7.72059219e-01,  4.54272377e-01],
       [-3.25370862e-01,  6.90553911e-01],
       [-4.77303848e-01,  7.18915208e-01],
       [-7.34839977e-01,  7.11904426e-01],
       [-4.42328219e-01,  1.03783255e+00],
       [ 1.24996894e-01, -1.83521801e-01],
       [ 1.98800321e-01, -5.04081580e-01],
       [ 1.13522768e+00,  2.33813655e-01],
       [ 1.08312993e+00,  6.48232818e-01],
       [ 3.73236254e-01,  1.00356084e+00],
       [-1.00501509e+00,  6.83173080e-01],
       [ 8.23262839e-01, -3.04438128e-01],
       [ 1.64412001e-02,  3.45953183e-01],
       [ 5.31106910e-01, -2.47925611e-01],
       [ 5.23162428e-01, -2.66944335e-01],
       [ 1.65318085e+00, -4.64422117e-01],
       [-9.30473774e-01,  5.64175937e-01],
       [-6.63248409e-01,  6.11403703e-01],
       [ 2.12400166e+00,  7.89163575e-01],
       [ 4.03076966e-01,  7.72360482e-01],
       [ 9.35218740e-01,  7.00351183e-01],
       [ 4.08444988e-01,  7.04520991e-02],
       [ 1.80015319e+00,  6.49808467e-01],
       [ 7.86634079e-01,  6.74980966e-01],
       [ 8.84566405e-01,  6.18501362e-01],
       [ 7.99979695e-01, -3.04419648e-01],
       [-6.28637192e-01,  4.02391810e-01],
       [ 3.63249727e-01,  2.67845610e-01],
       [ 1.72496864e+00, -1.99355524e-01],
       [ 7.56925814e-01, -1.42152283e-01],
       [ 3.54495148e-01, -4.05777492e-01],
       [ 1.34089358e-01, -1.20609940e-01],
       [-1.03620871e+00,  8.06052617e-01],
       [ 6.20173264e-01, -4.38862499e-01],
       [-1.17812060e+00, -3.52355160e-02],
       [ 1.96693676e+00, -2.30039374e-01],
       [-8.85556654e-01, -1.59118318e-01],
       [ 2.05786410e+00,  5.41067996e-02],
       [-7.25299578e-01,  2.29177115e-01],
       [ 5.56948720e-01,  9.20984658e-01],
       [ 1.13797188e+00,  2.45948506e-01],
       [ 3.55166864e-01,  5.62167302e-01],
       [ 2.19596030e-02,  2.65200482e-01],
       [ 1.68909517e-01,  9.08242187e-01],
       [ 2.01362452e+00, -1.34478907e-01],
       [-7.70416636e-01,  5.35013928e-01],
       [ 6.60044240e-01, -7.93617981e-02],
       [-3.12948269e-01,  1.36761050e+00],
       [ 1.98990985e-01, -5.11733521e-01],
       [ 1.33886673e+00, -4.79531580e-02],
       [ 2.62585386e-01,  7.77323827e-01],
       [-1.00242028e-01, -6.16645184e-02],
       [-3.15100772e-01, -1.60969104e-01],
       [ 9.87122942e-01,  9.99244470e-01],
       [ 6.47249607e-01, -5.38133773e-01],
       [ 1.17017805e-02,  9.69599386e-01],
       [ 1.41302481e-01,  3.39927492e-01],
       [ 9.57495171e-01,  7.56032411e-02],
       [-6.36611904e-01,  6.68006575e-01],
       [-2.68637197e-01,  1.24994662e+00],
       [ 1.80534120e+00,  5.50601073e-01],
       [ 2.10531840e+00,  2.63046496e-01],
       [-8.15061110e-01,  4.08095334e-01],
       [ 1.25959967e+00, -3.92628208e-01],
       [ 5.92413106e-01,  8.16726046e-01],
       [ 1.83504554e+00,  1.47169715e-01],
       [-1.66484003e-01,  1.22465791e+00],
       [ 8.38447226e-01,  7.59050436e-01],
       [ 6.07586015e-01, -2.88491229e-01],
       [ 1.10850860e-01,  3.26246475e-01],
       [ 1.42465678e-01,  6.21911383e-01],
       [ 8.05852075e-01,  5.74407607e-01],
       [ 2.29801084e+00,  2.54095029e-01],
       [ 1.21533730e+00, -4.02577630e-01],
       [ 9.84357999e-01,  3.75438061e-01],
       [ 5.50223115e-01,  9.45578172e-01],
       [ 1.46884783e-01,  7.78870132e-02],
       [ 9.37746720e-01,  3.97351699e-03],
       [ 9.09688996e-01,  1.06277085e+00],
       [ 4.05832044e-01, -5.61166271e-01],
       [ 1.09866975e+00, -7.29153536e-01],
       [ 1.94642692e-01,  3.51678664e-01],
       [ 6.86522507e-01, -4.86571162e-01],
       [ 5.32067568e-01,  6.66119749e-01],
       [-4.48163628e-01,  1.08311690e+00],
       [ 1.94400335e-01,  3.99913702e-01],
       [ 8.02501119e-01, -4.53073919e-01],
       [ 1.61182184e+00, -1.91517085e-01],
       [ 1.23087087e+00, -9.54763461e-02],
       [ 1.86665105e+00,  7.40294340e-01],
       [ 7.03423013e-01, -1.79832008e-01],
       [-6.66074536e-01,  7.94849275e-01],
       [ 1.18267030e-01, -1.76408098e-01],
       [-1.89838967e-01,  1.05016639e+00],
       [ 2.72185474e-01,  1.00258235e+00],
       [ 1.12629302e+00, -5.78459378e-01],
       [ 5.61912420e-01,  8.67681755e-01],
       [ 3.10850908e-01,  1.15160310e+00],
       [ 1.26776776e-01,  2.59722128e-01],
       [ 1.62880044e+00, -4.03562416e-01],
       [ 1.11841823e+00, -1.06809045e-01],
       [ 9.41557197e-01,  1.07591173e+00],
       [ 7.23078063e-01,  8.62030826e-01],
       [ 1.59826556e+00,  1.98574760e-01],
       [ 8.89205251e-01, -7.67866367e-01],
       [-6.34733452e-01,  1.78614552e-01],
       [-5.54733401e-02,  1.01749611e+00],
       [ 1.66818957e+00, -1.38465091e-01],
       [ 2.27642560e-01,  1.29218650e+00],
       [-8.47390298e-01,  1.31541507e-01],
       [ 1.12759262e+00, -5.46265680e-01],
       [ 2.63902173e-01,  7.59333273e-01],
       [ 7.11298218e-01, -4.51430562e-01],
       [ 1.05059895e+00,  1.50800650e-01],
       [-4.77891809e-01,  1.61739840e-01],
       [-1.32563933e-01,  9.82741473e-01],
       [ 5.51877862e-01, -3.05555746e-01],
       [ 2.12692325e+00, -4.56816764e-03],
       [ 1.72344835e+00, -4.37983681e-01],
       [ 1.62920852e+00,  1.77303450e-01],
       [ 2.27955327e-01,  1.30814833e+00],
       [-9.89255197e-01,  5.06286977e-01],
       [ 5.20737812e-01,  1.29093315e+00],
       [ 9.98402081e-01,  5.30563104e-01],
       [ 8.39234628e-02,  4.18615592e-01],
       [-7.62632732e-01,  2.84365614e-01],
       [-1.07792603e+00,  4.32377849e-01],
       [ 2.04307348e+00,  8.79603318e-03],
       [ 1.75452347e+00, -1.09102545e-01],
       [-1.29624574e+00,  6.00901204e-01],
       [ 1.14516141e+00,  2.23732423e-01],
       [ 9.56135341e-01, -3.32426302e-02],
       [-3.94263275e-01,  1.22179137e+00],
       [ 1.82686214e+00, -2.10006353e-01],
       [ 4.43136284e-01, -1.73288769e-01],
       [ 8.01706392e-01,  1.80738752e-01],
       [-6.56100562e-01,  8.36373440e-01],
       [ 1.94826958e+00, -4.78318164e-01],
       [ 5.03146664e-01, -3.96514094e-01],
       [ 1.30697509e+00, -2.52190775e-01],
       [ 2.02332298e+00, -4.43351977e-02],
       [-4.39649128e-03,  8.45759994e-01],
       [-2.31651532e-01,  8.94687172e-01],
       [-1.38529602e-01,  1.67884369e-01],
       [-3.02683738e-01,  9.61567563e-01],
       [ 2.00696672e+00, -5.16165413e-02],
       [-2.86863716e-01,  9.72268975e-01],
       [ 1.50499714e-01,  1.23979435e+00],
       [-7.69421768e-01,  2.17947399e-01],
       [ 6.38770660e-01,  7.34059867e-01],
       [-3.45214903e-01,  7.26200764e-01],
       [ 1.80189631e+00, -5.07145672e-01],
       [ 1.85012495e+00, -1.58736544e-01],
       [ 5.64670731e-01, -4.93349010e-01],
       [ 3.92069238e-01,  1.11607893e+00],
       [ 1.04130523e+00, -4.55915078e-01],
       [ 2.28668618e+00, -1.80600656e-01],
       [ 8.37395373e-01, -1.49180443e-01],
       [-1.16630981e-01,  9.87296600e-01],
       [ 7.71881510e-01, -6.30748797e-01],
       [ 2.65298096e-01, -3.84733801e-01],
       [ 1.09122646e+00, -3.16754515e-01],
       [ 2.15594277e-01,  5.35670809e-01],
       [-1.17479482e+00, -6.59184065e-04],
       [ 6.01016504e-01,  2.16709802e-01],
       [-1.01646884e+00,  6.88070092e-01],
       [-7.79088726e-01,  5.79013778e-01],
       [ 4.93093292e-01, -3.80088844e-01],
       [ 1.61724448e+00,  6.88256916e-01],
       [ 8.10808444e-01,  6.18381025e-01],
       [ 1.86740805e+00,  1.10281183e-01],
       [-7.22174648e-01,  6.50947878e-01],
       [-5.83107417e-02,  2.28058508e-01],
       [ 7.37209784e-01,  3.20761072e-01],
       [-1.47598169e+00,  3.57238973e-01],
       [ 1.88332091e+00, -1.09889696e-01],
       [ 4.94782005e-02,  4.91291561e-01]]
]
ys = [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0,
       1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0,
       1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
       0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0,
       1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1,
       1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0,
       0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
       0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1,
       1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
       1, 1]


model = MLP(2,[16,16,1])
ypred = [model(x) for x in X]
ypred

loss = sum(((ygt - yout)**2 for yout,ygt in zip(ys,ypred)),Value(0.0))/len(ys)
loss

loss.backward()

model.layers[0].neurons[0].w[0].data

model.layers[0].neurons[0].w[0].grad

for epoch in range(100):
    #forward pass
    ypred = [model(x) for x in X]
    loss = sum(((ygt - yout)**2 for yout,ygt in zip(ys,ypred)),Value(0.0))/len(ys)

    #backward pass
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    #update
    for p in model.parameters():
        p.data += -0.5 * p.grad

    print(f'epoch:{epoch},loss: {loss.data}')
    

model.layers[0].neurons[0].w[0].data


if len(Xs) == 1 and isinstance(Xs[0], list) and isinstance(Xs[0][0], list):
    X = Xs[0]
else:
    X = Xs


predictions = [1 if model(x).data > 0 else 0 for x in X]
correct = sum(p == y for p, y in zip(predictions, ys))
accuracy = (correct / len(ys)) * 100

print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(ys)} correct)")