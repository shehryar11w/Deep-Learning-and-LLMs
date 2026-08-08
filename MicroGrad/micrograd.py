class Value:
    def __init__(self,data,operation = None,parents = None):
        self.data = data
        self._backward = lambda: None
        self.grad = 0
        self.parents = parents or []
        self.operation = operation

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,'+',[self,other])
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out
    
    def __sub__(self, other):
        out = Value(self.data - other.data,'-',[self,other])
        def backward():
            self.grad += out.grad
            other.grad += out.grad * -1
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data,'*',[self,other])
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out
        
    def __pow__(self, other):
        out = Value(self.data ** other,f'** {other}',[self])
        def backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = backward
        return out
    
    def __rmul__(self, other): # other * self
        return self * other

    def __radd__(self, other): # other + self
        return self + other
    
    def __neg__(self): # -self
        return self * -1
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    def relu(self):
        out = Value(0 if self.data < 0 else self.data,'ReLU', [self])
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.parents:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
            print(v.grad)


x = Value(5)
y = Value(10)
z = x * y
z.backward()
print(z.data)
print(x.grad)