import math

phi = list(range(0,8))
a = 1.2
b = 1.1 
r = 1.15
layer = [1,2,2,3,3,4,1]
for p in phi:
    print("b",p,"------")
    print("- a:", math.pow(a,p))
    print("- b:", math.pow(b,p))
    print("- r:", math.pow(r,p))
    l =  math.pow(a,p)
    l = math.ceil(l)
    print([l*i for i in layer])