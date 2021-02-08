def cost_function(self,X,Y,weight,bias):
    n = len(X)
    total_error = 0.0
    for i in range(n):
        total_error += (Y[i] - (X[i] * weight + bias)) ** 2
    return total_error / n

def update_weights(X,Y,weight,bias,learning_rate):
    dw = 0
    db = 0
    n = len(X)
    for i in range(n):
        dw += -2 * X[i] * (Y[i] - (weight * X[i] + bias))
        db += -2 * (Y[i] - (weight * X[i] + bias))
    weight -= (dw /n ) * learning_rate
    bias -= (db / n) * learning_rate
    return weight,bias

x = [1,2,3,10,20,-2,-10,-100,-5,-20]
y = [0.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0]

weight , bias = update_weights(x,y,2,0.3,0.01)
print('weight={},bias={}'.format(weight,bias))

