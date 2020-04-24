import numpy as np
import random
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Change the floowing paras
kernel_name = 'linear'
C = 1000
polynomial = 7# para: p defined for Polynomials
sigma = 5 # para: sigma defined for RBF
pdf_name = 'Q5/svm_Q5_'+kernel_name+'_sigma_'+str(sigma)+'_C_'+str(C)+'.pdf'

#================== Generating Test Data ==================#
deviation = 0.4
np.random.seed(100) # get the same random data every time the program runs.
classA = np.concatenate(
                        (np.random.randn(10,2)*deviation + [1.5,0.5],
                         np.random.randn(10,2)*deviation + [-1.5,0.5]))
classB = np.random.randn(20,2)*deviation+[0.0,-0.5]
#classB = np.random.randn(20,2)*deviation+[2.0,-1.0]

inputs=np.concatenate((classA,classB))
targets=np.concatenate((np.ones(classA.shape[0]),
                        -np.ones(classB.shape[0])))
N=inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute,:]
targets = targets[permute]


#===========================================================#
# Implement Kernel Functions
def kernel(x, y, k_type):
    # Linear
    if (k_type == 'linear'):
        return np.dot(x, y) + 1
    # Polynomials ::: There is 1 para, p
    elif (k_type == 'poly'):
        return np.power((np.dot(x, y) + 1), polynomial)
    # RBF ::: The parameter sigma is used to control the smoothness of the boundary.
    elif (k_type == 'rbf'):
        difference = np.subtract(x, y)
        return math.exp((-np.dot(difference,difference))/(2*sigma*sigma))
    else:
        print("ERROR in kearnel name!!!")


# Pre-compute a matrix with these values: t(i)*t(j)*K
P = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        P[i][j] = targets[i]*targets[j]*kernel([inputs[i,0],inputs[i,1]], [inputs[j,0],inputs[j,1]], kernel_name)


#===========================================================#
# Implement the function objective
def objective(alpha):
    sum = 0
    alpha_sum = 0
    for i in range(N):
        for j in range(N):
            sum = sum + alpha[i]*alpha[j]*P[i][j]
    alpha_sum = np.sum(alpha)

    return sum - alpha_sum

#===========================================================#
# Implement the function zerofun
def zerofun(alpha):
    return np.dot(alpha, targets)

#===========================================================#
#Extract the non-zero α values
def separate_alpha(alpha):
    threshold = 0.00001
    not_zeros = []
    for v, element in enumerate(alpha):
        if (element > threshold):
            not_zeros.append((alpha[v], inputs[v][0], inputs[v][1], targets[v]))
            plt.plot(inputs[v][0], inputs[v][1], 'g+') #Plot
    return not_zeros

#===========================================================#
# Calculate the b value using equation (7).
def pre_b(alpha,v):
    pre_b = 0
    threshold = 0.00001
    for i in range(len(inputs)):
        pre_b = pre_b + alpha[i]*targets[i]*kernel(inputs[i], inputs[v], kernel_name)
    return pre_b - targets[v]
def b_main (alpha):
    b = 0
    threshold = 0.00001
    for j in range(len(inputs)):
        if (alpha[j] > threshold and alpha[j] < C):
            b = pre_b(alpha, j)
    return b

#===========================================================#
# Implement the indicator function
def indicator(svms, x, y):
    sum_ind = 0
    for i in range(len(svms)):
        sum_ind = sum_ind + svms[i][0]*svms[i][3]*kernel([x,y],[svms[i][1],svms[i][2]], kernel_name)
    sum_ind = sum_ind - b
    return sum_ind


# SVMs MAIN
if __name__ == "__main__":
    start = np.zeros(N)
    B = [(0, C) for b in range(N)]
    XC = {'type':'eq', 'fun':zerofun}
    
    ret_ = minimize(objective, start, bounds=B, constraints=XC)
    alpha = ret_['x']
    svm = separate_alpha(alpha)
    b = b_main(alpha)

    #Plotting
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b+')
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.')
    plt.axis('equal') # Force same scale on both axes
    #plt.savefig('svmplot_1.pdf') # Save a copy in a file
    blue_patch = mpatches.Patch(color='blue', label='ClassA')
    red_patch = mpatches.Patch(color='red', label='ClassB')
    #plt .show() # Show the plot on the screen
    
    xgrid = np.linspace(-5,5)
    ygrid = np.linspace(-4,4)
    
    grid = np.array([[indicator(svm, x, y)
                      for x in xgrid]
                     for y in ygrid])

    plt.contour(xgrid, ygrid, grid,
                (-1.0, 0.0, 1.0),
                colors=('red', 'black', 'blue'),
                linewidths=(1,3,1))

    # Show the patches
    black_patch = mpatches.Patch(color='black', label='Decision Boundry')
    plt.legend(handles=[blue_patch, red_patch, black_patch])

    # Show the labels
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(pdf_name)
    plt.show()
