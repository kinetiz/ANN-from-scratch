import time
import numpy as np

class ANN:
    def __init__(self, x, y, num_hidden = 1, learning_rate = 10, iterations = 1000, hidden_size=3):
        self.x = x
        self.y = y
        self.M = x.shape[0]
        self.N = x.shape[1]
        self.num_hidden = num_hidden
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        # initialize weights by random; shape = N x Hidden_size; no. of instance = num_hidden
        self.weights = self.build_weights()
        self.biases = self.build_biases()
        self.yhat = None

    def build_weights(self):
        # print("build_weights. . .")
        weights = []
        mid_lweight = []
        np.random.seed(1)
        # First hidden layer shape = N x HiddenSize
        first_lweight = np.random.rand(self.N, self.hidden_size)

        # Middle hidden layers shape = HiddenSize x HiddenSize
        mid_lweight = [np.random.rand(self.hidden_size, self.hidden_size) for i in range(self.num_hidden-1)]

        # Last hidden layers shape = HiddenSize x Output
        last_lweight = np.random.rand(self.hidden_size, self.y.shape[1])

        # Concat list
        weights.append(first_lweight)

        if mid_lweight:
            weights += mid_lweight

        weights.append(last_lweight)

        return weights

    def build_biases(self):
        # print("build_biases. . .")
        biases = []
        mid_lweight = []
        np.random.seed(1)

        # Middle hidden layers shape = HiddenSize x 1
        mid_lbiases = [np.random.rand(1, self.hidden_size) for i in range(self.num_hidden)]

        # Last hidden layers shape = Output x 1
        last_lbias = np.random.rand(1, self.y.shape[1])

        # Concat list
        biases += mid_lbiases

        biases.append(last_lbias)

        return biases

    def sigmoid(self, x, derivative=False):
        # print("Sigmoid . . .")
        y = 1 / (1 + np.exp(-x))
        if derivative:
            y = y * (1 - y)
        return y

    def feedforward(self, input):
        # print("Feed forward. . .")
        ls_w = []
        ls_a = []
        ls_z = []
        z = 0
        a = input
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
            ls_w.append(w)
            ls_z.append(z)
            ls_a.append(a)
        yhat = a
        return yhat, ls_w, ls_z, ls_a

    def backpropagate(self, ls_dz_a, ls_prime_sig_z, ls_dz_w, error = None, ls_dw = [], ls_db = []):
        # print("backprop. . .{}".format(len(ls_dz_a)))

        # --- 1) Get the required params for backpropagating, start from last layer backward to the first one
        w = ls_dz_a.pop(-1)
        prime_sig_z = ls_prime_sig_z.pop(-1)
        a = ls_dz_w.pop(-1)
        # print('w:{},\n prime_sig_z:{},\n a:{}'.format(w, prime_sig_z, a))

        # --- 2) Calculate error propagated backward from the last to the first layer
        # Check None error for the first instance which is the last layer where there is no prior error
        if error is not None:
            error = np.dot(error, w.T) * prime_sig_z
        else:
            error = w * prime_sig_z

        # --- 3) Calculate weight and bias gradient of current layer (dw, db)
        dw = np.dot(a.T, error)
        ls_dw.append(dw)

        # sum up error for every data point (sum errors from every row, axis=0)
        # and use as bias gradients to update the biases
        ls_db.append(np.sum(error, axis=0, keepdims=True))

        # print('dw:{}'.format(dw))

        # Finish the recursive when list of dz/da is empty
        # This means all deltas have been processed (popped out) by every layer backward until the first layer
        if len(ls_dz_a) == 0:
            return ls_dw, ls_db
        else:
            return self.backpropagate(ls_dz_a, ls_prime_sig_z, ls_dz_w, error, ls_dw, ls_db)

    def train(self):
        # print("Training. . .")
        last_loss = 9999999

        for iter in range(self.iterations):

            # 1) --- Feed forward
            self.yhat, ls_w, ls_z, ls_a = self.feedforward(self.x)
            # print("w:{},\n z:{},\n a:{}".format(ls_w, ls_z, ls_a))



            # 2) --- Print out Loss function
            loss = 1/(2*self.M) * np.power(self.y - self.yhat, 2)

            # Print out learning progress
            print("Iteration: {}, Loss: {}".format(iter, np.average(loss)))

            # Finish training if loss is not improved (not decrease from last iteration)
            if np.average(loss) > last_loss:
                print('Converge!....')
                break
            last_loss = np.average(loss)



            # 3) --- Back propagation for gradients calculation

            # - Prepare dz/da
            # dz/da = W of all layers except the 1st W. And the last one is d_loss/d_yhat
            # So we copy ls_w, remove the first W
            ls_dz_a = ls_w.copy()
            ls_dz_a.pop(0)
            # And then add d_loss as dz/da for the last layer (which is the starting point)
            d_loss = (self.yhat-self.y)/self.M
            ls_dz_a.append(d_loss)


            # - Prepare dz/dw
            # All dz/dw are from A except the last layer, so we remove last A.
            ls_dz_w = ls_a.copy()
            ls_dz_w.pop(-1)
            # Instead, X is used as dz/dw of the first layer
            ls_dz_w.insert(0, self.x)

            # - Prepare sigmoid prime
            ls_prime_sig_z = [self.sigmoid(z, derivative=True) for z in ls_z]


            # - Calculate weight gradients
            ls_dw, ls_db = self.backpropagate(ls_dz_a, ls_prime_sig_z, ls_dz_w)
            # print("ls_dw:{}".format(ls_dw))


            # - Update weights and biases
            for i in range(len(self.weights)):
                #print(i)
                self.weights[i] -= self.learning_rate * ls_dw[-(i + 1)]
                self.biases[i] -= self.learning_rate * ls_db[-(i + 1)]

            # time.sleep(0.001)

    def predict(self, x):
        print("Predicting . . .")
        yhat, ls_w, ls_z, ls_a = self.feedforward(x)
        return yhat

    def predictnob(self, input):
        # print("Feed forward. . .")
        ls_w = []
        ls_a = []
        ls_z = []
        z = 0
        a = input
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) #+ b.T
            a = self.sigmoid(z)
            ls_w.append(w)
            ls_z.append(z)
            ls_a.append(a)
        yhat = a
        return yhat

# -------------------------------------
# # --- test recursive function
# aa(val=10, ls=[])
#
# def aa(val, ls):
#     if val == 0:
#         return val, ls
#     else:
#         ls.append(val)
#         val-=1
#         return aa(val, ls)
#
# aa(val=10, ls=[])

# Prepare XOR data

# ---- Main --------

# Prepare XOR dataset
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Create neural network object with dataset and configurations
n1 = ANN(x, y, num_hidden = 2, learning_rate = 2, iterations = 10000, hidden_size=10)

# Train the network
n1.train()

# Test by predicting XOR case
print(n1.predict(np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])))

# Try again without bias to compare how biases influence the result
print(n1.predictnob(np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])))
