import keras
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

def plotBoundary(data, labels, clf_1, clf_2, clf_3, N):
    class_1 = data[labels == 1]
    class_0 = data[labels == 0]
    N = 300
    mins = data[:,:2].min(axis=0)
    maxs = data[:,:2].max(axis=0)
    X = np.linspace(mins[0], maxs[0], N)
    Y = np.linspace(mins[1], maxs[1], N)
    X, Y = np.meshgrid(X, Y)

    Z_nn = clf_1.predict_proba(np.c_[X.flatten(), Y.flatten()])[:, 0]
    Z_lr = clf_2.predict_proba(np.c_[X.flatten(), Y.flatten()])[:, 0]
    Z_nb = clf_3.predict_proba(np.c_[X.flatten(), Y.flatten()])[:, 0]

    # Put the result into a color plot
    Z_lr = Z_lr.reshape(X.shape)
    Z_nn = Z_nn.reshape(X.shape)
    Z_nb = Z_nb.reshape(X.shape)

    fig = plt.figure(figsize=(20,10))
    ax = fig.gca()
    cm = plt.cm.RdBu
        
    ax.contour(X, Y, Z_lr, (0.5,), colors='r', linewidths=0.5)
    ax.contour(X, Y, Z_nn, (0.5,), colors='b', linewidths=1)
    ax.contour(X, Y, Z_nb, (0.5,), colors='g', linewidths=0.5)
    ax.scatter(class_1[:,0], class_1[:,1], color='b', s=2, alpha=0.5)
    ax.scatter(class_0[:,0], class_0[:,1], color='r', s=2, alpha=0.5)
    proxy = [plt.Rectangle((0,0),1,1,fc = pc) 
    for pc in ['r', 'b', 'g']]
    ax.legend(proxy, ["Keras NN", "Regresion Logistica", "Naive Bayes"])
    plt.show()

class PlotBoundary(keras.callbacks.Callback):
    def plotBoundary(self):
        clear_output(wait=True)
        fig=plt.figure(figsize=(20,8))
        gs=GridSpec(2,2) # 2 rows, 3 columns
        ax=fig.add_subplot(gs[:,0]) # Second row, span all columns
        axLoss=fig.add_subplot(gs[0,1]) # First row, first column
        axAcc=fig.add_subplot(gs[1,1]) # First row, second column
        #self.fig, (self.ax, self.axLoss, self.axAcc )= plt.subplots(1,3, figsize=(20,4))
        ax.scatter(self.class_1[:,0], self.class_1[:,1], color='b', s=2, alpha=0.5)
        ax.scatter(self.class_0[:,0], self.class_0[:,1], color='r', s=2, alpha=0.5)
        Z = 1 - self.model.predict_proba(np.c_[self.X.flatten(), self.Y.flatten()])[:, 0]
        Z = Z.reshape(self.Z_shape)
        ax.contour(self.X, self.Y, Z, (0.5,), colors='k', linewidths=0.5)
        axAcc.plot(self.acc)
        axLoss.plot(self.loss)
        #self.fig.canvas.draw()
        plt.show()
        
        
    def __init__(self, data, labels, plots_every_batches=100, N = 300):
        self.plots_every_batches = plots_every_batches
        self.N = N
        mins = data[:,:2].min(axis=0)
        maxs = data[:,:2].max(axis=0)
        X_lin = np.linspace(mins[0], maxs[0], self.N)
        Y_lin = np.linspace(mins[1], maxs[1], self.N)
        self.X, self.Y = np.meshgrid(X_lin, Y_lin)
        self.Z_shape = self.X.shape
        self.acc = []
        self.loss = []
        self.class_1 = data[labels == 1]
        self.class_0 = data[labels == 0]
        #ax.set_ylabel('Alturas [cms]')
        #ax.set_xlabel('Pesos [kgs]')
        #plt.colorbar(cf, ax=ax)
        
    def on_train_begin(self, logs={}):
        self.plotBoundary()
        return
    
    def on_epoch_end(self, epoch, logs={}):
        return
    
    def on_batch_end(self, batch, logs={}):
        if batch%self.plots_every_batches == 0:
            self.acc.append(logs.get('acc'))
            self.loss.append(logs.get('loss'))
            self.plotBoundary()
        return