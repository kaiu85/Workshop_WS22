# PyTorch (https://pytorch.org/docs/stable/)
# eine (relativ) einfach zu benutzende Bibliothek, um neuronale Netze
# mit Graphikkartenunterstützung zu erstellen, trainieren und anzuwenden
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import sklearn

# Für graphischen Output nutzen wir meistens die Matplotlib-Bibliothek
import matplotlib.pyplot as plt

# Eine kategorielle Wahrscheinlichkeitsverteilung (Verallgemeinerung eines Würfel 
# auf "unfaire" (d.h. beliebig aufgeteilte) Wahrscheinlichkeiten für beliebig viele "Seiten")
from torch.distributions import Categorical
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from numpy import floor
from tqdm import tqdm

from scipy.special import softmax

# Um die Qualität zu verbessern, erzeugt diese Funktion nicht nur eine einzelne Zeichenkette, sondern eine 
# bestimmte Anzahl ("n_beams") und geben dann nur die Zeichenkette aus, die die höchste Wahrscheinlichkeit
# hat und der Verteilung der Daten, die unser RNN gelernt hat.

def sample_beam(rnn, device, prompt = '', n = 1000, n_beams = 32):

    # Wir können dem trainierten Netzwerk einen Starttext ("prompt") geben, von dem aus
    # es weiter Text erzeugen soll. Falls dieser nicht leer ist, geben wir ihn aus.
    if prompt != '':
        print('\nPrompt: ' + prompt + '\n')
    
    # Hier sagen wir PyTorch, dass wir das RNN nicht trainieren wollen, d.h. wir müssen uns
    # keine Information merken, die wir nur dazu brauchen, später die Ableitung (den Gradienten "grad")
    # der Kostenfunktion nach den Netzwerkparametern zu bestimmen, da wir das hier nicht vorhaben. 
    # Der Code zum Training kommt in einer der nächsten Zellen weiter unten.
    
    with torch.no_grad():        
    
        # Hier definieren wir die Kostenfunktion. Diese erhält eine Zeichensequenz und
        # entsprechende Wahrscheinlichkeitsvorhersagen unseres RNNs und gibt dann an,
        # wie wahrscheinlich diese Zeichenkette unter den momentanen Vorhersagen unseres 
        # RNNs ist.
        #
        # Im Training benutzen wir diese Funktion mit Trainingszeichenketten, die wir aus längeren
        # Texten extrahieren, und ändern die Parameter unseres Netzwerkes so, dass es die Trainings-
        # zeichenketten mit höheren Wahrscheinlichkeiten versieht.
        #
        # 
        #
        # CrossEntropyLoss klingt kompliziert, aber die
        loss_fn = nn.CrossEntropyLoss(reduction = 'mean')

        # convert data from chars to indices
        start_seq = list(prompt)
        for i, ch in enumerate(start_seq):
            start_seq[i] = rnn.char_to_ix[ch]

        # data tensor on device
        start_seq = torch.tensor(start_seq).to(device)
        start_seq = torch.unsqueeze(start_seq, dim=1)

        start_seq = start_seq.repeat(1, n_beams)

        #print('START SEQ SHAPE:')
        #print(start_seq.shape)

        # Re-initialize RNN
        hidden_state =  ( 
                            torch.zeros((rnn.num_layers, n_beams, rnn.hidden_size)).to(device),
                            torch.zeros((rnn.num_layers, n_beams, rnn.hidden_size)).to(device)
                        )
        
        complete_sequences = torch.zeros((len(prompt) + n,n_beams),dtype = start_seq.dtype).to(device)
        complete_outputs = torch.zeros((len(prompt) + n,n_beams, rnn.num_chars)).to(device)

        complete_sequences[:len(prompt),:] = start_seq

        # forward pass
        for i in range(len(prompt)):
            output, hidden_state = rnn(complete_sequences[i,:].unsqueeze(dim=0), hidden_state)
            complete_outputs[i,:,:] = output
            
        for i in range(n):
            
            for j in range(n_beams):
                softmax_output = F.softmax(torch.squeeze(complete_outputs[len(prompt)+i-1,j,:]), dim=0)
                dist = Categorical(softmax_output)
                index = dist.sample()
                complete_sequences[i+len(prompt),j] = index.item()
                
            output, hidden_state = rnn(complete_sequences[i+len(prompt),:].unsqueeze(dim=0), hidden_state)
            complete_outputs[len(prompt) + i,:,:] = output
            
        scores = torch.zeros((n_beams,)).to(device)

        min_score = 0
        min_index = 0

        for j in range(n_beams):

            output_seq = complete_outputs[:-1,j,:]
            target_seq = complete_sequences[1:,j]
            pred = torch.squeeze(output_seq.reshape(-1,rnn.num_chars))
            out = torch.squeeze(target_seq.reshape(-1))
            scores[j] = loss_fn(pred, out)

            if j == 0 or scores[j].item() < min_score:
                min_index = j
                min_score = scores[j].item()

        print('\n\nBEAM SCORES:')
        print(scores)

        print(min_score)
        print(min_index)

        print('\n\n========================\nBest BEAM Sequence %d:\n========================\n' % min_index)

        for i in range(len(prompt) + n):
            print(rnn.ix_to_char[complete_sequences[i,min_index].item()],end = '')

        print('\n')
        print('\n')
        
        
############ AB hier kommen noch ein paar (mehr oder weniger gut dokumentierte) Hilfsfunktionen

def simulate_gradient_descent_on_quadratic_potential(theta_start, learning_rate, n_steps):
    # Startwert für theta
    theta = theta_start

    # Stellen wir uns weiter vor, die Kostenfunktion hängt quadratisch von Theta ab 
    def cost(theta):

        cost = theta*theta;

        return cost

    # Dann können wir den Gradienten der Kostenfunktion direkt berechnen: 
    # Dieser ist die Ableitung der Kostenfunktion nach Theta
    def grad_cost(theta):
        grad = 2*theta

        return grad

    # Hier speichern wir den Verlauf der Parameter während des Gradientenabstiegs
    thetas = []
    # Hier speichern wir den Verlauf der Kostenfunktion während des Gradientenabstiegs
    costs = []

    for i in range(n_steps):

        # Speichere aktuelle Position
        thetas.append(theta)
        # Speichere aktuellen Wert der Kostenfunktion
        costs.append(cost(theta))

        # Mache einen Schritt entgegen dem Gradienten, gewichtet mit der Lernrate
        theta = theta - learning_rate*grad_cost(theta)

    theta_max = max(abs(theta_start),abs(thetas[-1]))
    xrange = np.arange(-theta_max,theta_max,0.01*theta_max)
    cost_line = cost(xrange)

    plt.figure(figsize=(20,10))    
    plt.subplot(1,2,1)
    for i in range(n_steps-1):
        plt.arrow(thetas[i],costs[i],thetas[i+1]-thetas[i],costs[i+1]-costs[i],head_width=0.1*theta_max, head_length=0.15*theta_max, length_includes_head = True, color = 'darkorange', overhang = 0.5)

    # Zeichne Kostenfunktion
    plt.plot(xrange, cost_line)
    # Zeichne Punkte
    plt.scatter(thetas,costs,s = 300.0, c=range(n_steps),cmap='jet',)

    plt.plot(thetas, costs)
    plt.xlabel('w',fontsize='xx-large')
    plt.ylabel('Wert der Zielfunktion',fontsize='xx-large')

    plt.subplot(1,2,2)
    plt.plot(costs)
    plt.scatter(range(n_steps),costs,s = 300.0, c=range(n_steps),cmap='jet',)
    plt.xlabel('Optimierungsschritt',fontsize='xx-large')
    plt.ylabel('Wert der Zielfunktion',fontsize='xx-large')
    
def log_metrics_to_file(path, numbers, create_file):
    
    if create_file:
        with open(path, 'w') as file:
            file.write("epoch loss\n")
        
    with open(path, 'a') as file:
        for n in numbers:
            file.write('%f ' % n)
        file.write("\n")
        
def read_log_data(log_file):
        
    log_mat = np.loadtxt(log_file, skiprows = 1)
    
    # put together values
    log_data = {
                "epoch": log_mat[:,0],
                "train loss": log_mat[:,1],
                "validation loss": log_mat[:,2]
               }
    
    return log_data

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

def plot_overfitting_demo(degrees):
    
    np.random.seed(0)

    n_samples = 30

    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.3

    plt.figure(figsize=(14, 5))
    for i in range(len(degrees)):
        ax = plt.subplot(1, len(degrees), i + 1)
        plt.setp(ax, xticks=(), yticks=())

        polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                 include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)
        
        yhat = pipeline.predict(X[:, np.newaxis])
        
        mse  = sklearn.metrics.mean_squared_error(y, yhat)
  
        # Evaluate the models using crossvalidation
        scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                                 scoring="neg_mean_squared_error", cv=10)

        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Polynom")
        plt.plot(X_test, true_fun(X_test), label="Wahre Funktion")
        plt.scatter(X, y, edgecolor='b', s=20, label="Stichprobe")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title("Grad {}\n Mittlerer quadratischer \nFehler auf Trainingsdaten:\n{:.2f}\n Mittlerer quadratischer \nFehler auf neuen Daten:\n{:.2f}(+/- {:.2f})".format(
            degrees[i], mse, -scores.mean(), scores.std()))
    plt.show()
    
def softmax_demo(output):
    
    probabilities = softmax(output)
    
    plt.figure(figsize=(14, 5))
    
    ax = plt.subplot(1, 2, 1)
    
    plt.axhline(y=0.0, color='k', linestyle='-')
    
    X_test = np.arange(len(output))
    plt.bar(X_test, output, 0.5, label="Output Aktivierungen o")
    plt.xlabel("Index")
    plt.ylabel("Output Activity")
    
    ax = plt.subplot(1, 2, 2)
    
    plt.axhline(y=0.0, color='k', linestyle='-')
    
    X_test = np.arange(len(output))
    plt.bar(X_test, probabilities, 0.5, label="Wahrscheinlichkeitsverteilung p")
    plt.xlabel("Index")
    plt.ylabel("Softmax Probability")
    
    plt.show()
    
def save_dataset_info(dataset_file, indices, filename):
    with open(filename, 'w') as file:
        file.write('%d %d %s\n' % (indices[0], indices[1], dataset_file))
        
def load_dataset_info(filename):
    with open(filename, 'r') as file:
        items = file.read().split()
        index0 = int(items[0])
        index1 = int(items[1])
        dataset_file = items[2]
    
    return dataset_file, [index0, index1]

def calculate_loss_on_dataset(data, rnn, num_heads, seq_len, device):
    
    loss_fn = nn.CrossEntropyLoss()
    
    data_size = len(data)
    
    iter_per_epoch = int(data_size / (seq_len + 1) / num_heads)
    
    length_per_head = int(data_size / num_heads)
    
    loss_sum = 0
    
    with torch.no_grad():
    
        input_seq = torch.zeros((seq_len, num_heads),dtype = data.dtype).to(device)
        target_seq = torch.zeros((seq_len, num_heads),dtype = data.dtype).to(device)

        hidden_state = ( 
                         torch.zeros((rnn.num_layers, num_heads, rnn.hidden_size)).to(device),
                         torch.zeros((rnn.num_layers, num_heads, rnn.hidden_size)).to(device)
                       )
        
        data_ptr = np.arange(num_heads)*length_per_head
        
        for i in range(iter_per_epoch):
            
            for j in range(num_heads):
                input_seq[:,j] = (data[data_ptr[j] : data_ptr[j]+seq_len]).squeeze()
                target_seq[:,j] = (data[data_ptr[j]+1 : data_ptr[j]+seq_len+1]).squeeze() 
                
            output, hidden_state = rnn(input_seq, hidden_state)
            
            # Jetzt bewegen wir alle Leseköpfe um die Länge der gerade gelesenen Sequenz weiter
            data_ptr += seq_len
            
            loss = loss_fn(torch.squeeze(output.reshape(-1,rnn.num_chars)), torch.squeeze(target_seq.reshape(-1)))
            loss_sum += loss.item()
                
        loss_sum /= iter_per_epoch
        
    return loss_sum  
    
def prepare_data(data_path, device):
    data = open(data_path, 'r').read()
    chars = sorted(list(set(data)))
    data_size, num_chars = len(data), len(chars)

    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }
    
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]

    # data tensor on device
    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)
    
    return data

def save_model_parameters(model_name, hidden_size, num_layers, num_chars, dropout):
    
    params_file = model_name + '_parameters.txt'
    
    with open(params_file, 'w') as file:
        file.write('%d %d %d %f\n' % (hidden_size, num_layers, num_chars, dropout))
        
def load_model_parameters(model_name):
    
    params_file = model_name + '_parameters.txt'
    
    with open(params_file, 'r') as file:
        items = file.read().split()
        
        hidden_size = int(items[0])
        num_layers = int(items[1])
        num_chars = int(items[2])
        dropout = float(items[3])
        
    return hidden_size, num_layers, num_chars, dropout    
    
class RNN(nn.Module):
    
    def __init__(self, num_chars, hidden_size, num_layers, char_to_ix, ix_to_char, dropout = 0.3):
        super(RNN, self).__init__()
        
        self.embedding = nn.Embedding(num_chars, num_chars)
        self.rnn = nn.LSTM(input_size=num_chars, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, num_chars)
        
        self.hidden_size = hidden_size
        self.num_chars = num_chars
        self.num_layers = num_layers
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char
    
    def forward(self, input_seq, hidden_state):
        
        embedding = self.embedding(input_seq)
        hidden_output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(hidden_output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())    
    
def train_rnn(
    model_name = 'vanilla', device = 'cuda', 
    data_path = './data/recipes.txt', train_percentage = 0.9, valid_percentage = 0.05, 
    hidden_size = 512, num_layers = 3, dropout = 0.3,
    seq_len = 100, num_heads = 16,
    epochs = 10,
    lr = 0.0003,
    load_chk = False,
    log_every = 1000):
    
    # Prepare Dataset
    
    data = open(data_path, 'r').read()
    chars = sorted(list(set(data)))
    data_size, num_chars = len(data), len(chars)
    
    char_to_ix = { ch:i for i,ch in enumerate(chars) }
    ix_to_char = { i:ch for i,ch in enumerate(chars) }

    with open('char_to_ix_' + model_name + '.pkl', 'wb') as output:    
        pickle.dump(char_to_ix, output)
    with open('ix_to_char_' + model_name + '.pkl', 'wb') as output:    
        pickle.dump(ix_to_char, output)
    
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]

    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)
    
    test_percentage = 1.0 - train_percentage - valid_percentage
      
    valid_index_start = int(floor(train_percentage*data_size))
    test_index_start = int(floor((train_percentage + valid_percentage)*data_size))

    train_data = data[0:(valid_index_start-1)]
    valid_data = data[valid_index_start:(test_index_start-1)]
    test_data = data[test_index_start:]
    
    train_data_size = len(train_data)

    save_dataset_info(data_path, [valid_index_start, test_index_start], 'dataset_' + model_name + '.txt')
    
    # Create RNN
    
    rnn = RNN(num_chars, hidden_size, num_layers, char_to_ix, ix_to_char, dropout).to(device)
    
    save_model_parameters(model_name, hidden_size, num_layers, num_chars, dropout)
    
    save_path = './CharRNN_' + model_name + '.pth'
        
    if load_chk:
        rnn.load_state_dict(torch.load(save_path))
        print("Model loaded successfully !!")
        print("----------------------------------------")
                
    # Loss Function and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
                
    # Prepare training loop    
    steps_done = 0

    iter_per_epoch = int(train_data_size / (seq_len + 1))

    input_seq = torch.zeros((seq_len, num_heads),dtype = data.dtype).to(device)
    target_seq = torch.zeros((seq_len, num_heads),dtype = data.dtype).to(device)
    
    for i_epoch in range(1, epochs+1):
        
        data_ptr = np.random.randint(train_data_size - seq_len - 1, size = num_heads)
                
        hidden_state = ( 
                         torch.zeros((num_layers, num_heads, hidden_size)).to(device),
                         torch.zeros((num_layers, num_heads, hidden_size)).to(device)
                       )
        
        running_loss = 0.0
        
        for i in tqdm(range(iter_per_epoch)):

            for j in range(num_heads):
                input_seq[:,j] = (train_data[data_ptr[j] : data_ptr[j]+seq_len]).squeeze()
                target_seq[:,j] = (train_data[data_ptr[j]+1 : data_ptr[j]+seq_len+1]).squeeze() 

            output, hidden_state = rnn(input_seq, hidden_state)

            loss = loss_fn(torch.squeeze(output.reshape(-1,num_chars)), torch.squeeze(target_seq.reshape(-1)))
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps_done += 1

            data_ptr += seq_len

            for j in range(num_heads):

                if data_ptr[j] + seq_len + 1 > train_data_size:

                    data_ptr[j] = ( data_ptr[j] + seq_len ) % train_data_size

                    hidden_state[0][:,j,:] = 0.0
                    hidden_state[1][:,j,:] = 0.0
            
            if steps_done % log_every == 0:
                validation_loss = calculate_loss_on_dataset(valid_data, rnn, num_heads, seq_len, device)
                log_metrics_to_file('log' + model_name + '.txt', [float(steps_done) / iter_per_epoch, loss.item(), validation_loss], steps_done == log_every)
                torch.save(rnn.state_dict(), save_path)

        print("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/iter_per_epoch))
        
        
def sample_from_rnn(model_name = 'vanilla', device = 'cuda', prompt = '\n', n = 1000):
    
    # Create Translation Tables
    
    with open('char_to_ix_' + model_name + '.pkl', 'rb') as file:    
        char_to_ix = pickle.load(file)
    with open('ix_to_char_' + model_name + '.pkl', 'rb') as file:    
        ix_to_char = pickle.load(file)
    
    # Create RNN
    
    hidden_size, num_layers, num_chars, dropout = load_model_parameters(model_name)
    
    rnn = RNN(num_chars, hidden_size, num_layers, char_to_ix, ix_to_char, dropout).to(device)
    
    save_path = './CharRNN_' + model_name + '.pth'
        
    rnn.load_state_dict(torch.load(save_path))
    
    with torch.no_grad():       
    
        start_seq = list(prompt)
        
        for i in range(len(start_seq)):
            
            ch = start_seq[i]
            start_seq[i] = rnn.char_to_ix[ch]

        start_seq = torch.tensor(start_seq).to(device)
        
        hidden_state =  ( 
                            torch.zeros((rnn.num_layers, 1, rnn.hidden_size)).to(device),
                            torch.zeros((rnn.num_layers, 1, rnn.hidden_size)).to(device)
                        )
                
        for i in range(len(prompt)):
            output, hidden_state = rnn(start_seq[i].reshape(1,1), hidden_state)
            
        output_string = ''    
        for i in range(n):
            
            softmax_output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(softmax_output)
            
            next_index = dist.sample()
            next_char = rnn.ix_to_char[next_index.item()]
            
            output_string += next_char
            
            output, hidden_state = rnn(next_index.reshape(1,1), hidden_state)
            
    return output_string

def load_dataset(model_name = 'vanilla', device = 'cuda', part = 'whole'):
    
    data_path, indices = load_dataset_info('dataset_' + model_name + '.txt')
    
    valid_index_start = indices[0]
    test_index_start = indices[1]
    
    # Prepare Dataset
    
    data = open(data_path, 'r').read()
    
    with open('char_to_ix_' + model_name + '.pkl', 'rb') as file:    
        char_to_ix = pickle.load(file)
    with open('ix_to_char_' + model_name + '.pkl', 'rb') as file:    
        ix_to_char = pickle.load(file)
    
    data = list(data)
    for i, ch in enumerate(data):
        data[i] = char_to_ix[ch]

    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(data, dim=1)
    
    if part == 'train':
        data = data[0:(valid_index_start-1)]
    elif part == 'valid':
        data = data[valid_index_start:(test_index_start-1)]
    elif part == 'test':
        data = data[test_index_start:]
        
    return data, char_to_ix, ix_to_char
            
def evaluate_rnn(model_name = 'vanilla', device = 'cuda', num_heads = 16, seq_len = 100):
    
    # Jetzt lesen wir die Metriken, die wir in der Trainingsschleife nach jeder Epoche gespeichert haben
    log_file = 'log' + model_name + '.txt'
    log_data = read_log_data(log_file)

    # Zuerst zeichnen wir eine Graphik, die die Lernkurven der Kostenfunktion ("loss") auf dem Trainings-
    # und Validierungsset darstellt
    plt.figure()
    plt.title('Loss')
    plt.plot(log_data['epoch'],log_data['train loss'],label="train loss")
    plt.plot(log_data['epoch'],log_data['validation loss'],label="validation loss")
    plt.xlabel('Epoche')
    plt.ylabel('Kostenfunktion')
    plt.legend(loc='best')
    plt.show()
    
    # Load dataset and translation tables
    
    data, char_to_ix, ix_to_char = load_dataset(model_name = model_name, device = device, part = 'test')
        
    # Create RNN
    
    hidden_size, num_layers, num_chars, dropout = load_model_parameters(model_name)
    
    rnn = RNN(num_chars, hidden_size, num_layers, char_to_ix, ix_to_char, dropout).to(device)
    
    save_path = './CharRNN_' + model_name + '.pth'
        
    rnn.load_state_dict(torch.load(save_path))    
    
    loss = calculate_loss_on_dataset(data, rnn, num_heads, seq_len, device)
    
    print('Loss on Test Set: %f' % loss)