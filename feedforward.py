import sys
import argparse
import random
import datetime

import pandas as pd
import numpy as np
import scipy.stats
import torch
import tqdm
import click

LOG2 = np.log(2)
HELDOUT_SIZE = .2
WORD_COL = 'predicate'
RATING_COL = 'response'

def pairs(xs):
    return zip(xs, xs[1:])

def initialized_linear(a, b):
    linear = torch.nn.Linear(a, b)
    torch.nn.init.xavier_uniform_(linear.weight)
    linear.bias.data.fill_(0.01)
    return linear

class FeedForward(torch.nn.Module):
    """ Generate feedforward network with given structure in terms of numbers of hidden units.
    Example: FeedForward([3,4,5,2]) will yield a network with structure:
    3 inputs ->
    ReLU ->
    4 hidden units ->
    ReLU ->
    5 hidden units ->
    ReLU ->
    2 outputs 
    """
    def __init__(self, structure, dropout=0, activation=torch.nn.ReLU(), transform=None):
        super().__init__()
        
        def layers():
            the_structure = list(structure)
            assert len(the_structure) >= 2
            for a, b in pairs(the_structure[:-1]):
                yield initialized_linear(a, b)
                yield torch.nn.Dropout(dropout)
                yield activation
            *_, penultimate, last = the_structure
            yield initialized_linear(penultimate, last)

        self.ff = torch.nn.Sequential(*layers())
        self.transform = identity if transform is None else transform

    def forward(self, x):
        return self.transform(self.ff(x))
                
def generate_xor_training_example(n):
    """ Generate n training examples for XOR function. """
    x1 = torch.Tensor([random.choice([0,1]) for _ in range(n)])
    x2 = torch.Tensor([random.choice([0,1]) for _ in range(n)])
    x = torch.stack([x1, x2], -1)
    y = (x1 != x2).float()
    return x,y

def epsilonify(x, eps=10**-5):
    """ Differentiably scale a value from [0,1] to [0+e, 1-e] """
    return (1-2*eps)*x + eps

def logistic(x):
    """ Differentiably squash a value from R to the interval (0,1) """
    return 1/(1+torch.exp(-x))

def logit(x):
    """ Differentiably blow up a value from the interval (0,1) to R """
    return torch.log(x) - torch.log(1-x)

def se_loss(y, yhat):
    """ Squared error loss.
    Appropriate loss for y and yhat \in R.
    Pushes yhat toward the mean of y. """
    return (y-yhat)**2

def bernoulli_loss(y, yhat):
    """ Appropriate loss for y \in {0,1}, yhat \in (0,1).
    But it's common to use this for y \in [0,1] and it still works.
    """
    return -(y*yhat.log() + (1-y)*(1-yhat).log())

def continuous_bernoulli_loss(x, lam):
    """ Appropriate loss for y \in [0,1], yhat \in (0,1).
    Technically more correct than Bernoulli loss for that case, 
    but more complex/annoying and potentially numerically unstable.
    See https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution
    """
    logZ = LOG2 + torch.log(torch.atanh(1-2*lam) / (1 - 2*lam))
    return logZ + bernoulli_loss(x, lam)

def continuous_bernoulli_mean(lam):
    """ Expectation of a Continuous Bernoulli distribution.
    See https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution 
    """
    return lam/(2*lam - 1) + 1/(2 * torch.atanh(1-2*lam))

def beta_loss(x, alpha, beta):
    unnorm = (alpha - 1)*torch.log(x) + (beta - 1)*torch.log(1-x)
    logZ = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return unnorm - logZ

def train_xor_example(batch_size=10, num_epochs=1000, print_every=100, structure=[2,3,1], **kwds):
    """ Example: Train a network to reproduce the XOR function. """
    net = FeedForward(structure)
    opt = torch.optim.Adam(params=net.parameters(), **kwds)
    for i in range(num_epochs):
        opt.zero_grad()
        x, y = generate_xor_training_example(batch_size)
        yhat = net(x).squeeze(-1)
        loss = se_loss(y, yhat).mean()
        if i % print_every == 0:
            print("epoch %d, loss = %s" % (i, str(loss.item())))
        loss.backward()
        opt.step()
    return net

def read_vectors(filename):
    print("Loading vectors from %s" % filename, file=sys.stderr)
    d = {}
    with open(filename) as infile:
        n, ndim = next(infile).strip().split()
        n = int(n)
        ndim = int(ndim)
        lines = list(infile)
        for line in tqdm.tqdm(lines):
            parts = line.strip().split(" ")
            numbers = list(map(float, parts[-ndim:]))
            vec = torch.Tensor(numbers)
            wordparts = parts[:-ndim]
            word = " ".join(wordparts)
            d[word] = vec
    print("Loaded.", file=sys.stderr)
    return d

def identity(x):
    return x

def run_classifier(net, vectors, words):
    x = torch.stack([vectors[word.lower()] for word in words])
    yhat = net(x).squeeze(-1)
    return yhat.detach()

def train_classifier(vectors,
                     words,
                     responses,
                     dev_words=None,
                     dev_responses=None,
                     structure=[300, 128, 1],
                     activation=torch.nn.ReLU(),
                     dropout=0,
                     loss=bernoulli_loss,
                     y_transform=None,
                     yhat_transform=logistic,
                     y_inverse_transform=None,
                     batch_size=1,
                     num_epochs=100,
                     print_every=1000,
                     **kwds):

    """
    For linear regression on raw values, set
       y_transform=None
       y_inverse_transform=None
       yhat_transform=None
       loss=se_loss
    For linear regression on log odds, set
       y_transform=logit
       y_inverse_transform=logistic
       yhat_transform=None
       loss=se_loss
    For quasi-logistic-regression, set
       y_transform=None
       y_inverse_transform=None
       yhat_transform=logistic
       loss=bernoulli_loss
    For mathematically correct logistic regression, set
       y_transform=None
       y_inverse_transform=None
       yhat_transform=logistic
       loss=continuous_bernoulli_loss
    """
    
    assert len(words) == len(responses)
    
    if y_transform is None:
        y_transform = identity
    if y_inverse_transform is None:
        y_inverse_transform = identity
    words = np.array(words)
    responses = np.array(responses)
    dev_words = np.array(dev_words)
    dev_responses = np.array(dev_responses)
    indices = range(len(words))

    diagnostics = {
        'epoch': [],
        'train_loss': [],
        'dev_loss': [],
        'train_r': [],
        'train_rho': [],
        'dev_r': [],
        'dev_rho': [],
    }

    train_x = torch.stack([vectors[word.lower()] for word in words])
    train_y = y_transform(torch.Tensor(responses))

    dev_x = torch.stack([vectors[word.lower()] for word in dev_words])
    dev_y = y_transform(torch.Tensor(dev_responses))
    
    net = FeedForward(structure, activation=activation, dropout=dropout, transform=yhat_transform)
    opt = torch.optim.Adam(params=net.parameters(), **kwds)
    for i in range(1, num_epochs+1):
        opt.zero_grad()
        if batch_size is None:
            batch = indices
        else:
            batch = random.sample(indices, batch_size)
        words_batch = words[batch]
        y = y_transform(torch.Tensor(responses[batch]))
        x = torch.stack([vectors[word.lower()] for word in words_batch])
        yhat = net(x).squeeze(-1)
        the_loss = loss(y, yhat).mean()
        if i % print_every == 0:
            diagnostics['epoch'].append(i)
            
            train_yhat = net(train_x).squeeze(-1)
            train_loss = loss(train_y, train_yhat).mean().item()
            (train_r, *_), (train_rho, *_) = evaluate_estimates(words, responses, train_yhat.detach())            
            diagnostics['train_loss'].append(train_loss)
            diagnostics['train_r'].append(train_r)
            diagnostics['train_rho'].append(train_rho)

            if dev_words is not None:
                dev_yhat = net(dev_x).squeeze(-1)
                dev_loss = loss(dev_y, dev_yhat).mean().item()
                (dev_r, *_), (dev_rho, *_) = evaluate_estimates(dev_words, dev_responses, dev_yhat.detach())
                diagnostics['dev_loss'].append(dev_loss)
                diagnostics['dev_r'].append(dev_r)
                diagnostics['dev_rho'].append(dev_rho)
                print("epoch %d, train loss = %s, dev loss = %s" % (i, str(train_loss), str(dev_loss)))
            else:
                print("epoch %d, train loss = %s" % (i, str(train_loss)))
                
        the_loss.backward()
        opt.step()

    return net.eval(), pd.DataFrame(diagnostics)

def evaluate_estimates(words, responses, estimates):
    """ Evaluate Spearman correlation between truth and estimate """
    df = pd.DataFrame({
        WORD_COL: words,
        RATING_COL: responses,
        'estimate': estimates,
    })
    means = df.groupby([WORD_COL]).mean().reset_index()
    r = scipy.stats.pearsonr(means[RATING_COL], means['estimate'])
    rho = scipy.stats.spearmanr(means[RATING_COL], means['estimate'])
    return r, rho

def train_dev_test_split(data, heldout_prop):
    words = list(set(data[WORD_COL]))
    random.shuffle(words)    
    n = len(words)
    num_train = n - int(heldout_prop * n)
    print("Training set: %d / %d words" % (num_train, n), file=sys.stderr)   
    training_words = words[:num_train]
    training_mask = np.array([word in training_words for word in data[WORD_COL]])

    heldout_words = words[num_train:]
    num_dev = int((n - num_train) / 2)
    dev_words = heldout_words[:num_dev]
    dev_mask = np.array([word in dev_words for word in data[WORD_COL]])
    print("   Dev set: %d / %d words" % (num_dev, n), file=sys.stderr)       
    
    test_words = heldout_words[num_dev:]
    test_mask = np.array([word in test_words for word in data[WORD_COL]])
    print("  Test set: %d / %d words" % (len(test_words), n), file=sys.stderr)          

    return data[training_mask].copy(), data[dev_mask].copy(), data[test_mask].copy()

def main(vectors_filename, subj_filename, split_seed=1, **kwds):
    vectors = read_vectors(vectors_filename)
    df = pd.read_csv(subj_filename)
    df[RATING_COL] = epsilonify(df[RATING_COL])
    random.seed(split_seed)
    train, dev, test = train_dev_test_split(df, HELDOUT_SIZE)
    net, diagnostics = train_classifier(
        vectors,
        train[WORD_COL],
        train[RATING_COL],
        dev[WORD_COL],
        dev[RATING_COL],
        **kwds)
    train['estimate'] = run_classifier(net, vectors, train[WORD_COL])
    dev['estimate'] = run_classifier(net, vectors, dev[WORD_COL])
    test['estimate'] = run_classifier(net, vectors, test[WORD_COL])
    print("Training: ", evaluate_estimates(
        train[WORD_COL], train[RATING_COL], train['estimate']))
    print("Dev: ", evaluate_estimates(
        dev[WORD_COL], dev[RATING_COL], dev['estimate']))
    save_output(kwds, net, train, dev, test)
    return net, (train, dev, test), diagnostics

def save_output(params, model, train, dev, test):
    date = datetime.datetime.now()
    s = "_".join("%s=%s" % (str(k), str(v)) for k, v in params) + "_" + str(date)
    filename = "output/subj_classifier_%s.pickle" % s
    torch.save(model, filename)

    train.to_csv("output/train_%s.csv" % s)
    dev.to_csv("output/dev_%s.csv" % s)
    test.to_csv("output/test_%s.csv" % s)


# Optimization notes (dev Spearman correlations)
# Loss functions -- 1000 epochs; default opt parameters; [300, 50, 1]
# Raw SE: 0.75
# Logit SE: 0.80
# Bernoulli: 0.82
# Continuous Bernoulli: 0.82
# So, go with Bernoulli.
# Activation: [ReLU, Tanh] --- Tanh is terrible (mid-60s correlations). Only try ReLU.

# Viable hyperparameters:
# Number of hidden layers: 0, 1, 2 --- 0 layers does the best on dev loss, but a lot worse on train loss...
# Number of hidden units: 16, 32, 64, 128, 256, 512
# Learning rate: [.1, .01, .001, .0001]
# Batch size: [8, 16, 32, 64, all]
# Dropout: [0, .1, .2, .3]

# EARLY STOPPING?
    
if __name__ == '__main__':
    main(*sys.argv[1:])