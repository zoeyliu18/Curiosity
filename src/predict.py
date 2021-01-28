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

from transformers import BertTokenizer, BertModel

LOG2 = np.log(2)
HELDOUT_SIZE = .4
WORD_COL = 'predicate'
RATING_COL = 'response'
CLASS = 'class'
BINARY = False

bert_model = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
bert_model = BertModel.from_pretrained(bert_model, output_hidden_states=True)
bert_model.eval()

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

def run_classifier(net, words):
    test_emb = []
    for word in words:
        word = word.lower()
        emb, l = get_sentence_bert(word,
                            bert_tokenizer,
                            bert_model,
                            layer=11,
                            GPU=False,
                            LSTM=False,
                            max_seq_len=30,
                            is_single=True)
        test_emb.append(emb)
    x = torch.stack(test_emb)
    yhat = net(x).squeeze(-1).detach()
    if BINARY:
        pred = []
        for i in range(len(yhat)):
            pred.append(' '.join(str(c) for c in yhat.tolist()[i]))
        return pred
    else:
        return yhat.detach()

def get_sentence_bert(s, bert_tokenizer, bert_model, layer = 12, GPU=False, LSTM=False, max_seq_len=None, is_single=True):
    s = "[CLS] " + s + " [SEP]"
    tokenized_text = bert_tokenizer.tokenize(s)
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    bert_output = torch.zeros((1,max_seq_len, bert_model.config.hidden_size))
    if GPU:
      tokens_tensor = tokens_tensor.cuda()
      segments_tensors = segments_tensors.cuda()
      bert_output = bert_output.cuda()      
    sl = min(len(indexed_tokens), max_seq_len)
    with torch.no_grad():
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        bert_output[:, :sl, :] = outputs[2][layer][:,:sl, :]
        bert_output = bert_output.squeeze()  # (max_seq_len, 768)
    if GPU:
      bert_output = bert_output.cpu()
    if LSTM:
        return bert_output, sl
    else:
        bert_mean = torch.mean(bert_output, axis=0)
        return bert_mean, sl


def train_classifier(words,
                     responses,
                     dev_words=None,
                     dev_responses=None,
                     activation=torch.nn.ReLU(),
                     dropout=0,
                     loss=bernoulli_loss,
                    # loss=torch.nn.BCELoss(),
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
    structure = None
    if BINARY:
        structure=[768, 128, 2]
    else:
        structure=[768, 128, 1]
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
    train_emb = []
    for word in words:
        word = word.lower()
        emb, l = get_sentence_bert(word,
                                bert_tokenizer,
                                bert_model,
                                layer=12,
                                GPU=False,
                                LSTM=False,
                                max_seq_len=30,
                                is_single=True)
        train_emb.append(emb)
    train_x = torch.stack(train_emb)
    train_y = None
    if BINARY:
        train_y = torch.Tensor(responses)
    else:
        train_y = y_transform(torch.Tensor(responses))
    dev_emb = []
    for word in words:
        word = word.lower()
        emb, l = get_sentence_bert(word,
                                bert_tokenizer,
                                bert_model,
                                layer=12,
                                GPU=False,
                                LSTM=False,
                                max_seq_len=30,
                                is_single=True)
        dev_emb.append(emb)
    dev_x = torch.stack(dev_emb)
    dev_y = None
    if BINARY:
        dev_y = torch.Tensor(responses)
    else:
        dev_y = y_transform(torch.Tensor(responses))      
    net = FeedForward(structure, activation=activation, dropout=dropout, transform=yhat_transform)
    opt = torch.optim.Adam(params=net.parameters(), **kwds)
    for i in range(1, num_epochs+1):
        opt.zero_grad()
        if batch_size is None:
            batch = indices
        else:
            batch = random.sample(indices, batch_size)
        words_batch = words[batch]
        y = None
        if BINARY:
            y = torch.Tensor(responses[batch])
        else:
            y = y_transform(torch.Tensor(responses[batch]))
        batch_emb = []
        for word in words_batch:
            word = word.lower()
            emb, l = get_sentence_bert(word,
                                bert_tokenizer,
                                bert_model,
                                layer=11,
                                GPU=False,
                                LSTM=False,
                                max_seq_len=30,
                                is_single=True)
            batch_emb.append(emb)
        x = torch.stack(batch_emb)
        yhat = net(x).squeeze(-1)
        the_loss = loss(y, yhat).mean()
        if i % print_every == 0:
            diagnostics['epoch'].append(i)            
            train_yhat = net(train_x).squeeze(-1)
            train_loss = loss(train_y, train_yhat).mean().item()
            if BINARY:
                train_acc = evaluate_classification(words, responses, train_yhat)
                diagnostics['train_loss'].append(train_loss)
                diagnostics['train_acc'].append(train_acc)
            else:
                (train_r, *_), (train_rho, *_) = evaluate_estimates(words, responses, train_yhat.detach())            
                diagnostics['train_loss'].append(train_loss)
                diagnostics['train_r'].append(train_r)
                diagnostics['train_rho'].append(train_rho)
            if dev_words is not None:
                dev_yhat = net(dev_x).squeeze(-1)
                dev_loss = loss(dev_y, dev_yhat).mean().item()
                if BINARY:
                    dev_acc = evaluate_classification(dev_words, dev_responses, dev_yhat)
                    diagnostics['dev_loss'].append(dev_loss)
                    diagnostics['dev_acc'].append(dev_acc)
                else:
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
    print(means)
    r = scipy.stats.pearsonr(means[RATING_COL], means['estimate'])
    rho = scipy.stats.spearmanr(means[RATING_COL], means['estimate'])
    return r, rho

def evaluate_classification(words, classes, estimates):
    """ Evaluate classification accuracy """
    pred = []
    for tok in estimates.tolist():
        pred.append(tok.split())
    gold = classes.tolist()
    acc = 0
    for i in range(len(pred)):
        new_pred = []
        for c in pred[i]:
            new_pred.append(float(c))
        if np.argmax(new_pred) == gold[i]:
            acc+=1
        print(new_pred, gold[i])
    return round(acc * 100 / len(pred), 2)


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

#def main(vectors_filename, subj_filename, split_seed=1, **kwds):
#    vectors = read_vectors(vectors_filename)
def main(subj_filename, split_seed=1, **kwds):
    df = pd.read_csv(subj_filename)
    df[RATING_COL] = epsilonify(df[RATING_COL])
    df[CLASS] = df[CLASS].astype(np.int)
    random.seed(split_seed)
    train, dev, test = train_dev_test_split(df, HELDOUT_SIZE)
    net, diagnostics = None, None
    if BINARY:
        net, diagnostics = train_classifier(
            train[WORD_COL],
            train[CLASS],
            dev[WORD_COL],
            dev[RATING_COL],
            **kwds)
    else:
        net, diagnostics = train_classifier(
            train[WORD_COL],
            train[RATING_COL],
            dev[WORD_COL],
            dev[RATING_COL],
            **kwds)
    train['estimate'] = run_classifier(net, train[WORD_COL])
    print(train['estimate'])
    dev['estimate'] = run_classifier(net, dev[WORD_COL])
    test['estimate'] = run_classifier(net, test[WORD_COL])
    if BINARY:
        print(train[CLASS])
        print(train['estimate'])
        print("Training: ", evaluate_classification(
            train[WORD_COL], train[CLASS], train['estimate']))
        print("Dev: ", evaluate_classification(
            dev[WORD_COL], dev[CLASS], dev['estimate']))
    else:
        print('non-binary')
        print(train[RATING_COL])
        print("Training: ", evaluate_estimates(
            train[WORD_COL], train[RATING_COL], train['estimate']))
        print("Dev: ", evaluate_estimates(
            dev[WORD_COL], dev[RATING_COL], dev['estimate']))
    save_output(kwds, net, train, dev, test)
    return net, (train, dev, test), diagnostics

def save_output(params, model, train, dev, test):
    date = datetime.datetime.now()
    s = "_".join("%s=%s" % (str(k), str(v)) for k, v in params) + "_" + str(date)
    filename = "classifier_%s.pickle" % s
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
# To add layers, extent structure; e.g. [768, 128, 1] -- > [768, 128, 128, 1]
# Number of hidden units: 16, 32, 64, 128, 256, 512
# Learning rate: [.1, .01, .001, .0001]
# Batch size: [8, 16, 32, 64, all]
# Dropout: [0, .1, .2, .3]

# EARLY STOPPING?
    
if __name__ == '__main__':
    main(*sys.argv[1:])
