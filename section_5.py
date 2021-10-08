import math
import torch
import pickle
import itertools
import numpy as np
from tqdm import tqdm

from util.data import get_data, normalize_data
from util.kernel import sanitise, increment_kernel, complexity, invert_bound
from util.trainer import train_network, SimpleNet
from util.nero import Nero

### Dependent variables
num_train_examples = 5
num_test_examples = 50
random_labels = False
binary_digits = True
depth = 7
width = 10000

alpha_list = [0.01, 0.1, 1.0, 10.0, 100.0]

### Data hyperparameters
batch_size = 50

### Training hyperparameters
init_lr = 0.01
decay = 0.9

### Estimator hyperparameters
num_samples = 10**7
num_estimator_batches = 10**3
cuda = True

### Get data

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

data = get_data( num_train_examples=num_train_examples,
                 num_test_examples=num_test_examples,
                 batch_size=batch_size, 
                 random_labels=random_labels, 
                 binary_digits=binary_digits )

full_batch_train_loader, full_batch_test_loader, train_loader, test_loader = data

print("\nTraining labels")
for data, target in full_batch_train_loader:
    data, target = normalize_data(data, target)
    print(target)

print("\nTraining networks at different target scales")
results = {}
for alpha in alpha_list:
    print(f"\ntarget scale {alpha}")
    train_acc_list = []
    test_acc_list = []
    for _ in range(100):
        train_acc, test_acc, model = train_network( train_loader = train_loader,
                                                    test_loader = test_loader,
                                                    depth=depth,
                                                    width=width,
                                                    init_lr=init_lr, 
                                                    decay=decay,
                                                    break_on_fit=False,
                                                    target_scale=alpha )
        print(f"Train acc: {train_acc[-1]}")
        print(f"Test acc: {test_acc}")
        train_acc_list.append(train_acc[-1])
        test_acc_list.append(test_acc)
    results[alpha] = (train_acc_list, test_acc_list)
fname = 'logs/holdout/varying-target-scales.pickle'
pickle.dump( results, open( fname, "wb" ) )

print("\nEstimating average test acc using NNGP")
acc_estimate_list = []
for _ in range(5):

    ### Kernel arithmetic
    data, target = list(full_batch_train_loader)[0]
    if cuda: data, target = data.cuda(), target.cuda()
    data, target = normalize_data(data, target)

    c = target.float()
    sigma = sanitise(torch.mm(data, data.t()) / data.shape[1])
    assert ( sigma == sigma.t() ).all()
    n = sigma.shape[0]

    for _ in range(depth-1):
        sigma = increment_kernel(sigma)
        assert ( sigma == sigma.t() ).all()
    
    p_estimate = 0
    for _ in tqdm(range(num_estimator_batches)):
        c0, c1 = complexity( sigma, c, num_samples )
        p_estimate += math.exp(-c0)
    p_estimate /= num_estimator_batches

    print(p_estimate)

    ### Kernel arithmetic with test point
    train_data, train_target = list(full_batch_train_loader)[0]
    test_data, test_target = list(full_batch_test_loader)[0]

    prob_sum = 0
    for test_idx in tqdm(range(test_data.shape[0])):
        test_datum = test_data[test_idx,:,:,:].unsqueeze(dim=0)
        test_targ = test_target[test_idx].unsqueeze(dim=0)

        data = torch.cat((train_data, test_datum), dim=0)
        target = torch.cat((train_target, test_targ), dim=0)

        if cuda: data, target = data.cuda(), target.cuda()
        data, target = normalize_data(data, target)

        c = target.float()
        sigma = sanitise(torch.mm(data, data.t()) / data.shape[1])
        assert ( sigma == sigma.t() ).all()
        n = sigma.shape[0]

        for _ in range(depth-1):
            sigma = increment_kernel(sigma)
            assert ( sigma == sigma.t() ).all()
            
        c0_, c1_ = complexity( sigma, c, num_samples )
        prob_sum += math.exp(-c0_)
    prob_sum /= test_data.shape[0]

    acc_estimate = prob_sum / p_estimate
    acc_estimate_list.append(acc_estimate)
fname = 'logs/holdout/nngp-average-acc.pickle'
pickle.dump( acc_estimate_list, open( fname, "wb" ) )


print("\nCompute the NNGP conditional accuracy")
results = {}
for alpha in alpha_list:
    print(f"\ntarget scale {alpha}")

    train_data, train_target = list(full_batch_train_loader)[0]
    test_data, test_target = list(full_batch_test_loader)[0]

    data = torch.cat((train_data, test_data), dim=0)
    target = torch.cat((train_target, test_target), dim=0)

    if cuda: data, target = data.cuda(), target.cuda()
    data, target = normalize_data(data, target)

    sigma = sanitise(torch.mm(data, data.t()) / data.shape[1]).cpu()
    assert ( sigma == sigma.t() ).all()

    for _ in range(depth-1):
        sigma = increment_kernel(sigma)
        assert ( sigma == sigma.t() ).all()

    sigma_bb = sigma[:num_train_examples, :num_train_examples]
    sigma_ab = sigma[num_train_examples:, :num_train_examples]
    sigma_ba = sigma[:num_train_examples, num_train_examples:]
    sigma_aa = sigma[num_train_examples:, num_train_examples:]

    x_b = target[:num_train_examples].float().cpu()
    x_a = target[num_train_examples:].float().cpu()

    sigma_bb_inv = torch.cholesky_inverse(torch.cholesky(sigma_bb))

    cond_mu = sigma_ab @ sigma_bb_inv @ x_b * alpha
    cond_sigma = sigma_aa - sigma_ab @ sigma_bb_inv @ sigma_ba
    
    m = torch.distributions.multivariate_normal.MultivariateNormal(cond_mu, cond_sigma)
    test_acc_list = []

    for _ in range(100):
        pred = m.sample()
        
        correct = (x_a == pred.sign()).sum().item()
        total = x_a.shape[0]
        test_acc_list.append(correct/total)
    results[alpha] = test_acc_list

fname = 'logs/holdout/varying-target-scales-nngp.pickle'
pickle.dump( results, open( fname, "wb" ) )

print("\nTest acc of rejection sampled networks")
train_acc_list = []
test_acc_list = []
pred_scale_list = []
for _ in tqdm(range(1000)):
    model = SimpleNet(depth, width).cuda()
    for p in model.parameters():
        p.data = torch.randn_like(p) / math.sqrt(p.shape[1])

    model.eval()
    correct = 0
    total = 0

    for data, target in full_batch_train_loader:
        data, target = (data.cuda(), target.cuda())
        data, target = normalize_data(data, target)

        y_pred = model(data).squeeze()
        correct += (target.float() == y_pred.sign()).sum().item()
        total += target.shape[0]

    train_acc = correct/total

    if train_acc == 1.0 or train_acc == 0.0:
        pred_scale = y_pred.abs().mean().item()

        correct = 0
        total = 0

        for data, target in tqdm(full_batch_test_loader):
            data, target = (data.cuda(), target.cuda())
            data, target = normalize_data(data, target)
            
            y_pred = model(data).squeeze()
            correct += (target.float() == y_pred.sign()).sum().item()
            total += target.shape[0]

        test_acc = correct/total

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        pred_scale_list.append(pred_scale)

fname = 'logs/holdout/rejection-sampling-networks.pickle'
pickle.dump( (train_acc_list, test_acc_list, pred_scale_list), open( fname, "wb" ) )