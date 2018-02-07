from scipy import io
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import kNN


def compute_accuracy(test_att, test_visual, test_id, test_label):
	# test_att: [2993, 312]
	# test viaual: [2993, 1024]
	# test_id: att2label [50]
	# test_label: x2label [2993]
	test_att = Variable(torch.from_numpy(test_att).float().cuda())
	att_pred = forward(test_att)
	outpred = [0] * 2933
	test_label = test_label.astype("float32")

	# att_pre [50, 1024],
	# test_visual: [2993, 1024]
	# test_id : [50]

	for i in range(2933):
		outputLabel = kNN.kNNClassify(test_visual[i, :], att_pred.cpu().data.numpy(), test_id, 1)
		outpred[i] = outputLabel
	outpred = np.array(outpred)
	acc = np.equal(outpred, test_label).mean()

	return acc


def data_iterator():
	""" A simple data iterator """
	batch_idx = 0
	while True:
		# shuffle labels and features
		idxs = np.arange(0, len(train_x))
		np.random.shuffle(idxs)
		shuf_visual = train_x[idxs]
		shuf_att = train_att[idxs]
		batch_size = 100

		for batch_idx in range(0, len(train_x), batch_size):
			visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]
			visual_batch = visual_batch.astype("float32")
			att_batch = shuf_att[batch_idx:batch_idx + batch_size]

			att_batch = Variable(torch.from_numpy(att_batch).float().cuda())
			visual_batch = Variable(torch.from_numpy(visual_batch).float().cuda())
			yield att_batch, visual_batch


f = io.loadmat('CUB_data/train_attr.mat')
train_att = np.array(f['train_attr'])
print('train attr:', train_att.shape)

f = io.loadmat('CUB_data/train_cub_googlenet_bn.mat')
train_x = np.array(f['train_cub_googlenet_bn'])
print('train x:', train_x.shape)

f = io.loadmat('CUB_data/test_cub_googlenet_bn.mat')
test_x = np.array(f['test_cub_googlenet_bn'])
print('test x:', test_x.shape)

f = io.loadmat('CUB_data/test_proto.mat')
test_att = np.array(f['test_proto'])
print('test att:', test_att.shape)

f = io.loadmat('CUB_data/test_labels_cub.mat')
test_x2label = np.squeeze(np.array(f['test_labels_cub']))
print('test x2label:', test_x2label)

f = io.loadmat('CUB_data/testclasses_id.mat')
test_att2label = np.squeeze(np.array(f['testclasses_id']))
print('test att2label:', test_att2label)

w1 = Variable(torch.FloatTensor(312, 700).cuda(), requires_grad=True)
b1 = Variable(torch.FloatTensor(700).cuda(), requires_grad=True)
w2 = Variable(torch.FloatTensor(700, 1024).cuda(), requires_grad=True)
b2 = Variable(torch.FloatTensor(1024).cuda(), requires_grad=True)

# must initialize!
w1.data.normal_(0, 0.02)
w2.data.normal_(0, 0.02)
b1.data.fill_(0)
b2.data.fill_(0)


def forward(att):
	a1 = F.relu(torch.mm(att, w1) + b1)
	a2 = F.relu(torch.mm(a1, w2) + b2)

	return a2


def getloss(pred, x):
	loss = torch.pow(x - pred, 2).sum()
	loss /= x.size(0)
	return loss


optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=1e-5, weight_decay=1e-2)

# # Run
iter_ = data_iterator()
for i in range(1000000):
	att_batch_val, visual_batch_val = next(iter_)

	pred = forward(att_batch_val)
	loss = getloss(pred, visual_batch_val)

	optimizer.zero_grad()
	loss.backward()
	# gradient clip makes it converge much faster!
	torch.nn.utils.clip_grad_norm([w1, b1, w2, b2], 1)
	optimizer.step()

	if i % 1000 == 0:
		print(compute_accuracy(test_att, test_x, test_att2label, test_x2label))
