import time
import torch
from tddl.data.loaders import train_loader, test_loader
from torch.autograd import Variable
from tqdm import tqdm, trange


class Trainer:
    def __init__(self, train_path, test_path, model, optimizer, writer):
        self.train_data_loader = train_loader(train_path)
        self.test_data_loader = test_loader(test_path)

        self.optimizer = optimizer

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

        self.writer = writer

    def test(self, loader=None):
        self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0
        total_time = 0

        if loader is None:
            loader = self.test_data_loader
        t = tqdm(loader, total=int(len(loader)))

        for i, (batch, label) in enumerate(t):
            batch = batch.cuda()
            # t0 = time.time()
            output = self.model(Variable(batch)).cpu() # TODO
            # t1 = time.time()
            # total_time = total_time + (t1 - t0)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)
            # t.set_postfix(acc=float(correct) / total)
        
        self.model.train()
        accuracy = float(correct) / total

        return accuracy
        # print("Accuracy :", accuracy)
        # print("Average prediction time", float(total_time) / (i + 1), i + 1)


    def train(self, epoches=10):
        for i in trange(epoches):
            print("Epoch: ", i)
            self.train_epoch()
            acc = self.test(loader=self.train_data_loader)
            self.writer.add_scalar("Accuracy/train", acc, i)
        # writer.close()
        # print("Finished fine tuning.")
        

    def train_batch(self, batch, label):
        self.model.zero_grad()
        input = Variable(batch)
        loss = self.criterion(self.model(input), Variable(label))
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        

    def train_epoch(self):
        t = tqdm(self.train_data_loader, total=int(len(self.test_data_loader)))
        for i, (batch, label) in enumerate(t):
            loss = self.train_batch(batch.cuda(), label.cuda())
            t.set_postfix(loss=loss)
            self.writer.add_scalar('Loss/train', loss, i)
