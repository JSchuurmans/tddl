import time
import torch
# from tddl.data.loaders import gettrain_loader, test_loader
from torch.autograd import Variable
from tqdm import tqdm, trange


class Trainer:
    def __init__(self, train_loader, test_loader, model, optimizer, writer, scheduler=None, save=None, **kwargs):
        self.train_data_loader = train_loader
        self.test_data_loader = test_loader

        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler is not None else None

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

        self.iteration = 0
        self.writer = writer

        if save is None:
            self.save_every_epoch = None
            self.save_location = "./"
            self.save_best = True
            self.save_final = True
            self.save_model_name = "model"
        else:  # TODO make this not error when dict save is partially filled
            self.save_every_epoch = save["save_every_epoch"]
            self.save_location = save["save_location"]
            self.save_best = save["save_best"]
            self.save_final = save["save_final"]
            self.save_model_name = save["save_model_name"]

    def test(self, loader=None):
        self.model.cuda()
        self.model.eval()
        correct = 0
        total = 0
        total_time = 0

        if loader is None:
            loader = self.test_data_loader
        if loader == "train":
            loader = self.train_data_loader

        t = tqdm(loader, total=int(len(loader)))
        for i, (batch, label) in enumerate(t):
            batch = batch.cuda()
            # t0 = time.time()
            output = self.model(Variable(batch)).cpu()  # TODO
            # t1 = time.time()
            # total_time = total_time + (t1 - t0)
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        self.model.train()
        accuracy = float(correct) / total

        return accuracy

    def train(self, epochs=10):
        prev_acc = 0
        for i in trange(epochs):
            # print("Epoch: ", i)
            self.train_epoch()
            if self.scheduler is not None:
                self.scheduler.step()

            acc = self.test(loader=self.train_data_loader)
            self.writer.add_scalar("Accuracy/train", acc, i)

            valid_acc = self.test(loader=self.test_data_loader)
            self.writer.add_scalar("Accuracy/validation", valid_acc, i)
            if valid_acc > prev_acc and self.save_best:
                torch.save(self.model, self.save_location + f"/{self.save_model_name}_best")

            if self.save_every_epoch is not None:
                if (i+1) % self.save_every_epoch == 0:
                    torch.save(self.model, self.save_location + f"/{self.save_model_name}_{i}")

        if self.save_final:
            torch.save(self.model, self.save_location + f"/{self.save_model_name}_final")

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
            self.writer.add_scalar('Loss/train', loss, self.iteration)
            self.iteration += 1


