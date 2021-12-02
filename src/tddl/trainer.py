import os
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.autograd import Variable
from ray import tune


class Trainer:
    def __init__(
        self, train_loader, test_loader, model, optimizer, writer, 
        scheduler=None, save=None, results={}, tuning=False,**kwargs
    ):
        self.train_data_loader = train_loader
        self.test_data_loader = test_loader

        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler is not None else None

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model.train()

        self.iteration = 0
        self.writer = writer
        self.results = results

        self.tuning = tuning

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
        steps = 0
        total_time = 0
        val_loss = 0.0

        if loader is None:
            loader = self.test_data_loader
        if loader == "train":
            loader = self.train_data_loader

        t = tqdm(loader, total=int(len(loader)))
        for i, (batch, label) in enumerate(t):
            with torch.no_grad():
                batch = batch.cuda() # TODO: change to .to_device(device)
                # t0 = time.time()
                # input = Variable(batch)
                output = self.model(Variable(batch)) #.cpu()  # commented cpu() out
                loss = self.criterion(output, Variable(label, requires_grad=False).cuda())
                val_loss += loss.cpu().numpy()
                # t1 = time.time()
                # total_time = total_time + (t1 - t0)
                pred = output.cpu().data.max(1)[1] # added .cpu()
                correct += pred.cpu().eq(label).sum() # TODO check if output.cpu() and pred.cpu() is necessary
                steps += label.size(0)

        self.model.train()
        accuracy = float(correct) / steps
        
        return accuracy, val_loss / steps


    def train(self, epochs=10):
        best_acc = 0
        for i in trange(epochs):
            self.train_epoch()
            if self.scheduler is not None:
                self.scheduler.step()

            acc, _ = self.test(loader=self.train_data_loader)
            self.writer.add_scalar("Accuracy/train", acc, i)

            valid_acc, valid_loss = self.test(loader=self.test_data_loader)
            self.writer.add_scalar("Accuracy/validation", valid_acc, i)
            self.writer.add_scalar("Loss/validation", valid_loss, i)
            if valid_acc > best_acc and self.save_best:
                self.results['best_epoch'] = i
                self.results['best_train_acc'] = acc
                self.results['best_valid_acc'] = valid_acc
                self.results['best_valid_loss'] = valid_loss
                torch.save(self.model, self.save_location + f"/{self.save_model_name}_best.pth")
                best_acc = valid_acc

            if self.save_every_epoch is not None and (i+1) % self.save_every_epoch == 0:
                torch.save(self.model, self.save_location + f"/{self.save_model_name}_{i}.pth")

            if np.isnan(valid_loss):
                valid_loss = 1e6
            
            #TODO tabulate results
            if self.tuning:
                with tune.checkpoint_dir(i) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(
                        (self.model.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict()), 
                        path,
                    )
            
                tune.report(loss=valid_loss, accuracy=valid_acc)

        if self.save_final:
            torch.save(self.model, self.save_location + f"/{self.save_model_name}_final.pth")
        
        return self.results

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
