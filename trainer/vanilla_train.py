"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
from __future__ import print_function

import time
from utils import get_accuracy
import trainer


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

    def train(self, train_loader, test_loader, epochs, criterion=None, writer=None):
        global loss_set
        if criterion == None:
            criterion = self.criterion
        model = self.model
        model.train()

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model, criterion)

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deom, eval_deoa, eval_subgroup_acc = self.evaluate(self.model, test_loader, criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deom, (eval_end_time - eval_start_time)))
            if self.record:
                train_loss, train_acc, train_deom, train_deoa, train_subgroup_acc = self.evaluate(self.model, train_loader, self.criterion)
                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('train_acc', train_acc, epoch)
                writer.add_scalar('train_deom', train_deom, epoch)
                writer.add_scalar('train_deoa', train_deoa, epoch)
                writer.add_scalar('eval_loss', eval_loss, epoch)
                writer.add_scalar('eval_acc', eval_acc, epoch)
                writer.add_scalar('eval_deom', eval_deom, epoch)
                writer.add_scalar('eval_deoa', eval_deoa, epoch)

                eval_contents = {}
                train_contents = {}
                for g in range(num_groups):
                    for l in range(num_classes):
                        eval_contents[f'g{g},l{l}'] = eval_subgroup_acc[g, l]
                        train_contents[f'g{g},l{l}'] = train_subgroup_acc[g, l]
                writer.add_scalars('eval_subgroup_acc', eval_contents, epoch)
                writer.add_scalars('train_subgroup_acc', train_contents, epoch)

            if self.scheduler is not None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, criterion=None):
        model.train()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            labels = targets

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == self.term-1:  # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
