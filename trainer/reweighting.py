"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
from __future__ import print_function
import torch
import torch.nn as nn
import time
from utils import get_accuracy
import trainer
from torch.utils.data import DataLoader


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

        self.eta = args.eta
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.reweighting_target_criterion = args.reweighting_target_criterion
        self.slmode = True if args.sv < 1 else False
        self.version = args.version

    def train(self, train_loader, test_loader, epochs, dummy_loader=None, writer=None):
        model = self.model
        model.train()
        num_groups = train_loader.dataset.num_groups
        num_classes = train_loader.dataset.num_classes

        extended_multipliers = torch.zeros((num_groups, num_classes))
        if self.cuda:
            extended_multipliers = extended_multipliers.cuda()
        _, Y_train, S_train = self.get_statistics(train_loader.dataset, batch_size=self.batch_size,
                                                  num_workers=self.num_workers)

        eta_learning_rate = self.eta
        print('eta_learning_rate : ', eta_learning_rate)
        n_iters = self.iteration
        print('n_iters : ', n_iters)

        violations = 0

        for iter_ in range(n_iters):
            start_t = time.time()
            weight_set = self.debias_weights(Y_train, S_train, extended_multipliers, num_groups, num_classes)

            for epoch in range(epochs):
                self._train_epoch(epoch, train_loader, model, weight_set)
                eval_start_time = time.time()
                eval_loss, eval_acc, eval_deom, eval_deoa, eval_subgroup_acc = self.evaluate(self.model, test_loader, self.criterion)
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
                    writer.add_scalar('eval_deopm', eval_deom, epoch)
                    writer.add_scalar('eval_deopa', eval_deoa, epoch)

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

            end_t = time.time()
            train_t = int((end_t - start_t) / 60)
            print('Training Time : {} hours {} minutes / iter : {}/{}'.format(int(train_t / 60), (train_t % 60),
                                                                              (iter_ + 1), n_iters))

            Y_pred_train, Y_train, S_train = self.get_statistics(train_loader.dataset, batch_size=self.batch_size,
                                                                 num_workers=self.num_workers, model=model)

            if self.reweighting_target_criterion == 'dp':
                acc, violations = self.get_error_and_violations_DP(Y_pred_train, Y_train, S_train, num_groups, num_classes)
            elif self.reweighting_target_criterion == 'eo':
                acc, violations = self.get_error_and_violations_EO(Y_pred_train, Y_train, S_train, num_groups, num_classes)
            extended_multipliers -= eta_learning_rate * violations

    def _train_epoch(self, epoch, train_loader, model, weight_set):
        model.train()

        running_acc = 0.0
        running_loss = 0.0
        avg_batch_time = 0.0

        for i, data in enumerate(train_loader):
            batch_start_time = time.time()
            # Get the inputs
            inputs, _, groups, targets, indexes = data
            labels = targets
            labels = labels.long()

            weights = weight_set[indexes[0]]

            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()
                groups = groups.cuda()

            outputs = model(inputs)

            loss = torch.mean(weights * nn.CrossEntropyLoss(reduction='none')(outputs, labels))
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            # zero the parameter gradients + backward + optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_end_time = time.time()
            avg_batch_time += batch_end_time - batch_start_time

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                avg_batch_time = 0.0

        last_batch_idx = i
        return last_batch_idx

    def get_statistics(self, dataset, batch_size=128, num_workers=2, model=None):

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False)

        if model is not None:
            model.eval()

        Y_pred_set = []
        Y_set = []
        S_set = []
        total = 0
        for i, data in enumerate(dataloader):
            inputs, _, sen_attrs, targets, indexes = data
            Y_set.append(targets)  # sen_attrs = -1 means no supervision for sensitive group
            S_set.append(sen_attrs)

            if self.cuda:
                inputs = inputs.cuda()
                groups = sen_attrs.cuda()
            if model is not None:
                outputs = model(inputs)
                Y_pred_set.append(torch.argmax(outputs, dim=1))
            total += inputs.shape[0]

        Y_set = torch.cat(Y_set)
        S_set = torch.cat(S_set)
        Y_pred_set = torch.cat(Y_pred_set) if len(Y_pred_set) != 0 else torch.zeros(0)
        return Y_pred_set.long(), Y_set.long().cuda(), S_set.long().cuda()

    # Vectorized version for DP & multi-class
    def get_error_and_violations_DP(self, y_pred, label, sen_attrs, num_groups, num_classes):
        acc = torch.mean(y_pred == label)
        total_num = len(y_pred)
        violations = torch.zeros((num_groups, num_classes))
        if self.cuda:
            violations = violations.cuda()
        for g in range(num_groups):
            for c in range(num_classes):
                pivot = len(torch.where(y_pred == c)[0]) / total_num
                group_idxs = torch.where(sen_attrs == g)[0]
                group_pred_idxs = torch.where(torch.logical_and(sen_attrs == g, y_pred == c))[0]
                violations[g, c] = len(group_pred_idxs)/len(group_idxs) - pivot
        return acc, violations

    # Vectorized version for EO & multi-class
    def get_error_and_violations_EO(self, y_pred, label, sen_attrs, num_groups, num_classes):
        acc = torch.mean((y_pred == label).float())
        violations = torch.zeros((num_groups, num_classes))
        if self.cuda:
            violations = violations.cuda()
        for g in range(num_groups):
            for c in range(num_classes):
                class_idxs = torch.where(label == c)[0]
                pred_class_idxs = torch.where(torch.logical_and(y_pred == c, label == c))[0]
                pivot = len(pred_class_idxs)/len(class_idxs)
                group_class_idxs = torch.where(torch.logical_and(sen_attrs == g, label == c))[0]
                group_pred_class_idxs = torch.where(torch.logical_and(torch.logical_and(sen_attrs == g, y_pred == c), label == c))[0]
                violations[g, c] = len(group_pred_class_idxs)/len(group_class_idxs) - pivot
        print('violations', violations)
        return acc, violations

    # update weight
    def debias_weights(self, label, sen_attrs, extended_multipliers, num_groups, num_classes):
        weights = torch.zeros(len(label))
        w_matrix = torch.sigmoid(extended_multipliers)  # g by c
        weights = w_matrix[sen_attrs, label]
        if self.slmode and self.version == 2:
            weights[sen_attrs == -1] = 0.5
        return weights

    def criterion(self, model, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)
