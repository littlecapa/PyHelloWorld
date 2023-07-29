import torch
from torch import nn
from sklearn.metrics import r2_score
from hw_machine import HW_Machine
from hw_data_loader import HW_Data_Loader
import logging

class HW_Trainer():

    INIT_HASH = 0

    def __init__(self, filename = "hw_machine.h5", data_len = 1000, noise = 0.0, learning_rate = 1e-2):
        self.dataset = HW_Data_Loader(data_len, noise)
        self.filename = filename
        self.learning_rate = learning_rate
        self.init_components()
        self.hash = self.INIT_HASH

    def init_components(self):
      self.machine = HW_Machine()
      self.optimizer = torch.optim.Adam(self.machine.parameters(), self.learning_rate, weight_decay=1e-5)
      self.criterion = torch.nn.MSELoss(reduction='mean')

    def rest(self):
        torch.save(self.machine.state_dict(), self.filename)
        self.hash = self.machine.get_hash_value()
        print(f"Resting Hash: {self.hash}")
        self.machine.eval()
        self.do_baseline_testing()
        self.machine = None

    def resume(self, eval = True):
      if self.machine == None:
        self.init_components()
        self.machine.load_state_dict(torch.load(self.filename))
        new_hash = self.machine.get_hash_value()
        print(f"Resuming Hash: {new_hash}")
        if self.hash != self.INIT_HASH:
            if new_hash != self.hash:
                logging.error(f"Old hash: {self.hash}, New Hash: {new_hash}")
        self.hash = new_hash
        if eval:
            self.machine.eval()
        else:
            self.machine.train()

    def do_training(self, batch_size = 1, train_size = 0.95, num_epochs = 10, shuffle = False):
        self.resume(eval = False)
        train_size = int(train_size * len(self.dataset))
        val_size = len(self.dataset) - train_size        
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
        for epoch in range(num_epochs):
            running_loss = 0.0
            self.machine.train()
            for x, y in train_loader:
                outputs = self.machine(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                running_loss += loss.item()
            logging.debug(f'Training Results! Epoch:{epoch}, Running Loss:{running_loss}, Items: {(len(train_dataset))}, Avg: {running_loss/(len(train_dataset))}')
            self.do_validation(val_loader)
        self.do_baseline_testing()
        self.rest()
       
    def do_validation(self, val_loader):
        self.machine.eval()
        running_loss = 0.0
        with torch.no_grad(): 
            for x, y in val_loader:
                outputs = self.machine(x)
                loss = self.criterion(outputs, y)
                running_loss += loss
        logging.debug(f'Validation Results! Running Loss:{running_loss}, Items: {len(val_loader)}, Avg: {running_loss/len(val_loader)}')

    def do_baseline_testing(self):
        self.machine.eval()
        X_list, y_list = self.dataset.get_baseline_test_data()
        running_loss = 0.0
        with torch.no_grad():
            for i in range(len(y_list)):
                outputs = self.machine(X_list[i])
                loss = self.criterion(outputs, y_list[i])
                running_loss += loss
        logging.debug(f'Results Baseline Testing! Running Loss:{running_loss}, Items: {len(y_list)}, Avg: {running_loss/len(y_list)}')


