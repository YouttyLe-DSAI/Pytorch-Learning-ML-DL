import torch
from torch.utils.data import Dataset, DataLoader

data = torch.randn(100,10)
labels = torch.randint(0,2, (100,))

class CustomeDataset(Dataset):
    # dinh nghia data voi label
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    # tra ve so luong mau cua data
    def __len__(self):
        return len(self.data)
    # lay ra 1 mau trong dataset
    def __getitems__(self, indx):
        # Lấy dữ liệu và nhãn tại vị trí idx
        sample = self.data[indx]
        return sample, labels
## 2 Khoi tao dataset
my_dataset = CustomeDataset(data,labels)
## 3 Khoi tao dataloder
train_loader = DataLoader(my_dataset,batch_size= 16, shuffle=True, num_workers=2)
## 4 Su dung vong lap Training
print(f'tong so mau: {len(my_dataset)}')
print(f'Tong so batch: {len(train_loader)}')

num_epochs = 2
for epoch in range(num_epochs):
    print(f"bat dau epoch {epoch + 1}")
    for i, (batch_samples, batch_labes) in enumerate(train_loader):
        print(f"Batch{i+1}: Data shape {batch_samples.shape}")
    print("End")
