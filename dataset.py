import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

class RobotDataSet(Dataset):

    def __init__(self, filename="data_list.csv", data_root="./exp7500", image_list=None, transform=None, device="cpu"):
        df = pd.read_csv(filename)
        if image_list is not None:
            selection = df["subdir"].apply(lambda x: int(x[2:])).isin(image_list)
            df = df[selection]

        self.y = df[[f"y{i}" for i in range(6)]].to_numpy()
        self.y = torch.tensor(self.y)

        self.img_names = list(df["fname"])

        self.data_root = data_root

        self.transform = transform
        if self.transform is None:
            self.transform =  transforms.Compose([
                transforms.ConvertImageDtype(torch.float32),
                transforms.Resize((512,512)),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

        self.target_transform = transforms.ConvertImageDtype(torch.float32)
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.img_names[idx])
        image = read_image(img_path)
        y = self.y[idx]
        image = self.transform(image)
        y = self.target_transform(y)
        
        return image, y



if __name__ == "__main__":

    ds = RobotDataSet("data_list.csv", image_list=[1])
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    for image, y in dl:
        print(image)
        print(image.shape)
        print(y)
