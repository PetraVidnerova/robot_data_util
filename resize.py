import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image


def main(src, dest, size=(512,512), fname="data_list.csv"):

    if os.path.exists(dest):
        raise ValueError("Destination already exists.")
    
    os.mkdir(dest)
    
    df = pd.read_csv(fname)

    df["new_name"] = df["fname"].apply(
        lambda x: ".".join(x.split("/")[-1].split(".")[:-1])+".pt"
    )
    df["new_path"] = df["fname"].apply(
        lambda x: "/".join(x.split("/")[:-1]) 
    )

    def transform_img(img):
        transform =  transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((512,512))
        ])
        return transform(img)    

    def convert(x):
        img_path = os.path.join(src, x["fname"])
        pt_path = os.path.join(dest, x["new_path"], x["new_name"])
        pt_dir = os.path.join(dest, x["new_path"])
        if not os.path.exists(pt_dir):
            os.makedirs(pt_dir)
        print(f"Converting {img_path} to {pt_path}.")
        img = read_image(img_path)
        img = transform_img(img)
        torch.save(img,  pt_path)

    df.apply(convert, axis=1)
    
    
if __name__ == "__main__":
    src = "./exp7500"
    dest = "./exp7500_512x512"
    main(src, dest)
