import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, path, txt_file, transform=None):
        super().__init__()
        self.path = path
        self.txt_file = txt_file
        try:
            with open(txt_file) as f:
                self.imgs = f.read().splitlines()
        except FileNotFoundError:
            self.imgs = os.listdir(path)
            with open(txt_file, 'w') as f:
                for img in self.imgs:
                    f.write(f'{img}\n')
        
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        item = self.imgs[index]
        img = Image.open(f'{self.path}/{item}')
        # Check if the image is grayscale and convert it to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        return img, item
    
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.profiler import profile, ProfilerActivity, schedule

    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor()
        ])

    # Create dataset loader and associated dataloader
    image_dataset = ImageDataset('500images', '500_images.txt', transform=
                                    transform)
    data_loader = DataLoader(image_dataset, batch_size=16, shuffle=True, num_workers=8)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
                 record_shapes=True, 
                 with_stack=True,
                 profile_memory=True) as prof:
        import time
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(device))

        # Get start time
        start_time = time.time()

        # Effectively load data and print it
        with open('batches.txt', 'w') as f:
            for batch_idx, sample in enumerate(data_loader):
                print(f'Batch {batch_idx} loaded')
                # print(i, sample)
                # Write to file
                for img in sample[1]:
                    f.write(f'{img}\n')
                prof.step()

        execution_time = time.time() - start_time

        print(f"The data loading tooks {execution_time} seconds.")

    print("Profiling data saved to ./logs for TensorBoard visualization")
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=10))


