import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import requests
import zipfile
import shutil


class TinyImageNet(Dataset):
    wordnet_to_idx = {}

    def __init__(self, root='data/tiny-imagenet-200', train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (bool): If True, creates dataset from training set, otherwise from validation set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root
        self.train = train
        self.transform = transform
      
        if self.train:
            self.image_folder = os.path.join(self.root_dir, 'train')
        else:
            self.image_folder = os.path.join(self.root_dir, 'val/images')

        if not os.path.exists(self.image_folder):
            self.download_file()
            self.unzip_file()

        self.data = []
        self.targets = []
        self.data_list = []

        if self.train:
            # Load training data
            for class_id, class_folder in enumerate(os.listdir(self.image_folder)):
                wordnet_id = class_folder  # Assuming class_folder name is the WordNet ID
                TinyImageNet.wordnet_to_idx[wordnet_id] = class_id                
                class_path = os.path.join(self.image_folder, class_folder, 'images')
                for image in os.listdir(class_path):
                    image_path = os.path.join(class_path, image)
                    self.data.append(image_path)
                    #image = Image.open(image_path).convert('RGB')
                    #self.data_list.append(image)
                    self.targets.append(class_id)
        else:
            # Load validation data
            with open(os.path.join(self.root_dir, 'val/val_annotations.txt')) as f:
                for line in f:
                    split_line = line.split()
                    image_name = split_line[0]
                    wordnet_id = split_line[1]
                    class_id = TinyImageNet.wordnet_to_idx[wordnet_id]
                    image_path = os.path.join(self.image_folder, image_name)
                    self.data.append(image_path)
                    #image = Image.open(image_path).convert('RGB')
                    #self.data_list.append(image)
                    self.targets.append(class_id)

        # Convert each PIL Image to a tensor and store in a list
        #tensor_images = [transforms.ToTensor()(img) for img in self.data_list]

        # Stack all tensors along a new dimension to create a 4D tensor
        #self.data = torch.stack(tensor_images)

        # Convert to NumPy array if necessary
        self.data = np.array(self.data)

    def download_file(self):
        tiny_imagenet_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)  # Create folder if it does not exist

        filename = tiny_imagenet_url.split('/')[-1].replace(" ", "_")  # Be sure to replace spaces with '_'
        file_path = os.path.join(self.root_dir, filename)

        r = requests.get(tiny_imagenet_url, stream=True)
        if r.ok:
            print("Saving to", os.path.abspath(file_path))
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 8):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
        else:  # HTTP status code 4XX/5XX
            print("Download failed: status code {}\n{}".format(r.status_code, r.text))

    def unzip_file(self):
        print('Unzipping...')
        with zipfile.ZipFile(os.path.join(self.root_dir, 'tiny-imagenet-200.zip'), 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)
        os.remove(os.path.join(self.root_dir, 'tiny-imagenet-200.zip'))
        print('Moving files around...')
        stuff_to_move = os.listdir(os.path.join(self.root_dir, 'tiny-imagenet-200'))
        for to_move in stuff_to_move:
            src = os.path.join(self.root_dir, 'tiny-imagenet-200', to_move)
            dst = os.path.join(self.root_dir, to_move)
            print('Moving file {} from {} to {}...'.format(to_move, src, dst))
            shutil.move(src, dst)
        print('Deleting old tiny-imagenet-200 folder...')
        os.rmdir(os.path.join(self.root_dir, 'tiny-imagenet-200'))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        #image = Image.fromarray(self.data[idx])
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

