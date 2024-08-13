# Voting Power Ensemble

## Getting the CINIC10 data for these experiments

To download the CINIC10 data follow the instructions given in this repository: https://github.com/BayesWatch/cinic-10/tree/master

After downloading, we need to divide the data into CIFAR10 and ImageNet. To do this run this python code given below:

```
import os
import shutil

# Define the source directory and the target directories
for i in ['train', 'valid', 'test']:
    source_dir = '/source/directory/to/cinic10/data'+i  # replace with the path to your cinic10 directory
    dest_dir = '/destination/directory/to/cinic10/data/'+i+str(2)
    cifar_dir = os.path.join(dest_dir, 'cifar')
    imagenet_dir = os.path.join(dest_dir, 'imagenet')

    # List of class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Create the cifar and imagenet directories with class subdirectories
    for cls in classes:
        os.makedirs(os.path.join(cifar_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(imagenet_dir, cls), exist_ok=True)

    # Move images to the appropriate directories
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        for img_name in os.listdir(class_dir):
            if img_name.startswith('c'):
                shutil.move(os.path.join(class_dir, img_name), os.path.join(cifar_dir, cls, img_name))
            elif img_name.startswith('n'):
                shutil.move(os.path.join(class_dir, img_name), os.path.join(imagenet_dir, cls, img_name))

    print("Images have been reorganized successfully.")
```
The data to be used in these experiments can be found in the destination directory

Link to Phising data: https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset
Link to DMOZ URL Classification data: https://www.kaggle.com/datasets/shawon10/url-classification-dataset-dmoz
