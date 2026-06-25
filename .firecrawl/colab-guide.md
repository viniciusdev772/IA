Are you interested in deep learning but don’t have access to an expensive GPU? You’re not alone! This comprehensive guide will show you exactly how to leverage Google Colab’s free T4 GPU to train your PyTorch and TensorFlow models without spending a dime.

Table of Contents

[Toggle](https://pylearnai.com/featured/deep-learning-google-colab-tutorial/#)

## Why Use Google Colab’s Free GPU?

- **Completely Free**: Access to NVIDIA T4 GPUs at zero cost
- **No Hardware Setup**: Skip the hassle of configuring drivers and CUDA
- **Pre-installed Libraries**: Many ML frameworks come pre-installed
- **Easy Sharing**: Collaborate with others through Google Drive integration
- **12-hour Runtime**: Sufficient for many training tasks

## Part 1: Setting Up Your Environment

### Checking Your GPU

First, let’s verify you have GPU access:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11 | `# Check if GPU is available`<br>`!nvidia``-``smi`<br>`# For PyTorch users`<br>`import``torch`<br>`print``(``"GPU Available:"``, torch.cuda.is_available())`<br>`print``(``"GPU Name:"``, torch.cuda.get_device_name(``0``)``if``torch.cuda.is_available()``else``"None"``)`<br>`# For TensorFlow users`<br>`import``tensorflow as tf`<br>`print``(``"TensorFlow GPU Available:"``, tf.config.list_physical_devices(``'GPU'``))` |

### Quick Installation Shortcuts

If the libraries aren’t already installed:

**For PyTorch:**

|     |     |
| --- | --- |
| 1 | `!pip install torch torchvision torchaudio``-``-``index``-``url https:``/``/``download.pytorch.org``/``whl``/``cu118` |

**For TensorFlow:**

|     |     |
| --- | --- |
| 1 | `!pip install tensorflow` |

**Restart Runtime Tip**: After installing packages, you often need to restart the runtime. Add this code to auto-restart:

|     |     |
| --- | --- |
| 1<br>2 | `import os`<br>`os.kill(os.getpid(), 9)` |

## Part 2: Dataset Upload Tricks

### Method 1: Google Drive Integration (Best for Large Datasets)

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5 | `from google.colab import drive`<br>`drive.mount('/content/drive')`<br>`# Access your dataset`<br>`dataset_path = '/content/drive/MyDrive/your_dataset_folder'` |

### Method 2: Direct Upload (Best for Small Files)

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7 | `from google.colab import files`<br>`uploaded = files.upload()`<br>`# Process uploaded file`<br>`import io`<br>`import pandas as pd`<br>`df = pd.read_csv(io.BytesIO(uploaded['your_file.csv']))` |

### Method 3: Download from URL (Best for Public Datasets)

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4 | `!wget https://raw.githubusercontent.com/username/repo/master/dataset.csv`<br>`# Or for compressed files`<br>`!wget https://example.com/dataset.zip`<br>`!unzip dataset.zip` |

### Method 4: Use Built-in Datasets (Fastest)

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7 | `# PyTorch`<br>`from torchvision.datasets import MNIST`<br>`train_dataset = MNIST(root='./data', train=True, download=True)`<br>`# TensorFlow`<br>`import tensorflow_datasets as tfds`<br>`mnist = tfds.load('mnist', split='train', as_supervised=True)` |

## Part 3: Training Models with PyTorch

### Basic PyTorch Training Loop with GPU Acceleration

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41<br>42<br>43<br>44<br>45<br>46<br>47<br>48<br>49<br>50<br>51<br>52<br>53<br>54<br>55<br>56<br>57<br>58<br>59<br>60<br>61<br>62<br>63<br>64<br>65<br>66<br>67<br>68<br>69<br>70<br>71<br>72<br>73<br>74<br>75<br>76<br>77<br>78<br>79<br>80 | `import torch`<br>`import torch.nn as nn`<br>`import torch.optim as optim`<br>`from torch.utils.data import DataLoader`<br>`from torchvision.datasets import CIFAR10`<br>`from torchvision.transforms import transforms`<br>`# Set device`<br>`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`<br>`print(f"Using device: {device}")`<br>`# Define transforms`<br>`transform = transforms.Compose([`<br>```transforms.ToTensor(),`<br>```transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`<br>`])`<br>`# Load dataset`<br>`train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)`<br>`train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)`<br>`# Define model`<br>`class SimpleConvNet(nn.Module):`<br>```def __init__(self):`<br>```super(SimpleConvNet, self).__init__()`<br>```self.conv1 = nn.Conv2d(3, 6, 5)`<br>```self.pool = nn.MaxPool2d(2, 2)`<br>```self.conv2 = nn.Conv2d(6, 16, 5)`<br>```self.fc1 = nn.Linear(16 * 5 * 5, 120)`<br>```self.fc2 = nn.Linear(120, 84)`<br>```self.fc3 = nn.Linear(84, 10)`<br>```def forward(self, x):`<br>```x = self.pool(torch.relu(self.conv1(x)))`<br>```x = self.pool(torch.relu(self.conv2(x)))`<br>```x = x.view(-1, 16 * 5 * 5)`<br>```x = torch.relu(self.fc1(x))`<br>```x = torch.relu(self.fc2(x))`<br>```x = self.fc3(x)`<br>```return x`<br>`model = SimpleConvNet().to(device)`<br>`criterion = nn.CrossEntropyLoss()`<br>`optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)`<br>`# Training loop`<br>`num_epochs = 5`<br>`save_path = './model_checkpoints'`<br>`os.makedirs(save_path, exist_ok=True)`<br>`for epoch in range(num_epochs):`<br>```running_loss = 0.0`<br>```for i, (inputs, labels) in enumerate(train_loader):`<br>```# Move data to device`<br>```inputs, labels = inputs.to(device), labels.to(device)`<br>``<br>```# Zero the parameter gradients`<br>```optimizer.zero_grad()`<br>``<br>```# Forward + backward + optimize`<br>```outputs = model(inputs)`<br>```loss = criterion(outputs, labels)`<br>```loss.backward()`<br>```optimizer.step()`<br>``<br>```# Print statistics`<br>```running_loss += loss.item()`<br>```if i % 200 == 199:`<br>```print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')`<br>```running_loss = 0.0`<br>``<br>```# Save checkpoint after each epoch`<br>```torch.save({`<br>```'epoch': epoch,`<br>```'model_state_dict': model.state_dict(),`<br>```'optimizer_state_dict': optimizer.state_dict(),`<br>```'loss': running_loss,`<br>```}, f'{save_path}/model_epoch_{epoch}.pth')`<br>``<br>`print('Finished Training')` |

### Loading Checkpoints to Resume Training

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7 | `checkpoint = torch.load(f'{save_path}/model_epoch_2.pth')`<br>`model.load_state_dict(checkpoint['model_state_dict'])`<br>`optimizer.load_state_dict(checkpoint['optimizer_state_dict'])`<br>`epoch = checkpoint['epoch']`<br>`loss = checkpoint['loss']`<br>`# Continue training from this point` |

## Part 4: Training Models with TensorFlow

### Basic TensorFlow/Keras Training

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41<br>42<br>43<br>44<br>45<br>46<br>47<br>48<br>49<br>50<br>51<br>52 | `import tensorflow as tf`<br>`from tensorflow.keras import datasets, layers, models`<br>`import matplotlib.pyplot as plt`<br>`import os`<br>`# Check for GPU`<br>`physical_devices = tf.config.list_physical_devices('GPU')`<br>`print("Num GPUs Available: ", len(physical_devices))`<br>`tf.config.experimental.set_memory_growth(physical_devices[0], True)`<br>`# Load and preprocess the CIFAR10 dataset`<br>`(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()`<br>`# Normalize pixel values to be between 0 and 1`<br>`train_images, test_images = train_images / 255.0, test_images / 255.0`<br>`# Create the model`<br>`model = models.Sequential([`<br>```layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),`<br>```layers.MaxPooling2D((2, 2)),`<br>```layers.Conv2D(64, (3, 3), activation='relu'),`<br>```layers.MaxPooling2D((2, 2)),`<br>```layers.Conv2D(64, (3, 3), activation='relu'),`<br>```layers.Flatten(),`<br>```layers.Dense(64, activation='relu'),`<br>```layers.Dense(10)`<br>`])`<br>`# Compile the model`<br>`model.compile(optimizer='adam',`<br>```loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),`<br>```metrics=['accuracy'])`<br>`# Create checkpoint callback`<br>`checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"`<br>`checkpoint_dir = os.path.dirname(checkpoint_path)`<br>`os.makedirs(checkpoint_dir, exist_ok=True)`<br>`# Create a callback that saves the model's weights every epoch`<br>`cp_callback = tf.keras.callbacks.ModelCheckpoint(`<br>```filepath=checkpoint_path,`<br>```verbose=1,`<br>```save_weights_only=True,`<br>```save_freq='epoch')`<br>`# Train the model with checkpoint saving`<br>`history = model.fit(train_images, train_labels, epochs=10,`<br>```validation_data=(test_images, test_labels),`<br>```callbacks=[cp_callback])`<br>`# Save the entire model`<br>`model.save('saved_model/my_model')` |

### Resuming Training from Checkpoint

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7 | `# Load the model weights`<br>`model.load_weights('training_checkpoints/cp-0005.ckpt')`<br>`# Continue training`<br>`history = model.fit(train_images, train_labels, epochs=5,`<br>```validation_data=(test_images, test_labels),`<br>```initial_epoch=5) # Start from epoch 5` |

## Part 5: Common Errors and Their Fixes

### Error 1: CUDA Out of Memory

**Error Message:**

|     |     |
| --- | --- |
| 1 | `RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.90 GiB total capacity; 14.73 GiB already allocated; 964.88 MiB free; 14.83 GiB reserved in total by PyTorch)` |

**Fix:**

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14 | `# Reduce batch size`<br>`train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Smaller batch size`<br>`# Or use gradient accumulation`<br>`accumulation_steps = 4  # Accumulate gradients over 4 batches`<br>`optimizer.zero_grad()`<br>`for i, (inputs, labels) in enumerate(train_loader):`<br>```outputs = model(inputs.to(device))`<br>```loss = criterion(outputs, labels.to(device))`<br>```loss = loss / accumulation_steps  # Normalize loss`<br>```loss.backward()`<br>```if (i + 1) % accumulation_steps == 0:  # Update weights every few batches`<br>```optimizer.step()`<br>```optimizer.zero_grad()` |

### Error 2: Runtime Disconnection

**Problem:** Colab disconnects after periods of inactivity or long training runs.

**Fix:**

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6 | `// Run this in the browser console (F12) to keep your session alive`<br>`function ClickConnect(){`<br>```console.log("Working");`<br>```document.querySelector("colab-toolbar-button#connect").click()`<br>`}`<br>`setInterval(ClickConnect, 60000)` |

### Error 3: Package Version Conflicts

**Error Message:**

|     |     |
| --- | --- |
| 1 | `ImportError: cannot import name 'softmax' from 'tensorflow.python.ops.nn_ops'` |

**Fix:**

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5 | `# Uninstall problematic versions`<br>`!pip uninstall -y tensorflow tensorflow-gpu`<br>`# Install specific compatible versions`<br>`!pip install tensorflow==2.12.0` |

### Error 4: Dataset Loading Issues

**Error Message:**

|     |     |
| --- | --- |
| 1 | `FileNotFoundError: [Errno 2] No such file or directory: '/content/data/train'` |

**Fix:** Check your paths and use absolute paths when possible:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6 | `import os`<br>`print("Current working directory:", os.getcwd())`<br>`!ls -la  # List all files in current directory`<br>`# Create directory if it doesn't exist`<br>`os.makedirs('/content/data/train', exist_ok=True)` |

## Part 6: Advanced Performance Tips

### 1\. Use Mixed Precision Training for Speed

**PyTorch Implementation:**

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19 | `from torch.cuda.amp import autocast, GradScaler`<br>`scaler = GradScaler()`<br>`for epoch in range(num_epochs):`<br>```for inputs, labels in train_loader:`<br>```inputs, labels = inputs.to(device), labels.to(device)`<br>``<br>```optimizer.zero_grad()`<br>``<br>```# Use autocast for mixed precision`<br>```with autocast():`<br>```outputs = model(inputs)`<br>```loss = criterion(outputs, labels)`<br>``<br>```# Scale gradients and optimize`<br>```scaler.scale(loss).backward()`<br>```scaler.step(optimizer)`<br>```scaler.update()` |

**TensorFlow Implementation:**

|     |     |
| --- | --- |
| 1<br>2<br>3 | `# Enable mixed precision`<br>`from tensorflow.keras import mixed_precision`<br>`mixed_precision.set_global_policy('mixed_float16')` |

### 2\. Optimize Data Loading

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11 | `# For PyTorch: Use num_workers and pin_memory for faster data loading`<br>`train_loader = DataLoader(`<br>```train_dataset,`<br>```batch_size=64,`<br>```shuffle=True,`<br>```num_workers=2,  # Parallelize data loading`<br>```pin_memory=True  # Speed up CPU to GPU transfers`<br>`)`<br>`# For TensorFlow: Use prefetching and caching`<br>`train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)` |

### 3\. Monitor and Visualize Training (TensorBoard Integration)

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16 | `# PyTorch with TensorBoard`<br>`from torch.utils.tensorboard import SummaryWriter`<br>`writer = SummaryWriter('runs/experiment_1')`<br>`# Log metrics during training`<br>`for epoch in range(num_epochs):`<br>```running_loss = 0.0`<br>```for i, (inputs, labels) in enumerate(train_loader):`<br>```# Training code here...`<br>```running_loss += loss.item()`<br>``<br>```# Log average loss for the epoch`<br>```writer.add_scalar('training loss', running_loss / len(train_loader), epoch)`<br>``<br>```# Add model graph`<br>```writer.add_graph(model, inputs.to(device))` |

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10 | `# TensorFlow with TensorBoard`<br>`tensorboard_callback = tf.keras.callbacks.TensorBoard(`<br>```log_dir="./logs",`<br>```histogram_freq=1,`<br>```write_graph=True`<br>`)`<br>`model.fit(train_images, train_labels, epochs=10,`<br>```validation_data=(test_images, test_labels),`<br>```callbacks=[tensorboard_callback])` |

### 4\. Prevent Disconnections

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8 | `# Save frequently (every 100 batches)`<br>`if i % 100 == 0:`<br>```torch.save({`<br>```'batch': i,`<br>```'epoch': epoch,`<br>```'model_state_dict': model.state_dict(),`<br>```'optimizer_state_dict': optimizer.state_dict(),`<br>```}, f'{save_path}/checkpoint_e{epoch}_b{i}.pth')` |

Another effective method is to use a “keepalive” script:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13 | `from IPython.display import display, Javascript`<br>`import time`<br>`def keep_alive(delay_sec=60):`<br>```"""Execute Javascript to keep the Colab runtime alive."""`<br>```display(Javascript('''`<br>```function click(){`<br>```document.querySelector('#top-toolbar > colab-connect-button').click();`<br>```}`<br>```setInterval(click, ''' + str(delay_sec*1000) + ''');`<br>```'''))`<br>``<br>`keep_alive()` |

## Part 7: Working with Custom Datasets

### Creating a Custom Dataset in PyTorch

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29 | `from torch.utils.data import Dataset`<br>`import os`<br>`from PIL import Image`<br>`class CustomImageDataset(Dataset):`<br>```def __init__(self, annotations_file, img_dir, transform=None):`<br>```self.img_labels = pd.read_csv(annotations_file)`<br>```self.img_dir = img_dir`<br>```self.transform = transform`<br>```def __len__(self):`<br>```return len(self.img_labels)`<br>```def __getitem__(self, idx):`<br>```img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])`<br>```image = Image.open(img_path).convert('RGB')`<br>```label = self.img_labels.iloc[idx, 1]`<br>``<br>```if self.transform:`<br>```image = self.transform(image)`<br>``<br>```return image, label`<br>`# Using the custom dataset`<br>`train_dataset = CustomImageDataset(`<br>```annotations_file='/content/drive/MyDrive/annotations.csv',`<br>```img_dir='/content/drive/MyDrive/images',`<br>```transform=transform`<br>`)` |

### Creating a Data Generator in TensorFlow

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28 | `from tensorflow.keras.preprocessing.image import ImageDataGenerator`<br>`# Create a data generator with augmentation`<br>`train_datagen = ImageDataGenerator(`<br>```rescale=1./255,`<br>```rotation_range=20,`<br>```width_shift_range=0.2,`<br>```height_shift_range=0.2,`<br>```shear_range=0.2,`<br>```zoom_range=0.2,`<br>```horizontal_flip=True,`<br>```fill_mode='nearest'`<br>`)`<br>`# Load images from directory`<br>`train_generator = train_datagen.flow_from_directory(`<br>```'/content/drive/MyDrive/dataset/train',`<br>```target_size=(150, 150),`<br>```batch_size=32,`<br>```class_mode='binary'`<br>`)`<br>`# Train using the generator`<br>`model.fit(`<br>```train_generator,`<br>```steps_per_epoch=train_generator.samples // 32,`<br>```epochs=10`<br>`)` |

## Troubleshooting Guide: When Things Go Wrong

### 1\. Colab Disconnects Frequently

**Causes:**

- Inactivity
- Long-running cells
- Browser tab closed
- Limited resources

**Solutions:**

- Use the keep-alive script shown above
- Break training into smaller epochs
- Save checkpoints more frequently
- Run important cells at the beginning of your session

### 2\. Training Is Too Slow

**Causes:**

- Large dataset
- Complex model
- Inefficient data loading
- CPU operations

**Solutions:**

- Use a smaller subset of data for prototyping
- Apply data caching and prefetching
- Ensure operations are running on GPU:

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7 | `# Check where your tensors are`<br>`print(f"Input tensor device: {inputs.device}")`<br>`# Profile your code to find bottlenecks`<br>`with torch.autograd.profiler.profile(use_cuda=True) as prof:`<br>```model(inputs)`<br>`print(prof.key_averages().table(sort_by="cuda_time_total"))` |

### 3\. Model Accuracy Is Poor

**Potential issues:**

- Learning rate too high/low
- Overfitting
- Underfitting
- Data quality issues

**Solutions:**

- Implement learning rate scheduling
- Add regularization (dropout, weight decay)
- Increase model complexity
- Check for data imbalance

|     |     |
| --- | --- |
| 1<br>2<br>3<br>4<br>5<br>6<br>7 | `# Learning rate scheduling in PyTorch`<br>`scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(`<br>```optimizer, mode='min', factor=0.1, patience=3, verbose=True`<br>`)`<br>`# After validation in your training loop`<br>`scheduler.step(val_loss)` |

## Conclusion: Taking Your Skills Further

You’ve now learned how to effectively use Google Colab’s free T4 GPU resources to train deep learning models. This knowledge allows you to:

1. Experiment with complex models without expensive hardware
2. Prototype ideas quickly and efficiently
3. Share your work with collaborators instantly
4. Gradually scale up to more powerful resources when needed

For your next steps, consider:

- Exploring pre-trained models for transfer learning
- Experimenting with different architectures
- Participating in Kaggle competitions using these techniques
- Collaborating with others by sharing your notebooks

Remember that Google Colab’s free tier has limitations, including:

- 12-hour maximum runtime
- Potential disconnections
- Limited storage
- No guaranteed GPU availability

But with the techniques in this guide, you can maximize your productivity within these constraints!

Happy training, and don’t forget to save your work frequently!

* * *