# ocr_dataset.py
import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

# All valid characters in your readings
VOCAB = "0123456789."

# Mapping chars to integers
char_to_num = {c: i + 1 for i, c in enumerate(VOCAB)}  # +1 to reserve 0 for blank
num_to_char = {i + 1: c for i, c in enumerate(VOCAB)}
num_to_char[0] = ""  # for CTC blank

def encode_label(label):
    return [char_to_num[c] for c in label]

def decode_label(seq):
    return ''.join([num_to_char.get(i, '') for i in seq if i != 0])

class OCRDataset(Sequence):
    def __init__(self, root='dataset', img_width=200, img_height=50, batch_size=32, shuffle=True):
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.samples = []
        for subfolder in os.listdir(root):
            full_dir = os.path.join(root, subfolder)
            if not os.path.isdir(full_dir):
                continue
            for file in os.listdir(full_dir):
                if file.endswith(".png"):
                    path = os.path.join(full_dir, file)
                    label = os.path.splitext(file)[0].split("_")[1]
                    self.samples.append((path, label))

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)

    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = np.zeros((len(batch_samples), self.img_height, self.img_width, 1), dtype=np.float32)
        labels = []
        label_lengths = []
        input_lengths = np.full((len(batch_samples), 1), self.img_width // 4, dtype=np.int32)  # rough guess

        for i, (path, label_str) in enumerate(batch_samples):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img.astype(np.float32) / 255.0
            images[i, :, :, 0] = img

            label = encode_label(label_str)
            labels.append(label)
            label_lengths.append(len(label))

        max_len = max(label_lengths)
        padded_labels = np.zeros((len(labels), max_len), dtype=np.int32)
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = label

        return {
            'input': images,
            'label': padded_labels,
            'input_length': input_lengths,
            'label_length': np.array(label_lengths).reshape(-1, 1)
        }, np.zeros((len(images),))  # dummy loss
