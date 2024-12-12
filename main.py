import os
import pickle
import tarfile
from collections import defaultdict
from math import sqrt
from urllib.request import urlretrieve
from functools import reduce
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

NUM_BINS = 8

def download_and_extract_cifar10(root='./data'):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    filepath = os.path.join(root, filename)
    extract_path = os.path.join(root, 'cifar-10-batches-py')
    if not os.path.exists(filepath):
        os.makedirs(root, exist_ok=True)
        urlretrieve(url, filepath)
    if not os.path.exists(extract_path):
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=root)
    return extract_path

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f, encoding='bytes')

def load_cifar10(samples_per_class=100, root='./data', train=True):
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    data_path = download_and_extract_cifar10(root)
    class_images = defaultdict(list)
    class_counts = {name: 0 for name in class_names}

    batch_files = [f'data_batch_{i}' for i in range(1, 6)] if train else ['test_batch']

    for batch_file in batch_files:
        batch_path = os.path.join(data_path, batch_file)
        batch = load_batch(batch_path)

        images = batch[b'data']
        labels = batch[b'labels']

        for img, label in zip(images, labels):
            class_name = class_names[label]

            if class_counts[class_name] < samples_per_class:
                img_reshaped = img.reshape(3, 32, 32).transpose(1, 2, 0)
                class_images[class_name].append(img_reshaped)
                class_counts[class_name] += 1

            if all(count >= samples_per_class for count in class_counts.values()):
                break

        if all(count >= samples_per_class for count in class_counts.values()):
            break

    return dict(class_images)

def update_histogram(histograms, bins):
    bin_size = 256 // NUM_BINS
    r_hist, g_hist, b_hist = histograms
    r_bin, g_bin, b_bin = bins

    r_hist[r_bin // bin_size] += 1
    g_hist[g_bin // bin_size] += 1
    b_hist[b_bin // bin_size] += 1

    return r_hist, g_hist, b_hist

def calculate_histograms(image_input):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = Image.fromarray(image_input).convert('RGB')
    size = image.size[0] * image.size[1]

    r_hist = np.zeros(NUM_BINS, dtype=np.float32)
    g_hist = np.zeros(NUM_BINS, dtype=np.float32)
    b_hist = np.zeros(NUM_BINS, dtype=np.float32)

    r_hist, g_hist, b_hist = reduce(update_histogram, list(image.getdata()), (r_hist, g_hist, b_hist))

    r_hist = r_hist / size
    g_hist = g_hist / size
    b_hist = b_hist / size

    return np.stack([r_hist, g_hist, b_hist], axis=0)

def update_average_histogram(current_avg, image_histograms):
    avg_hist, count = current_avg

    avg_hist[0] = list(map(lambda x, y: x + y, avg_hist[0], image_histograms[0]))
    avg_hist[1] = list(map(lambda x, y: x + y, avg_hist[1], image_histograms[1]))
    avg_hist[2] = list(map(lambda x, y: x + y, avg_hist[2], image_histograms[2]))

    return avg_hist, count + 1

def calculate_class_average(class_name, images):
    avg_hist = np.zeros((3, NUM_BINS), dtype=np.float32)

    avg_hist, count = reduce(update_average_histogram, images, (avg_hist, 0))

    avg_hist[0] = list(map(lambda x: x / count, avg_hist[0]))
    avg_hist[1] = list(map(lambda x: x / count, avg_hist[1]))
    avg_hist[2] = list(map(lambda x: x / count, avg_hist[2]))

    return class_name, avg_hist

def cosine_similarity(hist1, hist2):
    flat_hist1 = hist1.flatten()
    flat_hist2 = hist2.flatten()

    dot_product = reduce(lambda acc, pair: acc + pair[0] * pair[1], zip(flat_hist1, flat_hist2), 0)
    norm1 = sqrt(reduce(lambda acc, x: acc + x ** 2, flat_hist1))
    norm2 = sqrt(reduce(lambda acc, x: acc + x ** 2, flat_hist2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def classify_image(image_histogram, class_averages):
    best_class = None
    best_similarity = 0.0

    compute_similarity = lambda acc, item: (
        item[0], cosine_similarity(image_histogram, item[1])
    ) if cosine_similarity(image_histogram, item[1]) > acc[1] else acc

    best_class, best_similarity = reduce(compute_similarity, class_averages, (None, 0.0))

    return best_class, best_similarity

def plot_histograms(histograms, bins_num=8):
    colors = ['red', 'green', 'blue']
    labels = ['Red', 'Green', 'Blue']

    for i in range(3):
        plt.plot(histograms[i], color=colors[i], label=f'{labels[i]} Component')

    plt.title('Normalized Color Histograms')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    image_path = input("Enter the image path: ")
    image_histogram = calculate_histograms(image_path)

    images_with_classes = load_cifar10(100)
    class_averages = [
        calculate_class_average(class_name, list(map(calculate_histograms, images)))
        for class_name, images in images_with_classes.items()
    ]

    predicted_class, similarity = classify_image(image_histogram, class_averages)

    print(f"Predicted Class: {predicted_class} with similarity: {similarity:.4f}")
    plot_histograms(image_histogram, NUM_BINS)
