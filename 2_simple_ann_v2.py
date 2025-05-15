"""
Eğer sık sık ağırlık ve bias güncellemeleri yapılıyorsa, delta değerlerini
    (yani önceki gradyanlar) saklayarak momentum optimizasyonu kullanılabilir.
Mini-batch Gradient Descent
from functools import lru_cache @lru_cache(maxsize=128)

Memoization vs Dynamic Programming
    Memoization genellikle dinamik programlamanın bir parçası olarak görülür:
    Memoization: "Yukarıdan aşağıya" bir yaklaşım kullanır. Rekürsif fonksiyonların
        sonuçlarını saklar.
    Dynamic Programming: "Aşağıdan yukarıya" bir yaklaşım kullanır. Tüm alt
        problemleri iteratif bir şekilde çözer.
"""

from PIL import Image
import os
import math
import random
#import glob
#import pickle
#import string
import pprint


def save_model(nn, filename="2_model.pkl"):
    pass


def load_model(filename="2_model.pkl"):
    pass


def process_image(image_path, target_size=(50, 50)):
    with Image.open(image_path) as img:
        img = img.convert("L")  # Grayscale
        img = img.resize(target_size)
        data = img.getdata()
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance)
        # Standart sapmanın sıfır olup olmadığını kontrol et
        if std_dev == 0:
            # Tüm pikseller aynı değerde, normalizasyon gereksiz
            normalized_data = [0 for _ in data]
        else:
            normalized_data = [(x - mean) / std_dev for x in data]
        #data = [pixel / 255.0 for pixel in img.getdata()] # Kontrastı azaltma için tercih edilmeyebilir.
        #print(min(normalized_data), max(normalized_data))
    return normalized_data


def load_data(data_dir, target_size=(50, 50)):
    data = []

    file_list = sorted(os.listdir(data_dir))

    labels = sorted(set(fname[0] for fname in file_list if fname[0].isalpha()))

    for filename in file_list:
        if not filename[0].isalpha():
            continue # Eğer dosya ismi alfabetik bir karakterle başlamıyorsa geç

        label = filename[0]
        image_path = os.path.join(data_dir, filename)

        if not os.path.isfile(image_path):
            continue  # Eğer bu bir dosya değilse geç

        inputs = process_image(image_path, target_size) # Görüntüyü işle
        targets = [1 if lbl == label else 0 for lbl in labels] # One-hot encoding

        data.append((inputs, targets))  # Veri setine ekle

    return data


class SimpleNN:
    def __init__(self, layer_sizes):
        """
        Katman büyüklüklerini kullanarak sinir ağı oluşturur.
        Args:
            layer_sizes (list[int]): Her bir katmanın düğüm sayısını içerir.
        """
        self.layer_sizes = layer_sizes
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()#[0] * (len(layer_sizes) - 1)
        self.early_stopping = False

    def initialize_weights(self):
        """
        Xavier Initialization kullanarak ağırlık matrislerini oluşturur.
        Returns:
            list: Her katman için ağırlık matrisleri.
        """
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            limit = math.sqrt(2.0 / (input_size + output_size))  # Xavier limit for ReLu
            weights.append(
                [
                    [random.uniform(-limit, limit) for _ in range(output_size)]
                    for _ in range(input_size)
                ]
            )
        print("weights")
        pprint.pprint(weights, indent=2, width=80)
        return weights

    def initialize_biases(self):
        """
        Bias değerlerini Xavier Initialization'a uygun şekilde başlatır.
        Returns:
            list: Her katman için bias değerleri.
        """
        biases = []
        for layer_size in self.layer_sizes[1:]:  # Giriş katmanını atla
            #biases.append([0.0 for _ in range(layer_size)])
            limit = math.sqrt(1.0 / layer_size)
            #print("\t", limit)
            biases.append(
                [random.uniform(-limit, limit) for _ in range(layer_size)]
            )
        print("biases")
        pprint.pprint(biases)
        return biases

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1.0 - x)

    def forward(self, inputs):
        print("forward")
        print(inputs)
        print([inputs])
        for weight_matrix, bias_vector in zip(self.weights, self.biases):
            print("weight_matrix")
            pprint.pprint(weight_matrix)
            print("bias_vector")
            pprint.pprint(bias_vector)
            print("->")
            for i, j in zip(weight_matrix, bias_vector):
                print("\t", i, " - ", j)
            z = [
                sum([5, 6])
            ]

        # Debugging mesajları
        print("debug")
        for weight_row, bias in zip(weight_matrix, bias_vector):
            print(f"Weight row: {weight_row}, Bias: {bias}")

        print(f"z values (weighted sum + bias): {z}")


    def backward(self, inputs, targets, learning_rate=0.1):
        pass

    def train(self, training_data, epochs=1000, initial_learning_rate=0.1):
        for epoch in range(epochs):
            total_error = 0
            for inputs, targets in training_data:
                self.forward(inputs)
                self.backward(inputs, targets, learning_rate)
                total_error += sum((t - o) ** 2 for t, o in zip(targets, self.output_layer))

            print(f"Epoch {epoch}, Error: {total_error:.4f}")
            if epoch % 10 == 0:
                learning_rate *= 0.1
                print(f"New learning_rate: {learning_rate}")

    def predict(nn, inputs, label_map):
        pass


def main():
    training_data = load_data("0_train_data")

    input_size = 50 * 50
    hidden_sizes = [256, 128]  # Dinamik katman boyutları
    output_size = 8
    layer_sizes = [input_size] + hidden_sizes + [output_size]

    layer_sizes = [2] + [3] + [2]
    training_data =  [
        [[10, 20], "A"],
        [[1, 2], "A"],
        [[5, 6], "B"]
    ]
    print(layer_sizes)
    pprint.pprint(training_data)
    nn = SimpleNN(layer_sizes)

    # Eğitim
    nn.early_stopping = True
    nn.train(training_data, epochs=10, initial_learning_rate =0.1)

    # Test
    test_dir = "9_ai_test_data"
    test_data = load_data(test_dir, target_size=(50, 50))
    correct = 0
    for inputs, targets in test_data:
        prediction = nn.predict(inputs)
        if prediction == targets.index(1):
            correct += 1
        else:
            print("f\t{input_name} -> {output_name}")
    print(f"Accuracy: {correct / len(test_data):.2%}")

    save_model(nn)
    nn = load_model(nn)

if __name__ == "__main__":
    main()


















"""
def get_preprocess_train_data(train_data_dir="0_train_data"):
    train_data = glob.glob(os.path.join(train_data_dir, "*"))
    train_data.sort() # Muhtemelen zaten sıralı, windows için
    #print(train_data == glob.glob(os.path.join(train_data_dir, "*")))
    return train_data


def get_image_data_1d(full_image_name):
    img = Image.open(full_image_name)
    pixels = list(img.getdata())

    for pixel in pixels:
        print(pixel)


def get_image_data_2d(full_image_name):
    img = Image.open(full_image_name)
    pixels = list(img.getdata())

    width, height = img.size
    pixels_2d = [pixels[i * width:(i + 1) * width] for i in range(height)]

    for row in pixels_2d:
        print(row)


class Ann():
    _id = 0

    def init(input_layer_nodes, output_layer_nodes):
        self.id = Ann._id
        Ann._id += 1

        self.input_layer = _create_layer(input_layer_nodes)
        self.output_layer = _create_layeroutput_layer_nodes


    def _create_layer(nodes_number):
        neuron = {
            id: -1,
            train_state: True,
            doorstep: None,
            weights: {}
        }
        return [ for nodes in range(nodes_number)]


    def _create_neuron(self, layer_name, id):
        pass


    def create_noral():
        self.norals = [{id:i, aktiflesme_dozu:None, baglantılar:{id:None, weight:None}} for i in self.nodes]


ann = Ann(2, 2)
"""
