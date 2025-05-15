import os
import random
import math
from PIL import Image
import pickle
import string


def save_model(nn, filename="3_model.pkl"):
    model_data = {
        "weights_input_hidden": nn.weights_input_hidden,
        "weights_hidden_output": nn.weights_hidden_output,
        "bias_hidden": nn.bias_hidden,
        "bias_output": nn.bias_output,
    }
    with open(filename, "wb") as f:
        pickle.dump(model_data, f)


def load_model(filename="3_model.pkl"):
    with open(filename, "rb") as f:
        model_data = pickle.load(f)
    nn = SimpleNN(len(model_data["weights_input_hidden"]),
                  len(model_data["bias_hidden"]),
                  len(model_data["bias_output"]))
    nn.weights_input_hidden = model_data["weights_input_hidden"]
    nn.weights_hidden_output = model_data["weights_hidden_output"]
    nn.bias_hidden = model_data["bias_hidden"]
    nn.bias_output = model_data["bias_output"]
    print(len(model_data["weights_input_hidden"]),
                  len(model_data["bias_hidden"]),
                  len(model_data["bias_output"]))
    return nn


# Görüntüyü yükleyip normalize etme
def process_image(image_path, target_size=(28, 28)):
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
        #data = [pixel / 255.0 for pixel in img.getdata()]
    return normalized_data


# Sinir ağı sınıfı
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Ağırlıkları rastgele başlat
        #self.weights_input_hidden = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        #self.weights_hidden_output = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        #Xavier Initialization (Glorot Initialization)
        limit = math.sqrt(1.0 / input_size)
        self.weights_input_hidden = [
            [random.uniform(-limit, limit) for _ in range(hidden_size)] for _ in range(input_size)
        ]
        self.weights_hidden_output = [
            [random.uniform(-limit, limit) for _ in range(output_size)] for _ in range(hidden_size)
        ]

        # Bias değerlerini rastgele başlat
        self.bias_hidden = [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-0.5, 0.5) for _ in range(output_size)]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        # Gizli katman hesaplaması
        self.hidden_layer = [
            self.sigmoid(sum(i * w for i, w in zip(inputs, self.weights_input_hidden[j])) + self.bias_hidden[j])
            for j in range(self.hidden_size)
        ]

        # Çıkış katmanı hesaplaması
        self.output_layer = [
            self.sigmoid(sum(h * w for h, w in zip(self.hidden_layer, self.weights_hidden_output[k])) + self.bias_output[k])
            for k in range(self.output_size)
        ]
        return self.output_layer

    def backward(self, inputs, targets, learning_rate=0.1):
        # Çıkış hatası
        output_errors = [targets[k] - self.output_layer[k] for k in range(self.output_size)]
        output_deltas = [error * self.sigmoid_derivative(self.output_layer[k]) for k, error in enumerate(output_errors)]

        # Gizli katman hatası
        hidden_errors = [
            sum(output_deltas[k] * self.weights_hidden_output[j][k] for k in range(self.output_size))
            for j in range(self.hidden_size)
        ]
        hidden_deltas = [error * self.sigmoid_derivative(self.hidden_layer[j]) for j, error in enumerate(hidden_errors)]

        # Ağırlıkları ve bias'ları güncelle
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_hidden_output[j][k] += learning_rate * output_deltas[k] * self.hidden_layer[j]
                self.bias_output[k] += learning_rate * output_deltas[k]

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += learning_rate * hidden_deltas[j] * inputs[i]
                self.bias_hidden[j] += learning_rate * hidden_deltas[j]

    def train(self, training_data, epochs=1000, learning_rate=0.1):

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

    """
    def predict(self, inputs):
        outputs = self.forward(inputs)
        return outputs.index(max(outputs))
    """
    def predict(nn, inputs, label_map):
        """
        nn: SimpleNN model
        inputs: Modelin tahmin yapacağı giriş verileri
        label_map: {index: class_label} formatında sınıf haritası
        """
        outputs = nn.forward(inputs)

        # Maksimum değeri bul
        max_value = max(outputs)

        # Maksimum değerlerin tüm indekslerini bul
        max_indices = [i for i, val in enumerate(outputs) if val == max_value]

        if len(max_indices) > 1:
            # Birden fazla maksimum değer varsa, rastgele birini seç
            print(f"Multiple maximum values found at indices {max_indices}")
            predicted_index = max_indices[0]  # İlkini seçebilir veya rastgele birini seçebilirsiniz
        else:
            predicted_index = max_indices[0]

        # Tahmin edilen sınıfı döndür
        predicted_label = label_map[predicted_index]
        return predicted_label


def load_data(data_dir, target_size=(28, 28)):
    """
    Dosyaları isimlerinin ilk harfine göre etiketler ve sıralı bir şekilde işler.

    Args:
        data_dir (str): Verinin bulunduğu klasör.
        target_size (tuple): Görüntülerin işlenirken yeniden boyutlandırılacağı hedef boyut.

    Returns:
        list: Her bir veri noktası (input, target) çiftinden oluşan liste.
    """
    data = []  # İşlenmiş veriler burada depolanacak
    labels = sorted(set(fname[0].upper() for fname in os.listdir(data_dir) if fname[0].isalpha()))
    # Etiketleme için ilk harflerin sıralı kümesi

    for filename in os.listdir(data_dir):
        if not filename[0].isalpha():
            continue  # Eğer dosya ismi alfabetik bir karakterle başlamıyorsa geç

        label = filename[0].upper()  # Etiket dosya isminin ilk harfi
        image_path = os.path.join(data_dir, filename)

        if not os.path.isfile(image_path):
            continue  # Eğer bu bir dosya değilse geç

        inputs = process_image(image_path, target_size)  # Görüntüyü işle
        targets = [1 if lbl == label else 0 for lbl in labels]  # One-hot encoding

        data.append((inputs, targets))  # Veri setine ekle

    return data


# Eğitim ve test
def main():
    label_map = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H"
    }
    #label_map = {i: letter for i, letter in enumerate(string.ascii_uppercase)}

    data_dir = "0_train_data"
    training_data = load_data(data_dir, target_size=(35, 35))

    # Ağ oluştur
    #input_size = 28 * 28
    input_size = 35 * 35
    hidden_size = 256
    output_size = 8
    nn = SimpleNN(input_size, hidden_size, output_size)

    # Eğitim
    nn.train(training_data, epochs=1, learning_rate=0.1)

    # Test
    test_dir = "9_ai_test_data"
    test_data = load_data(test_dir, target_size=(35, 35))  # Burada ayrı bir test seti kullanmak daha doğru olur
    correct = 0
    for inputs, targets in test_data:
        prediction = nn.predict(inputs, label_map)
        actual_label = label_map[targets.index(1)]
        print(f"Prediction: {prediction}, Actual: {actual_label}, Outputs: {nn.output_layer}, targets.index(1): {targets.index(1)}")
        if prediction == targets.index(1):
            correct += 1
    print(f"Accuracy: {correct / len(test_data):.2%}")

    save_model(nn)

if __name__ == "__main__":
    main()
