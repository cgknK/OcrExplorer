"""
P 1: Değerlendirme (Model Performansının Analizi) yapılmalı.
P 2: Özellik mühendisliği (feature engineering nedir?
    Ayrıca "Frekans Tabanlı Özellikler: Fourier dönüşümü ile zaman serisindeki
        periyodik yapıları analiz " bak.
P 3: Fine-Tuning uygula.
P 4: Görüntü sınıflandırma (image classification), nesne tespiti (object detection)
    ve segmentasyon (segmentation), bilgisayarla görü (computer vision) alanı
    tekniklerine bak.
P 5: CNN'lerde yzamsal Yapı (Spatial Structure).
P 6: CNN ile RNN hafıza özelliğini birleştirmeye çalış (saçma ise düşün).
P 7: Eğer çok yavaş ise, binary image geç.
P 8: Backpropagation yerine yarısı ile optimize ederek ilermelemeyi dene.
P 9: f(x) = ((1 / (0.5 + e^-x)) - 1) * a   - > dene. a sınırlar.
    ve bunu Tanh ile kıyasla.
    Hatta:
        swish'den esinlenerek -> f(x) = (x/2^c) * (1/(1+e^(-x))) ve c bir sabit.
    Birde hızlandırma için Hard Sigmoid.
    Ayrıca:
         Sigmoid Dönüşümünün Dezavantajları;
            Gradyan Doygunluğu Sorununu Çözmez.
            Zincirleme Türev Etkisi (Vanishing Gradient) sorunu devam eder.
            Bunları incale.
P A: RGBG düzeninde (bayer filtresi) giriş ver, gerçek göz gibi yada kamera.
P B: functools.cache, functools.lru_cache eklenebilir.
P C: Uzamsal bilgi ekle, input katmanına veya input katmanında.
P D: Bias nasıl ayarlanacak? (Hatta öğrenerek bias'ı nasıl ayarlayacak.)
P E: Sigmoid girdi ve çıktı tablosunu nasıl bir interpolasyon ile doğru sayılacak
    bir doğrulukla hesaplarız. Örneğin a, b, c için sonuçlar a2, b2, c2. (a+b)/2 için
    sonucu nasıl hesaplarsın tablodaki değerler ile.
P F: reinforcement learning and GAN.
P G: Overfitting ve Underfitting ölçümnü.
P H: Normalizasyon;
        Sabit aralığa ölçekleme
        Ortalama çıkarma ve standart sapma il ölçekleme
        Min-Max normalizasyonu
        Logaritmik veya köklü normalizasyon
        Z-score normalizasyonu
        Gama düzeltmesi
"""
"""
Resmi yeniden boyutlandırma: Görüntüleri sabit bir boyuta yeniden boyutlandırabilirsiniz.
    Bu, görüntüleri aynı boyuta getirecektir, ancak orijinal görüntülerin bazı özelliklerini
    kaybedebilir.
Spatial Pyramid Pooling (SPP): Bu yaklaşım, görüntüleri farklı boyutlarda bölümlere ayırır
    ve her bölüm için özellikler çıkarır. Bu, görüntülerin farklı boyutlarda ve yönlerdeki
    özelliklerini yakalamaya yardımcı olabilir.
Convolutional Neural Networks (CNN) ile: CNN'ler, görüntülerin farklı boyutlarda ve yönlerdeki
    özelliklerini yakalamak için tasarlanmıştır. CNN'leri kullanarak, görüntülerin farklı
    boyutlarda ve yönlerdeki özelliklerini çıkarabilirsiniz.
Fully Convolutional Networks (FCN): FCN'ler, görüntülerin farklı boyutlarda ve yönlerdeki
    özelliklerini yakalamak için tasarlanmıştır. FCN'leri kullanarak, görüntülerin farklı
    boyutlarda ve yönlerdeki özelliklerini çıkarabilirsiniz.
Spatial Transformer Networks (STN): STN'ler, görüntülerin farklı boyutlarda ve yönlerdeki
    özelliklerini yakalamak için tasarlanmıştır. STN'leri kullanarak, görüntülerin farklı
    boyutlarda ve yönlerdeki özelliklerini çıkarabilirsiniz.
"""

from PIL import Image#, ImageDraw, ImageFont
import random
import os
import glob


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
