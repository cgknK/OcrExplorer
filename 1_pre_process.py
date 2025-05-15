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
P 5: CNN ile RNN hafıza özelliğini birleştirmeye çalış (saçma ise düşün).
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


def test_sort_vs_sorted():
    import time

    test_list = glob.glob(os.path.join("0_train_data", "*"))

    def measure_time(func):
        start_time = time.time()
        cpu_start_time = time.process_time()
        func(test_list)
        cpu_end_time = time.process_time()
        end_time = time.time()
        cpu_time = cpu_end_time - cpu_start_time
        wall_time = end_time - start_time
        print(f"CPU Time: {cpu_time:.6f} seconds")
        print(f"Wall Time: {wall_time:.6f} seconds")

    def test_sorted(test_list):
        for i in range(1_000_000):
            fonts = sorted(test_list)

    def test_sort(test_list):
        for j in range(1_000_000):
            fonts = test_list.sort()

    print("\nTest Sorted:")
    measure_time(test_sorted)

    print("\nTest Sort:")
    measure_time(test_sort)

    print("\nTest Sorted:")
    measure_time(test_sorted)

    print("\nTest Sort:")
    measure_time(test_sort)

    print("\nTest Sorted:")
    measure_time(test_sorted)

    print("\nTest Sort:")
    measure_time(test_sort)


def get_available_train_data(train_data_dir="0_train_data"):
    train_data = glob.glob(os.path.join(train_data_dir, "*"))
    train_data.sort() # Muhtemelen zaten sıralı, windows için
    #print(train_data == glob.glob(os.path.join(train_data_dir, "*")))
    return train_data


def resize_image(image, new_size):
    """
    Resizes the given image to the specified new size.

    Args:
        image (str): The path to the image file.
        new_size (tuple): The new size of the image (width, height).

    Returns:
        Image: The resized image.
    """
    img = Image.open(image)
    img = img.resize(new_size)
    return img


def scale_image(image, scale_factor):
    """
    Scales the given image by the specified scale factor.

    Args:
        image (str): The path to the image file.
        scale_factor (float): The scale factor.

    Returns:
        Image: The scaled image.
    """
    img = Image.open(image)
    width, height = img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    """
    if new_width < width or new_height < height:
        img = img.resize((new_width, new_height), Image.LANCZOS)
    else:
        img = img.resize((new_width, new_height), Image.BICUBIC) #zaten default
    """
    img = img.resize((new_width, new_height))

    return img


def add_padding_to_image(image, padding_size):
    """
    Adds padding to the given image.

    Args:
        image (str): The path to the image file.
        padding_size (tuple): The size of the padding (left, top, right, bottom).

    Returns:
        Image: The image with padding.
    """
    img = Image.open(image)
    width, height = img.size
    left, top, right, bottom = padding_size
    new_width = width + left + right
    new_height = height + top + bottom
    new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    new_img.paste(img, (left, top)) #default parametre mask=None
    return new_img


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


