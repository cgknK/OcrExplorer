from PIL import Image
import numpy as np

def max_pooling(image_path, pool_size=2, stride=2):
    # Görüntüyü yükle
    img = Image.open(image_path).convert("L")  # Gri tonlamalıya çevir
    img_array = np.array(img)

    # Giriş boyutlarını al
    input_height, input_width = img_array.shape

    # Çıkış boyutlarını hesapla
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    # Çıkış matrisini oluştur
    pooled_output = np.zeros((output_height, output_width), dtype=np.uint8)

    # Havuzlama işlemi
    for i in range(0, input_height - pool_size + 1, stride):
        for j in range(0, input_width - pool_size + 1, stride):
            # Bölgeyi seç
            region = img_array[i:i + pool_size, j:j + pool_size]
            # Maksimum değeri bul
            max_value = np.max(region)
            # Çıkış matrisine yaz
            pooled_output[i // stride, j // stride] = max_value

    # Çıkışı Pillow Image formatına çevir ve kaydet
    pooled_image = Image.fromarray(pooled_output)
    return pooled_image

# Örnek kullanım
image_path = "3_input.jpg"  # Havuzlama yapılacak görüntü dosyası
pooled_image = max_pooling(image_path, pool_size=4, stride=4)  # 4x4 havuzlama, adım 4
pooled_image.show()  # Çıktı görüntüsünü göster
pooled_image.save("3_pooled_output.jpg")  # Çıktıyı kaydet
