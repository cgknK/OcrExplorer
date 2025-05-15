"""
P 1: Harflerin tamamı görüntüde yer almayabiliyor.
P 2: Harflerin başlangıcı 0, 0 iken bile 0, 0'dan başlanarak çizilmiyor.
P 3: A harfi yerine bazen dikdörtgenler var, bu diğer harflerde de olabilir.
P 4: letters'in içermediği karakterli fontlarda dahil ediliyor (uzak doğu gibi)
P 5: Tam anlamıyla parametrize değil kod.
P 6: Gürültünün daha anlamlı olabilmesi için;
    a) sadece harf içerisinde gürültü oluşturulabilmeli,
    b) arka plan gürültüsü, harf ile aynı renkte olabilmeli,
    c) gürültü oldukça? geliştirilmeli.
P 7: Eğitim için oluşturulan sentetik görüntüler rastgele şekilde dönmeli.
P 8: Veri arttırma seçeneği eklenmeli.
P 9: Önce harfi döndürmek daha mantıklı olabilir.
P A: Background noise döndürmeden sonra uygulanmalı. (parametrik)
P B: Char noise karakterin çiziminede uygulanmalı. (parametrik)
P C: Train data set'te italik, kalın, altı çizili gibi çeşitlilik olmalı.
P D: Döndürme işlemleri için "affine transformation veya perspective transformation",
    de kullanılabilir.
    Ayrıca araştır:
        Affine ve Perspective Transformation Kullanımına Uygun Durumlar
    Affine: Düz döndürme, ölçekleme, kaydırma gibi işlemler için uygundur. Örneğin,
        bir metnin belli bir açıyla döndürülmesi ancak paralel çizgilerin korunması
        gerekiyorsa affine dönüşüm tercih edilmelidir.
    Perspective: Eğer harfin eğilmiş, döndürülmüş veya farklı bir perspektiften
    görünüyormuş gibi görünmesi isteniyorsa perspektif dönüşüm tercih edilmelidir.
    Özellikle optik karakter tanıma (OCR) gibi uygulamalarda daha zengin ve gerçekçi
    eğitim verisi üretmek için faydalıdır.
P E: Resize ve scaling ile daha fazla eğitim ve test verisi sağla.
"""
from PIL import Image, ImageDraw, ImageFont
import random
import os
import glob
import math


# Yazı tiplerini yüklemek için bir fonksiyon
def get_available_fonts(fonts_dir="C:/Windows/Fonts"):
    return glob.glob(os.path.join(fonts_dir, "*.ttf"))


def generate_contrasting_color(base_color):
    return tuple((c + 128) % 256 for c in base_color)


def generate_contrasting_color_with_luminance(base_color):
    luminance = 0.299 * base_color[0] + 0.587 * base_color[1] + 0.114 * base_color[2]
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


# Arka plan gürültüsü ekleme fonksiyonu
def add_background_noise(image, intensity=50):
    if intensity == 0:
        return image

    pixels = image.load()
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            noise = tuple(min(255, max(0, pixels[i, j][k] + random.randint(-intensity, intensity))) for k in range(3))
            pixels[i, j] = noise
    return image


# Harfe gürültü ekleme fonksiyonu
def add_text_noise(draw, x, y, text, font, intensity=10):
    if intensity == 0:
        return

    for _ in range(intensity):
        noisy_x = x + random.randint(-intensity, intensity)
        noisy_y = y + random.randint(-intensity, intensity)
        noisy_color = tuple(random.randint(0, 255) for _ in range(3))
        draw.text((noisy_x, noisy_y), text, fill=noisy_color, font=font)


def rotate_image(image, angle, background_color=(255, 255, 255)):
    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=background_color)


# Harf oluşturma fonksiyonu
def create_random_letter_image(letter, output_path, fonts):
    # Resim boyutları
    width, height = random.randint(40, 200), random.randint(40, 200)

    # Rastgele renk oluştur
    background_color = tuple(random.randint(0, 255) for _ in range(3))
    text_color = tuple(random.randint(0, 255) for _ in range(3))

    # Resim oluştur
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Yazı tipi ayarla (rastgele seçiliyor)
    font_path = random.choice(fonts)
    try:
        font_size = random.randint(10, min(width, height) - 1)
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"{font_path} yüklenemedi. Varsayılan bir yazı tipi kullanılacak.")
        font = ImageFont.load_default()

    # Harf boyutlarını al (textbbox kullanarak)
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Harf için rastgele merkezileştirilmiş bir konum belirle
    max_attempts = 100  # Rastgele deneme sayısı
    for _ in range(max_attempts):
        x = random.randint(0, max(0, width - text_width))
        y = random.randint(0, max(0, height - text_height))
        # Koşul: Harfin dışarı taşmaması
        if x >= 0 and y >= 0 and (x + text_width <= width) and (y + text_height <= height):
            break
    else:
        # Eğer tüm denemeler başarısız olursa merkeze yerleştir
        x = random.randint(0, max(0, (width - text_width) // 2))
        y = random.randint(0, max(0, (height - text_height) // 2))

    # Harfi çiz
    draw.text((x, y), letter, fill=text_color, font=font)

    suffix_1 = ""
    suffix_2 = ""
    suffix_3 = ".jpg"
    letter_noise_probability = 0.05
    background_noise_probability = 0.1
    # Harfe gürültü ekle
    if random.random() < letter_noise_probability:
        add_text_noise(draw, x, y, letter, font, intensity=5)
        suffix_1 = "text_"
        suffix_3 = "noise.jpg"

    # Arka plan gürültüsü ekle
    if random.random() < background_noise_probability:
        image = add_background_noise(image, intensity=25)
        suffix_2 = "background_"
        suffix_3 = "noise.jpg"

    if random.random() < 0.5:
        background_color_rotate = background_color
        if random.random() < 0.5:
            background_color_rotate = tuple(random.randint(0, 255) for _ in range(3))

        # Harfi rastgele bir açıyla döndür
        random_angle = random.uniform(-180, 180)
        image = rotate_image(image, random_angle, background_color_rotate)

    # Resmi kaydet
    #font_name = font.path.split("\\")[-1].split(".")[0]
    #print(font_name)
    #print(font.getname())
    font_name = font.getname()[0] + "_"
    image.save(output_path + font_name + suffix_1 + suffix_2 + suffix_3) # isim ve uzantı neden yazmıyor?
    #print(f"{output_path} dosyası oluşturuldu.")


def main():
    # Örnek kullanım
    output_dir = "0_train_data"
    os.makedirs(output_dir, exist_ok=True)

    # Yazı tiplerini al
    fonts = get_available_fonts()

    if not fonts:
        print("Hiçbir yazı tipi bulunamadı!")
    else:
        # Rastgele harfler oluştur
        letters = "ABCDEFGH"
        piece = 5
        width_piece = math.floor(math.log10(abs(piece - 1))) + 1 if piece != 0 else 1
        print(width_piece)
        for i, letter in enumerate(letters):
            for _ in range(piece):
                output_path = os.path.join(output_dir, f"{letter}_{i}{_:0{width_piece}}_")
                create_random_letter_image(letter, output_path, fonts)
            print(i, letter, f"+{piece}")


if __name__ == "__main__":
    main()
