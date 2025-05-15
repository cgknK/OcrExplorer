from PIL import Image, ImageDraw, ImageFont

# Test fonksiyonu
def test_letter_drawing():
    image = Image.new("RGB", (250, 250), (250, 250, 250))

    # Resim boyutları ve renkler
    background_color = (255, 255, 255)  # Beyaz arka plan
    text_color = (0, 0, 0)  # Siyah yazı rengi

    # Yazı tipi ayarı (Windows sisteminde bir yazı tipi seçimi)
    font_path = "C:/Windows/Fonts/arial.ttf"  # Mevcut bir yazı tipi yolu
    font_size = 50
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Yazı tipi yüklenemedi. Varsayılan bir yazı tipi kullanılacak.")
        font = ImageFont.load_default()

    # Harfi çizilecek olan metin
    letter = "A"

    # Yazı tipi metriklerini al
    ascent, descent = font.getmetrics()
    # Ascent'i sıfır noktasına telafi edecek şekilde bir `y_offset` hesapla
    y_offset = ascent

    # Harf kutusunu hesapla
    temp_image = Image.new("RGB", (1, 1), background_color)  # Geçici bir görüntü
    temp_draw = ImageDraw.Draw(temp_image)
    bbox = temp_draw.textbbox((0, 0), letter, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Görüntü boyutunu harfin boyutuna göre ayarla
    padding = 10  # Harf kutusuna ek kenarlık
    width, height = text_width + padding * 2, text_height + padding * 2
    image = image#Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Harfi tam `(0, 0)`'dan başlatacak şekilde hizala
    x, y = 0, -y_offset
    draw.text((x, y), letter, fill=text_color, font=font)

    # Harfin sınır kutusunu çiz
    draw.rectangle(
        [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y],
        outline="red", width=2
    )

    # Resmi kaydet veya göster
    image.show()  # Varsayılan görüntüleyicide göster
    image.save("2_0_test_char_A_adjusted.jpg")  # Kaydetmek için yorum kaldırabilirsiniz

# Testi çalıştır
if __name__ == "__main__":
    test_letter_drawing()
