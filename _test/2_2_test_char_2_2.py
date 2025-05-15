from PIL import Image, ImageDraw, ImageFont

# Test fonksiyonu
def test_letter_drawing():
    # Resim boyutları ve renkler
    image_size = (250, 250)  # Görüntü boyutu
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
    # Geçici bir görüntüde harfin kutusunu hesapla
    temp_image = Image.new("RGB", (1, 1), background_color)
    temp_draw = ImageDraw.Draw(temp_image)
    bbox = temp_draw.textbbox((0, 0), letter, font=font)

    # Harfin sınır kutusunun (0, 0) noktasına göre ofsetini hesapla
    x_offset = -bbox[0]
    y_offset = -bbox[1]

    # Görüntü oluştur
    image = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(image)

    # Harfi (0, 0) noktasına hizala
    draw.text((x_offset, y_offset), letter, fill=text_color, font=font)

    # Harfin sınır kutusunu çiz
    draw.rectangle(
        [bbox[0] + x_offset, bbox[1] + y_offset, bbox[2] + x_offset, bbox[3] + y_offset],
        outline="red", width=2
    )

    # Resmi kaydet veya göster
    image.show()  # Varsayılan görüntüleyicide göster
    image.save("2_2_test_char_A_aligned.jpg")  # Kaydetmek için yorum kaldırabilirsiniz

# Testi çalıştır
if __name__ == "__main__":
    test_letter_drawing()
