from PIL import Image, ImageDraw, ImageFont

# Test fonksiyonu
def test_letter_drawing():
    # Resim boyutları
    width, height = 52, 53
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

    # Resim oluştur
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    # Harfi çiz (x, y) = (0, 0)
    letter = "A"
    x, y = 0, 0
    draw.text((x, y), letter, fill=text_color, font=font)

    # Harfin sınır kutusunu al ve çerçeve çiz
    bbox = draw.textbbox((x, y), letter, font=font)
    draw.rectangle(bbox, outline="red", width=2)

    # Resmi kaydet veya göster
    image.show()  # Varsayılan görüntüleyicide göster
    image.save("0_test_char_A.jpg")  # Kaydetmek için yorum kaldırabilirsiniz

# Testi çalıştır
if __name__ == "__main__":
    test_letter_drawing()
