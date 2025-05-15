from PIL import Image

img = Image.open("4_A_012_Haettenschweiler_.jpg")
img = img.convert("L")  # Grayscale
img = img.resize((50, 50), )
img.show()
img.save("4_save.png")
