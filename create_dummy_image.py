from PIL import Image

# Create a small black image
img = Image.new('RGB', (10, 10), color = 'black')
img.save('dummy_image1.png')

# Create a slightly different image (e.g., one white pixel)
img_variant = Image.new('RGB', (10, 10), color = 'black')
img_variant.putpixel((0,0), (255,255,255)) # one white pixel
img_variant.save('dummy_image2.png')

img_another = Image.new('RGB', (10, 10), color = 'white')
img_another.save('dummy_image3.png')

print("Dummy images created: dummy_image1.png, dummy_image2.png, dummy_image3.png")
