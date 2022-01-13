from PIL import Image

with Image.open('westbrook.jpg') as image:
    w, h = image.size
    #rgb_im = im.convert('RGB')

    for x in range(w):
        for y in range(h):
            r, g, b = image.getpixel((x, y))
            image.putpixel((x, y), (int(r/2), int(g/2), int(b/2)))
    
    image.show()
    image.save('Q2.jpg')

