from PIL import Image

raw_images_file = open('../data/raw_image_list.txt', 'r')
resize_images_file = open('../data/resize_image_list.txt', 'w')

project_path = '/Users/chris/Classes/CS 412/classproject/leaf_project/data'

resize_to = 128, 128  # pixels

for raw_image_location in raw_images_file:
    img = Image.open(project_path + raw_image_location[1:-1])
    img.thumbnail(resize_to, Image.ANTIALIAS)
    img.save(project_path + raw_image_location[1:-5]+'_resize.jpg', "JPEG")
    img.close()

    resize_images_file.write(raw_image_location[:-5]+'_resize.jpg\n')

raw_images_file.close()
resize_images_file.close()
