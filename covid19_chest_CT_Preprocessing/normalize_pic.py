import os
from PIL import Image


def resize_by_label(type, label):
    cur_dir = os.getcwd()
    filename = os.listdir(cur_dir + '/' + type + '-images-gray/' + str(label))
    size_m = 64
    size_n = 64
    base_dir = cur_dir + '/' + type + '-images-gray/' + str(label) + '/'
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    new_dir = cur_dir + '/' + type + '-images/' + str(label)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    order = 0
    for img in filename:
        image = Image.open(base_dir + img)
        image_size = image.resize((size_m, size_n), Image.ANTIALIAS)
        print('\rprocess %f%%' % (order/len(filename)*100), end='')
        order += 1
        image_size.save(new_dir + '/' + str(label) + type + '_' + str(order) + ".png")

def colorful_to_gray(input_img_path, output_img_path):
    """
    彩色图转单色图
    :param input_img_path: 图片路径
    :param output_img_path: 输出图片路径
    """

    img = Image.open(input_img_path)
    # 转化为黑白图片
    img = img.convert("L")
    img.save(output_img_path)


def c2g(dataset_type, label):
    dataset_dir = '/Users/jacklin/PycharmProjects/Final_Thesis/covid19-chest_CT_test/' + dataset_type + '-images-color/' + str(label)
    base_dir = output_dir = '/Users/jacklin/PycharmProjects/Final_Thesis/covid19-chest_CT_test/' + dataset_type + '-images-gray'
    output_dir = '/Users/jacklin/PycharmProjects/Final_Thesis/covid19-chest_CT_test/' + dataset_type + '-images-gray/' + str(label)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # 获得需要转化的图片路径并生成目标路径
    image_filenames = [(
        os.path.join(dataset_dir, file_dir),
        os.path.join(output_dir, file_dir)
    ) for file_dir in os.listdir(dataset_dir)[1:]]
    # 转化所有图片
    for path in image_filenames:
        print(path)
        if path[0][-9::] == '.DS_Store':
            print(path[0][-9::])
            continue
        elif path[0][-7::] == '.icloud':
            print(path[0][-9::])
            continue
        colorful_to_gray(path[0], path[1])
if __name__ == "__main__":
    for type in ['test', 'train']:
        for lable in [0,1,2]:
            c2g(type,lable)
    for type in ['test', 'train']:
        for lable in [0,1,2]:
            resize_by_label(type,lable)

