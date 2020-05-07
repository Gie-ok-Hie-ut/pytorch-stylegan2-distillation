# Program For Renaming Images in Ascending Order

import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def main():
    cur_dir = './'
    paths = sorted(make_dataset(cur_dir, 5000))


    for i in range(len(paths)):
        from_name1 = paths[i]
        to_name1 = cur_dir + '\\' + str(i).zfill(6) + '.png'

        from_name2 = cur_dir + '_w\\' + paths[i].split('\\')[6].split('.')[0] + '.pt'
        to_name2 = cur_dir + '_w\\' + str(i).zfill(6) + '.pt'

        os.rename(from_name1, to_name1, )
        os.rename(from_name2, to_name2, )

    return

if __name__ == '__main__':
    main()