import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from scipy.io import loadmat
from datetime import datetime


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default="imdb",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=256,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()

    return args


def main_gender():
    args = get_args()
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    output_path = "./imdb_mat/imdb"
    mat_path = "./imdb_mat/" + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    print("Gender: Training-Set : Folder from 00 ~ 89")
    out_imgs = []
    out_genders = []
    valid_sample_num = 0

    sample_num = 460723
    for i in tqdm(range(sample_num)):
        if np.isnan(gender[i]):
            continue
        if(int(full_path[i][0].split('/')[0]) >= 90):
            continue

        out_imgs.append(str(full_path[i][0]))
        out_genders.append(int(gender[i]))

        valid_sample_num += 1

    output = {"image": np.array(out_imgs), "gender": np.array(out_genders)}
    scipy.io.savemat(output_path + "_gender_train.mat", output)

    print("Gender: Testing-Set : Folder from 90 ~ 99")
    out_imgs = []
    out_genders = []
    valid_sample_num = 0

    sample_num = 460723
    for i in tqdm(range(sample_num)):
        if np.isnan(gender[i]):
            continue
        if(int(full_path[i][0].split('/')[0]) < 90):
            continue

        out_imgs.append(str(full_path[i][0]))
        out_genders.append(int(gender[i]))

        valid_sample_num += 1

    output = {"image": np.array(out_imgs), "gender": np.array(out_genders)}
    scipy.io.savemat(output_path + "_gender_test.mat", output)


def main_age():
    args = get_args()
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    output_path = "./imdb_mat/imdb"
    mat_path = "./imdb_mat/" + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)


    print("Age is divided into three groups : ~25 , 35~45, 60~")
    print("Age: Training-Set : Folder from 00 ~ 89")
    out_imgs = []
    out_ages = []
    valid_sample_num = 0

    sample_num = 460723
    for i in tqdm(range(sample_num)):
        if np.isnan(age[i]):
            continue
        if(int(full_path[i][0].split('/')[0]) >= 90):
            continue

        agetp = int(age[i])
        if (agetp > 25 and agetp < 35) or (agetp > 45 and agetp < 60 ):
            continue
        if agetp < 0 :
            continue

        out_imgs.append(str(full_path[i][0]))
        out_ages.append(int(age[i]))

        valid_sample_num += 1

    output = {"image": np.array(out_imgs), "age": np.array(out_ages)}
    scipy.io.savemat(output_path + "_age_train.mat", output)


    print("Age: Testing-Set : Folder from 90 ~ 99")
    out_imgs = []
    out_ages = []
    valid_sample_num = 0

    sample_num = 460723
    for i in tqdm(range(sample_num)):
        if np.isnan(age[i]):
            continue
        if(int(full_path[i][0].split('/')[0]) < 90):
            continue

        agetp = int(age[i])
        if (agetp > 25 and agetp < 35) or (agetp > 45 and agetp < 60 ):
            continue
        if agetp < 0 :
            continue

        out_imgs.append(str(full_path[i][0]))
        out_ages.append(int(age[i]))

        valid_sample_num += 1

    output = {"image": np.array(out_imgs), "age": np.array(out_ages)}
    scipy.io.savemat(output_path  + "_age_test.mat", output)
    
    

if __name__ == '__main__':
    print("There should be './imdb_mat/imdb.mat' file (download it from IMDB-WIKI dataset)")
    main_gender()
    main_age()