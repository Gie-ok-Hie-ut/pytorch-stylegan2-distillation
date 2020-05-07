import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


def generate_set(args, g_ema, classifier, device):
    with torch.no_grad():
        g_ema.eval()
        classifier.eval()

        print("[Classify & Save Pair]")
        
        if args.attribute == 'gender':
            total_class = 2
        else:
            total_class = 3
        class_count = []
        class_w = []
        for i in range(0, total_class):
            class_count.append(0)
            class_w.append(0)


        conf_threshold = 0.9
        print("Confidence threshold : " + str(conf_threshold))
        while True:

            # Generate Images
            sample_z = torch.randn(args.sample, args.latent, device=device)
            sample, latent = g_ema([sample_z], return_latents = True)

            # Classify Images
            sample_resize = F.interpolate(sample, size=224)
            y1 = classifier.forward(sample_resize)
            y2 = classifier.fc.forward(y1)
            sm = nn.Softmax(dim=1)
            y3 = sm(y2)

            class_predicted = np.argmax(y3.cpu().data, axis=1)
            conf_now = y3[0][class_predicted]

            # Add W
            if (conf_now > conf_threshold) and (class_count[class_predicted.data] < args.pics):
                class_count[class_predicted] += 1
                class_w[class_predicted] += latent

                # Fill All Class More Than args.pics
                print(class_count)
                check = False
                for i in range(0, total_class):
                    if class_count[i] >= args.pics:
                        check = True
                    else:
                        check = False
                        break
                if check == True: break

        # Calculate Mean
        mean_class_prev = 0
        for i in range(0, total_class):
            print("[Calcuate Mean]")
            mean_class = class_w[i] / class_count[i]
            torch.save(mean_class - mean_class_prev, './mean_vector_'+ args.attribute +'.pt')
            mean_class_prev = mean_class
            print(class_count[i])


def generate_multiple(args,g_ema, device):
    with torch.no_grad():
        g_ema.eval()
        g_ema.set_transition('./mean_vector_'+ args.attribute +'.pt')

        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample1 = g_ema([sample_z], transition_degree =  1.0)[0]
            sample2 = g_ema([sample_z], transition_degree =  0.5)[0]
            sample3 = g_ema([sample_z], transition_degree =  0.0)[0]
            sample4 = g_ema([sample_z], transition_degree = -0.5)[0]
            sample5 = g_ema([sample_z], transition_degree = -1.0)[0]

            utils.save_image(sample1, f'sample/{str(i).zfill(6)}_00.png', nrow=1, normalize=True, range=(-1, 1), )
            utils.save_image(sample2, f'sample/{str(i).zfill(6)}_01.png', nrow=1, normalize=True, range=(-1, 1), )
            utils.save_image(sample3, f'sample/{str(i).zfill(6)}_02.png', nrow=1, normalize=True, range=(-1, 1), )
            utils.save_image(sample4, f'sample/{str(i).zfill(6)}_03.png', nrow=1, normalize=True, range=(-1, 1), )
            utils.save_image(sample5, f'sample/{str(i).zfill(6)}_04.png', nrow=1, normalize=True, range=(-1, 1), )


def generate_pair(args, g_ema, classifier, device):

    target_path = args.synthetic_path
    if target_path is None:
        print("Target dataset path is not defined")
        return

    with torch.no_grad():
        g_ema.eval()
        g_ema.set_transition('./mean_vector_'+ args.attribute +'.pt')

        cnt = 0
        while cnt < args.pics:

            # Generate Transition Images
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample1 = g_ema([sample_z], transition_degree =  1.0)[0]
            sample2 = g_ema([sample_z], transition_degree =  0.5)[0]
            sample3 = g_ema([sample_z], transition_degree =  0.0)[0]
            sample4 = g_ema([sample_z], transition_degree = -0.5)[0]
            sample5 = g_ema([sample_z], transition_degree = -1.0)[0]

            sample_batch = torch.cat((sample1, sample2, sample3, sample4, sample5), 0)

            # Classify Images
            sample_prep_batch = F.interpolate(sample_batch, size = 224)

            y1 = classifier.forward(sample_prep_batch)
            y2 = classifier.fc.forward(y1)
            sm = nn.Softmax(dim=1)
            y3 = sm(y2)

            class_list = np.argmax(y3.cpu().data, axis=1)

            # Find Suitable Transition Images
            init_class0 = 0
            for a in range(0,4):
                if class_list[a] < class_list[a+1]:
                    init_class0 = a
                    break

            class0_conf = y3[init_class0][class_list[init_class0]]
            class1_conf = y3[init_class0+1][class_list[init_class0+1]]

            # Save Transition Images
            if class0_conf > 0.9 and class1_conf >0.9:
                print("Save Image: " + str(cnt).zfill(6) + ".png" )
                utils.save_image(sample_batch[init_class0], target_path + '/train_A/' + str(cnt).zfill(6) + '.png', nrow=1,
                                 normalize=True, range=(-1, 1), )
                utils.save_image(sample_batch[init_class0 + 1], target_path + '/train_B/' + str(cnt).zfill(6) + '.png', nrow=1,
                                 normalize=True, range=(-1, 1), )
                cnt+=1

def plot_transition_confidence(args, g_ema, classifier, device):
    with torch.no_grad():
        g_ema.eval()
        g_ema.set_transition('./mean_vector_'+ args.attribute +'.pt')

        # Generate Transition Images
        sample_z = torch.randn(args.sample, args.latent, device=device)

        class0_plot = []
        class1_plot = []

        division = 100

        for i in tqdm(range(division)):
            sample_temp, _ = g_ema([sample_z], transition = i/division)

            sample_temp2 = F.interpolate(sample_temp, size=224)
            y1 = classifier.forward(sample_temp2)
            y2 = classifier.fc.forward(y1)
            sm = nn.Softmax(dim=1)
            y3 = sm(y2)

            print(y3.cpu().numpy())
            class0_plot.append(y3[0][0].cpu().numpy())
            class1_plot.append(y3[0][1].cpu().numpy())


            utils.save_image(sample_temp,
                             f'./sample_transition/{str(i).zfill(6)}.png',
                             nrow=1,
                             normalize=True, range=(-1, 1), )

        plt.plot(class0_plot)
        plt.plot(class1_plot)
        plt.show()


def requires_grad(self, model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', type=str, default='pair', choices=['set', 'pair', 'multiple', 'plot'], help='mode', )
    parser.add_argument('--attribute', type=str, default='gender', choices=['age', 'gender'], help='mode', )
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=200)


    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--stylegan2_ckpt', type=str, default="official_1024_stylegan2-ffhq.pt")
    parser.add_argument('--classifier_ckpt', type=str, default="../0_simple_classifier/checkpoint/ckpt_gender_final.pt")
    parser.add_argument('--synthetic_path', type=str )

    args = parser.parse_args()

    if args.attribute == 'gender':
        total_class = 2
    else:
        total_class = 3
    args.latent = 512
    args.n_mlp = 8

    # Load Generator
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    stylegan2_checkpoint = torch.load(args.stylegan2_ckpt)
    g_ema.load_state_dict(stylegan2_checkpoint['g_ema'])


    # Load VGG16 Classifier
    classifier = models.vgg16(pretrained=False).cuda()

    classifier.fc = nn.Linear(1000,total_class).cuda()
    for p in classifier.parameters():
        p.requires_grad = False
    classifier = torch.load(args.classifier_ckpt)


    if args.phase == 'set':
        # [1] Preprocessing for calculating and saving Mean
        generate_set(args, g_ema, classifier, device)
    elif args.phase == 'pair':
        # [2] Preprocessing for generating [Original Image, Transition Image] Pair
        generate_pair(args, g_ema, classifier, device)
    elif args.phase == 'multiple':
        # [3] Preprocessing for generating 5 transition multiple images
        generate_multiple(args, g_ema, device)
    elif args.phase == 'plot':
        # [4] Plot transition confidence
        plot_transition_confidence(args, g_ema, device)
