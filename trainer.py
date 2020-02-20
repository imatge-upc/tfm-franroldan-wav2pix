import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from onehot2image_dataset import OneHot2YoutubersDataset, Rescale
from txt2image_dataset import Text2ImageDataset
from models.gan_factory import gan_factory
from utils import Utils, Logger
from PIL import Image
import os
import numpy as np

class Trainer(object):
    def __init__(self, type, dataset, split, lr, lr_lower_boundary, lr_update_type, lr_update_step, diter, vis_screen, save_path, l1_coef, l2_coef, pre_trained_gen,
                 pre_trained_disc, batch_size, num_workers, epochs, h, scale_size, num_channels, k, lambda_k, gamma, project, concat):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f)

        self.generator = gan_factory.generator_factory(type, dataset, batch_size, h, scale_size, num_channels).cuda()
        #self.generator = gan_factory.generator_factory(type, dataset, batch_size, h, scale_size, num_channels)
        self.discriminator = gan_factory.discriminator_factory(type, batch_size, h, scale_size, num_channels).cuda()
        #self.discriminator = gan_factory.discriminator_factory(type, batch_size, h, scale_size, num_channels)
        print(self.discriminator)
        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
            #self.discriminator.load_state_dict(torch.load(pre_trained_disc), map_location = "cpu")
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
            #self.generator.load_state_dict(torch.load(pre_trained_gen), map_location = "cpu")
        else:
            self.generator.apply(Utils.weights_init)
        self.dataset_name = dataset
        if self.dataset_name == 'birds':
            self.dataset = Text2ImageDataset(config['birds_dataset_path'], split=split)
        elif self.dataset_name == 'flowers':
            self.dataset = Text2ImageDataset(config['flowers_dataset_path'], split=split)
        elif self.dataset_name =='youtubers':
            self.dataset = OneHot2YoutubersDataset(config['youtubers_dataset_path'],
                                                   transform=Rescale(64),
                                                   split=split)
        else:
            print('Dataset not supported, please select either birds or flowers.')
            exit()

        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.lr_lower_boundary = lr_lower_boundary
        self.lr_update_type = lr_update_type
        self.lr_update_step = lr_update_step
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.DITER = diter
        self.apply_projection = project
        self.apply_concat = concat

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.h = h
        self.scale_size = scale_size
        self.num_channels = num_channels
        self.k = k
        self.lambda_k = lambda_k
        self.gamma = gamma

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers)

        self.optimD = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), lr=0.0004, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), lr=0.0001, betas=(self.beta1, 0.999))

        self.logger = Logger(vis_screen, save_path)
        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path
        self.type = type


    def train(self, cls=False):

        if self.type == 'wgan':
            self._train_wgan(cls)
        elif self.type == 'gan':
            self._train_gan(cls)
        elif self.type == 'vanilla_wgan':
            self._train_vanilla_wgan()
        elif self.type == 'vanilla_gan':
            self._train_vanilla_gan()
        elif self.type == 'began':
            self._train_began()
        elif self.type =='acgan':
            self._train_acgan()
        elif self.type == 'segan':
            self.train_segan_and_lsgan()

    def _train_wgan(self, cls):
        print('Starting training for WGAN...')
        one = torch.FloatTensor([1])
        mone = one * -1

        one = Variable(one).cuda()
        #one = Variable(one)
        mone = Variable(mone).cuda()
        #one = Variable(mone)

        gen_iteration = 0
        for epoch in range(self.num_epochs):
            iterator = 0
            data_iterator = iter(self.data_loader)

            while iterator < len(self.data_loader):

                if gen_iteration < 25 or gen_iteration % 500 == 0:
                    d_iter_count = 100
                else:
                    d_iter_count = self.DITER

                d_iter = 0

                # Train the discriminator
                while d_iter < d_iter_count and iterator < len(self.data_loader):
                    d_iter += 1

                    for p in self.discriminator.parameters():
                        p.requires_grad = True

                    self.discriminator.zero_grad()

                    sample = next(data_iterator)
                    iterator += 1
                    if self.dataset_name != 'youtubers':
                        right_images = sample['right_images']
                        right_embed = sample['right_embed']
                        wrong_images = sample['wrong_images']

                        right_images = Variable(right_images.float()).cuda()
                        #right_images = Variable(right_images.float())
                        right_embed = Variable(right_embed.float()).cuda()
                        #right_embed = Variable(right_embed.float())
                        wrong_images = Variable(wrong_images.float()).cuda()
                        #wrong_images = Variable(wrong_images.float())
                    else:
                        right_images = sample['face']
                        right_embed = sample['onehot']

                        right_images = Variable(right_images.float()).cuda()
                        #right_images = Variable(right_images.float())
                        right_embed = Variable(right_embed.float()).cuda() #Change to long if using embedding layer.
                        #right_embed = Variable(right_embed.float())
                    outputs, _ = self.discriminator(right_images, right_embed, project=self.apply_projection, concat=self.apply_concat)
                    real_loss = torch.mean(outputs)
                    real_loss.backward(mone)

                    if cls:
                        outputs, _ = self.discriminator(wrong_images, right_embed, project=self.apply_projection, concat=self.apply_concat)
                        wrong_loss = torch.mean(outputs)
                        wrong_loss.backward(one)

                    noise = Variable(torch.randn(right_images.size(0), self.noise_dim), volatile=True).cuda()
                    #noise = Variable(torch.randn(right_images.size(0), self.noise_dim), volatile=True)
                    noise = noise.view(noise.size(0), self.noise_dim, 1, 1)

                    fake_images = Variable(self.generator(right_embed, noise, project=self.apply_projection).data)
                    outputs, _ = self.discriminator(fake_images, right_embed, project=self.apply_projection, concat=self.apply_concat)
                    fake_loss = torch.mean(outputs)
                    fake_loss.backward(one)

                    ## NOTE: Pytorch had a bug with gradient penalty at the time of this project development
                    ##       , uncomment the next two lines and remove the params clamping below if you want to try gradient penalty
                    gp = Utils.compute_GP(self.discriminator, right_images.data, right_embed, fake_images.data, LAMBDA=10,
                                          project=self.apply_projection)
                    gp.backward()

                    d_loss = real_loss - fake_loss

                    if cls:
                        d_loss = d_loss - wrong_loss

                    self.optimD.step()

                    for p in self.discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                # Train Generator
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                #noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise, project=self.apply_projection)
                outputs, _ = self.discriminator(fake_images, right_embed, project=self.apply_projection, concat=self.apply_concat)

                g_loss = torch.mean(outputs)
                #g_loss = torch.sum((outputs)**2)
                g_loss.backward(mone)
                g_loss = - g_loss
                self.optimG.step()

                gen_iteration += 1

                #self.logger.draw(right_images, fake_images)
                self.logger.log_iteration_wgan(epoch, gen_iteration, d_loss, g_loss, real_loss, fake_loss)
                
            #self.logger.plot_epoch(gen_iteration)

            if (epoch+1) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path,self.save_path,  epoch)

    def _train_gan(self, cls):
        criterion = nn.MSELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0
        print('Starting train for GAN')
        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                if self.dataset_name != 'youtubers':

                    right_images = sample['right_images']
                    right_embed = sample['right_embed']
                    wrong_images = sample['wrong_images']
                    wrong_images = Variable(wrong_images.float()).cuda()
                    #wrong_images = Variable(wrong_images.float())
                else:
                    right_images = sample['face']
                    right_embed = sample['onehot']
                right_images = Variable(right_images.float()).cuda()
                #right_images = Variable(right_images.float())
                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))
                auxiliary_noise = torch.randn(right_embed.size(0),128)
                #right_embed = right_embed + auxiliary_noise
                auxiliary_noise = Variable(auxiliary_noise.float()).cuda()
                #auxiliary_noise = Variable(auxiliary_noise.float())
                right_embed = Variable(right_embed.float()).cuda()
                #right_embed = Variable(right_embed.float())

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).cuda()
                #real_labels = Variable(real_labels)
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                #smoothed_real_labels = Variable(smoothed_real_labels)
                fake_labels = Variable(fake_labels).cuda()
                #fake_labels = Variable(fake_labels)

                # Train the discriminator
                self.discriminator.zero_grad()


                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                #noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images, projected_embed = self.generator(right_embed, noise, auxiliary_noise, project=self.apply_projection)
                outputs, _ = self.discriminator(fake_images, projected_embed, project=self.apply_projection, concat=self.apply_concat)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                outputs, activation_real = self.discriminator(right_images, projected_embed, project=self.apply_projection, concat=self.apply_concat)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                if cls:
                    outputs, _ = self.discriminator(wrong_images, projected_embed, project=self.apply_projection, concat=self.apply_concat)
                    wrong_loss = criterion(outputs, fake_labels)
                    wrong_score = outputs


                d_loss = real_loss + fake_loss

                if cls:
                    d_loss = d_loss + wrong_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images, projected_embed = self.generator(right_embed, noise, auxiliary_noise, project=self.apply_projection)
                outputs, activation_fake = self.discriminator(fake_images, projected_embed, project=self.apply_projection, concat=self.apply_concat)
                _, activation_real = self.discriminator(right_images, projected_embed, project=self.apply_projection, concat=self.apply_concat)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)


                #======= Generator Loss function============
                # This is a customized loss function, the first term is the mean square error loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                #===========================================
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)

                g_loss.backward()
                self.optimG.step()

                if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch,d_loss, g_loss, real_score, fake_score)
                    #self.logger.draw(right_images, fake_images)

            #self.logger.plot_epoch_w_scores(epoch)

            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)

    def _train_acgan(self):
        """ Based on https://github.com/gitlimlab/ACGAN-PyTorch"""

        # Function to compute the current classification accuracy
        def compute_acc(preds, labels):
            #TODO: Correct this function!!!
            correct = 0
            preds_ = preds.data.max(1)[1]
            correct = preds_.eq(labels.data).cpu().sum()
            acc = float(correct) / float(len(labels.data)) * 100.0
            return acc

        avg_loss_D = 0.0
        avg_loss_G = 0.0
        avg_loss_A = 0.0
        # loss functions
        disc_criterion = nn.MSELoss()
        aux_criterion = nn.NLLLoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0
        print('Starting training GAN with auxiliary classifier')
        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                if self.dataset_name != 'youtubers':

                    right_images = sample['right_images']
                    right_embed = sample['right_embed']
                    wrong_images = sample['wrong_images']
                    wrong_images = Variable(wrong_images.float()).cuda()
                    #wrong_images = Variable(wrong_images.float())
                else:
                    right_images = sample['face']
                    right_embed_noextra = sample['onehot']
                    raw_wav = sample['audio']
                    fake_aux = torch.cat([torch.zeros(right_embed_noextra.size(1)), torch.ones(1)], 0)\
                        .unsqueeze(1).permute(1, 0).repeat(right_embed_noextra.size(0), 1)
                    right_embed = torch.cat([right_embed_noextra, torch.zeros(right_embed_noextra.size(0), 1)], 1)
                    wrong_images = sample['wrong_face']
                raw_wav = Variable(raw_wav.float()).cuda()
                #raw_wav = Variable(raw_wav.float())
                right_images = Variable(right_images.float()).cuda()
                #right_images = Variable(right_images.float())
                right_embed = Variable(right_embed.float()).cuda()
                #right_embed = Variable(right_embed.float())
                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))
                right_embed_noextra = Variable(right_embed_noextra.float()).cuda()
                #right_embed_noextra = Variable(right_embed_noextra.float())
                wrong_images = Variable(wrong_images.float()).cuda()
                #wrong_images = Variable(wrong_images.float())

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).cuda()
                #real_labels = Variable(real_labels)
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                #smoothed_real_labels = Variable(smoothed_real_labels)
                fake_labels = Variable(fake_labels).cuda()
                #fake_labels = Variable(fake_labels)

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, aux_output, activation_real = self.discriminator(right_images, right_embed, project=self.apply_projection)
                _, targets = right_embed.max(dim=1)
                targets = Variable(torch.LongTensor(targets.cpu()))
                dis_errD_real = disc_criterion(outputs.squeeze(), smoothed_real_labels).cuda()
                #dis_errD_real = disc_criterion(outputs.squeeze(), smoothed_real_labels)
                aux_errD_real = aux_criterion(aux_output.cpu().float(), targets.cpu().long()).cuda()
                #aux_errD_real = aux_criterion(aux_output.cpu().float(), targets.cpu().long())
                errD_real = dis_errD_real + aux_errD_real
                #errD_real.backward(retain_graph=True)
                D_x = outputs.data.mean()
                # compute the current classification accuracy
                accuracy = compute_acc(aux_output.cuda(), real_labels.cuda().long())
                #accuracy = compute_acc(aux_output, real_labels.long())
                _, targets_fake = fake_aux.max(dim=1)
                targets_fake = Variable(torch.LongTensor(targets_fake.cpu()))
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                #noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images, _ = self.generator(raw_wav, right_embed_noextra, noise, noise, project=self.apply_projection)
                outputs, aux_output, activation_fake= self.discriminator(fake_images, fake_aux, project=self.apply_projection)
                dis_errD_fake = disc_criterion(outputs.squeeze(), fake_labels).cuda()
                #dis_errD_fake = disc_criterion(outputs.squeeze(), fake_labels)
                aux_errD_fake = aux_criterion(aux_output.cpu().float(), targets_fake.cpu().long()).cuda()
                #aux_errD_fake = aux_criterion(aux_output.cpu().float(), targets_fake.cpu().long())
                outputs, aux_output, activation_fake = self.discriminator(wrong_images, fake_aux, project=self.apply_projection)
                cls_errD_fake = disc_criterion(outputs.squeeze(), fake_labels).cuda()
                #cls_errD_fake = disc_criterion(outputs.squeeze(), fake_labels)
                cls_aux_errD_fake = aux_criterion(aux_output.cpu().float(), targets_fake.cpu().long()).cuda()
                #cls_aux_errD_fake = aux_criterion(aux_output.cpu().float(), targets_fake.cpu().long())
                errD_fake = dis_errD_fake + aux_errD_fake + cls_errD_fake + cls_aux_errD_fake
                #errD_fake.backward(retain_graph=True)
                D_G_z1 = outputs.data.mean()
                errD = errD_real + errD_fake
                errD.backward(retain_graph=True)
                self.optimD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.zero_grad()
                disc_output, aux_output, activation_fake = self.discriminator(fake_images, fake_aux, project=self.apply_projection)
                dis_errG = disc_criterion(disc_output.squeeze(), real_labels).cuda()
                #dis_errG = disc_criterion(disc_output.squeeze(), real_labels)
                aux_errG = aux_criterion(aux_output, targets.cuda().long()).cuda()
                #aux_errG = aux_criterion(aux_output, targets.long())
                errG = dis_errG - aux_errG\
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)
                errG.backward()
                D_G_z2 = disc_output.data.mean()
                self.optimG.step()

                # compute the average loss
                curr_iter = epoch * len(self.data_loader) + iteration
                all_loss_G = avg_loss_G * curr_iter
                all_loss_D = avg_loss_D * curr_iter
                all_loss_A = avg_loss_A * curr_iter
                all_loss_G += errG.data[0]
                all_loss_D += errD.data[0]
                all_loss_A += accuracy
                avg_loss_G = all_loss_G / (curr_iter + 1)
                avg_loss_D = all_loss_D / (curr_iter + 1)
                avg_loss_A = all_loss_A / (curr_iter + 1)

                print(
                    '[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
                    % (epoch, self.num_epochs, iteration, len(self.data_loader),
                       errD.data[0], avg_loss_D, errG.data[0], avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
                
                """if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch,d_loss, g_loss, errD_real, fake_score)
                    #self.logger.draw(right_images, fake_images)"""

            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)


    def _train_vanilla_wgan(self):
        one = Variable(torch.FloatTensor([1])).cuda()
        #one = Variable(torch.FloatTensor([1]))
        mone = one * -1
        gen_iteration = 0

        for epoch in range(self.num_epochs):
         iterator = 0
         data_iterator = iter(self.data_loader)

         while iterator < len(self.data_loader):

             if gen_iteration < 25 or gen_iteration % 500 == 0:
                 d_iter_count = 100
             else:
                 d_iter_count = self.DITER

             d_iter = 0

             # Train the discriminator
             while d_iter < d_iter_count and iterator < len(self.data_loader):
                 d_iter += 1

                 for p in self.discriminator.parameters():
                     p.requires_grad = True

                 self.discriminator.zero_grad()

                 sample = next(data_iterator)
                 iterator += 1

                 right_images = sample['right_images']
                 right_images = Variable(right_images.float()).cuda()
                 #right_images = Variable(right_images.float())

                 outputs, _ = self.discriminator(right_images)
                 real_loss = torch.mean(outputs)
                 real_loss.backward(mone)

                 noise = Variable(torch.randn(right_images.size(0), self.noise_dim), volatile=True).cuda()
                 #noise = Variable(torch.randn(right_images.size(0), self.noise_dim), volatile=True)
                 noise = noise.view(noise.size(0), self.noise_dim, 1, 1)

                 fake_images = Variable(self.generator(noise).data)
                 outputs, _ = self.discriminator(fake_images)
                 fake_loss = torch.mean(outputs)
                 fake_loss.backward(one)

                 ## NOTE: Pytorch had a bug with gradient penalty at the time of this project development
                 ##       , uncomment the next two lines and remove the params clamping below if you want to try gradient penalty
                 # gp = Utils.compute_GP(self.discriminator, right_images.data, right_embed, fake_images.data, LAMBDA=10)
                 # gp.backward()

                 d_loss = real_loss - fake_loss
                 self.optimD.step()

                 for p in self.discriminator.parameters():
                     p.data.clamp_(-0.01, 0.01)

             # Train Generator
             for p in self.discriminator.parameters():
                 p.requires_grad = False

             self.generator.zero_grad()
             noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
             #noise = Variable(torch.randn(right_images.size(0), 100))
             noise = noise.view(noise.size(0), 100, 1, 1)
             fake_images = self.generator(noise)
             outputs, _ = self.discriminator(fake_images)

             g_loss = torch.mean(outputs)
             g_loss.backward(mone)
             g_loss = - g_loss
             self.optimG.step()

             gen_iteration += 1

             #self.logger.draw(right_images, fake_images)
             self.logger.log_iteration_wgan(epoch, gen_iteration, d_loss, g_loss, real_loss, fake_loss)

         #self.logger.plot_epoch(gen_iteration)

         if (epoch + 1) % 50 == 0:
             Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, epoch)

    def _train_vanilla_gan(self):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0

        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                right_images = sample['right_images']

                right_images = Variable(right_images.float()).cuda()
                #right_images = Variable(right_images.float())

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).cuda()
                #real_labels = Variable(real_labels)
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                #smoothed_real_labels = Variable(smoothed_real_labels)
                fake_labels = Variable(fake_labels).cuda()
                #fake_labels = Variable(fake_labels)


                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                #noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(noise)
                outputs, _ = self.discriminator(fake_images)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                #noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(noise)
                outputs, activation_fake = self.discriminator(fake_images)
                _, activation_real = self.discriminator(right_images)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                # ======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)

                g_loss.backward()
                self.optimG.step()


                if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch, d_loss, g_loss, real_score, fake_score)
                    #self.logger.draw(right_images, fake_images)

            #self.logger.plot_epoch_w_scores(iteration)

            if (epoch) % 50 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, epoch)


    def _train_began(self):
        print("Starting training for BEGAN")
        criterion = nn.L1Loss()
        lr = self.lr

        measure_history = []
        iteration = 0

        for epoch in range(self.num_epochs):
            for _, sample in enumerate(self.data_loader):
                iteration += 1
                image = sample['face']
                identity = sample['onehot']
                image = Variable(image.float()).cuda()
                #image = Variable(image.float())
                identity = Variable(identity.float()).cuda()
                #identity = Variable(identity.float())
                noise = Variable(torch.randn(image.size(0), 100)).cuda()
                #noise = Variable(torch.randn(image.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)

                #Train discriminator:
                self.discriminator.zero_grad()
                gen_z = self.generator(noise, identity)
                outputs_d_z = self.discriminator(gen_z.detach(), noise, identity)
                outputs_d_x = self.discriminator(image, noise,  identity)
                #real_loss_d = criterion(outputs_d_x, image)
                #fake_loss_d = criterion(outputs_d_z, gen_z.detach())
                real_loss_d = torch.mean(torch.abs(outputs_d_x - image))
                fake_loss_d = torch.mean(torch.abs(outputs_d_z - gen_z))
                lossD = real_loss_d + self.k *fake_loss_d
                lossD.backward()
                self.optimD.step()

                #Train Generator:
                self.generator.zero_grad()
                gen_z = self.generator(noise, identity)
                outputs_g_z = self.discriminator(gen_z, noise, identity)
                #lossG = criterion(outputs_g_z, gen_z)
                lossG = torch.mean(torch.abs(outputs_g_z - gen_z))
                lossG.backward()
                self.optimG.step()

                balance = (self.gamma * real_loss_d - fake_loss_d).data[0]
                self.k += self.lambda_k * balance
                self.k = max(min(1, self.k), 0)

                convg_measure = real_loss_d.data[0] + np.abs(balance)
                measure_history.append(convg_measure)

                if iteration % 5 == 0:
                    self.logger.log_iteration_began(iteration, epoch, lossD, lossG, real_loss_d, fake_loss_d, lr)

                if epoch % 10 == 0:
                    Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)

                #Update Learning rate:
                if self.lr_update_type == 1:
                    lr = self.lr * 0.95 ** (iteration // self.lr_update_step)
                elif self.lr_update_type == 2:
                    if iteration % self.lr_update_step == self.lr_update_step - 1:
                        lr *= 0.5
                elif self.lr_update_type == 3:
                    if iteration % self.lr_update_step == self.lr_update_step - 1:
                        lr = min(lr * 0.5, self.lr_lower_boundary)
                else:
                    if iteration % self.lr_update_step == self.lr_update_step - 1:
                        cur_measure = np.mean(measure_history)
                        if cur_measure > prev_measure * 0.9999:
                            lr = min(lr * 0.5, self.lr_lower_boundary)
                        prev_measure = cur_measure

                for p in self.optimG.param_groups + self.optimD.param_groups:
                    p['lr'] = lr


    def train_segan_and_lsgan(self):
        criterion = nn.MSELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0
        print('Starting train for GAN')
        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                if self.dataset_name != 'youtubers':

                    right_images = sample['right_images']
                    right_embed = sample['right_embed']
                    wrong_images = sample['wrong_images']
                    wrong_images = Variable(wrong_images.float()).cuda()
                    #wrong_images = Variable(wrong_images.float())
                else:
                    right_images = sample['face']
                    right_embed = sample['onehot']
                    raw_wav = sample['audio']
                right_images = Variable(right_images.float()).cuda()
                #right_images = Variable(right_images.float())
                raw_wav = Variable(raw_wav.float()).cuda()
                #raw_wav = Variable(raw_wav.float())
                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))
                auxiliary_noise = torch.randn(right_embed.size(0), 128)
                # right_embed = right_embed + auxiliary_noise
                auxiliary_noise = Variable(auxiliary_noise.float()).cuda()
                #auxiliary_noise = Variable(auxiliary_noise.float())
                right_embed = Variable(right_embed.float()).cuda()
                #right_embed = Variable(right_embed.float())

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).cuda()
                #real_labels = Variable(real_labels)
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                #smoothed_real_labels = Variable(smoothed_real_labels)
                fake_labels = Variable(fake_labels).cuda()
                #fake_labels = Variable(fake_labels)
                # Train the discriminator
                self.discriminator.zero_grad()

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                #noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images, projected_embed = self.generator(raw_wav, right_embed, noise, auxiliary_noise,
                                                              project=self.apply_projection)
                outputs, _ = self.discriminator(fake_images, projected_embed, project=self.apply_projection,
                                                concat=self.apply_concat)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs
                outputs, activation_real = self.discriminator(right_images, projected_embed,
                                                              project=self.apply_projection, concat=self.apply_concat)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs
                d_loss = real_loss + fake_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                #noise = Variable(torch.randn(right_images.size(0), 100))
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images, projected_embed = self.generator(raw_wav, right_embed, noise, auxiliary_noise,
                                                              project=self.apply_projection)
                outputs, activation_fake = self.discriminator(fake_images, projected_embed,
                                                              project=self.apply_projection, concat=self.apply_concat)
                _, activation_real = self.discriminator(right_images, projected_embed, project=self.apply_projection,
                                                        concat=self.apply_concat)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                # ======= Generator Loss function============
                # This is a customized loss function, the first term is the mean square error loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                # ===========================================
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)

                g_loss.backward()
                self.optimG.step()

                if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch, d_loss, g_loss, real_score, fake_score)
                    # self.logger.draw(right_images, fake_images)

                    # self.logger.plot_epoch_w_scores(epoch)

            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)

    def predict(self):
        print('Starting inference...')
        for id, sample in enumerate(self.data_loader):
            if self.dataset_name != 'youtubers':
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                txt = sample['txt']
            else:
                right_images = sample['face']
                right_embed = sample['onehot']
                raw_wav = sample['audio']
                paths = sample['audio_path']
                token = (right_embed == 1).nonzero()[:,1]
                txt = [self.dataset.youtubers[idx] + str(id) for idx in token]
                #txt = [path for path in paths]
                #print(txt)

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))
            raw_wav = Variable(raw_wav.float()).cuda()
            #raw_wav = Variable(raw_wav.float())
            right_images = Variable(right_images.float()).cuda()
            #right_images = Variable(right_images.float())
            right_embed = Variable(right_embed.float()).cuda()
            #right_embed = Variable(right_embed.float())
            auxiliary_noise = torch.randn(right_embed.size(0), 128)
            # right_embed = right_embed + auxiliary_noise
            auxiliary_noise = Variable(auxiliary_noise.float()).cuda()
            #auxiliary_noise = Variable(auxiliary_noise.float())
            # Train the generator
            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            #noise = Variable(torch.randn(right_images.size(0), 100))
            noise = noise.view(noise.size(0), 100, 1, 1)
            #self.generator.eval()
            fake_images, _ = self.generator(raw_wav, right_embed, noise, auxiliary_noise, project=self.apply_projection)
            #self.logger.draw(right_images, fake_images)

            for image, t in zip(fake_images, txt):
                #print(image.data.shape)
                im = image.data.mul_(127.5).add_(127.5).permute(1, 2, 0).cpu().numpy()
                rgb = np.empty((64, 64, 3), dtype=np.float32)
                rgb[:,:,0] = im[:,:,2]
                rgb[:,:,1] = im[:,:,1]
                rgb[:,:,2] = im[:,:,0]
                im = Image.fromarray(rgb.astype('uint8'))
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                print(t)







