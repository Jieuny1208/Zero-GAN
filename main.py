import json
import time
import datetime
import shutil
import os

from torch.optim import Adam
import torch.nn.functional as F

from models.zero_gan import ZeroGanGenerator, Discriminator, ImageToInput
from utils.data_utils import *
from utils.visualizer import Visualizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train():
    # 결과 경로 설정
    time_now=time.time()
    formatted_time = datetime.datetime.fromtimestamp(time_now).strftime('%Y_%m_%d_%H_%M')
    result_path = 'result/train_' + formatted_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    visualizer = Visualizer(path=result_path + '/images')
    
    checkpoint = None
    start_epoch = 1 # 이어서 학습 할 epoch, 처음부터 할거면 1
    # result 폴더의 경로 그대로 입력
    # checkpoint = torch.load(f'result/train_2024_05_26_20_30/ZeroGan_Adam_state_epoch_{start_epoch}.pt') # 이어서 학습할거면 start_epoch 설정하고 주석해제

    opt = None
    with open ("config.json", "r", encoding='utf-8') as f:
        opt = json.load(f)
    # 현재 train 설정 저장
    shutil.copy("config.json", result_path + '/config.json')

    dataloader = load_data(categories=opt['categories'],
                            image_size=(opt['image_size'], opt['image_size']),
                            train_batch_size=opt['train_batch_size'],
                            num_workers=0, ratio=opt['ratio'],
                            mode='train'
                            )

    # 이미지를 patch 할 때 사용
    patch_vector_size = (opt['patch_size']**2) * opt['in_channels'] 
    
    model = ZeroGanGenerator(opt)
    discriminator = Discriminator(opt)
    image_to_input = ImageToInput(opt)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state'])
        discriminator.load_state_dict(checkpoint['d_model_state'])
        image_to_input.load_state_dict(checkpoint['image_to_input_state'])

    model.to(device)
    discriminator.to(device)
    image_to_input.to(device)

    optimizer = Adam(model.parameters(), lr=opt['lr'])
    d_optimizer = Adam(discriminator.parameters(), lr=opt['lr']) # 독립적으로 lr 가져야 할듯

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])

        
    # loss_function = nn.MSELoss().to(device)
    loss_function = nn.L1Loss().to(device)
    d_loss_function = nn.BCELoss().to(device)

    model.train()
    discriminator.train()

    num_epoch = opt['epoch']
    losses=[]
    d_losses=[]
    if checkpoint is not None:
        losses = checkpoint['losses']
    for epoch in range(start_epoch, num_epoch+1):
        total_loss = 0
        d_total_loss = 0
        original_sample_img=None
        reconstructed_sample_img=None
        for i, batch in enumerate(dataloader):
            input= batch['image'].to(device)
            labels = batch['label'].float().to(device) # label: 0=normal, 1=abnormal (아마도)

            # real/fake에 대한 label, 위의 normal/abnormal label이랑은 별개
            real_labels = torch.ones(input.shape[0]).to(device)
            fake_labels = torch.zeros(input.shape[0]).to(device)

            # input_classifier (latent_vector_dim)
            # input_features (batch_size, ?, grid_size, grid_size)
            input_classifier, input_features = discriminator(input)
            d_loss_real = d_loss_function(input_classifier, real_labels)

            loss_accum = 0
            mask_iter = opt['mask_iterations']
            gen_img_accum = torch.zeros(input.shape[0],opt['in_channels'],opt['image_size'],opt['image_size']).to(device)
            for idx in range(mask_iter):
                embeddings, _ = image_to_input(input)
                # latent_i : 입력 이미지의 embedding? / shape:(batch_size, (image_size/patch_size)**2, embed_dim)
                # gen_img : 재생성된 이미지 / shape:(batch_size, in_channels, image_size, image_size)
                # latent_o : 재생성된 이미지(gen_img)의 latent vector / shape:(batch_size, latent_vector_dim, ?, ?)
                latent_i, gen_img, latent_o = model(embeddings)
                gen_img_accum += gen_img
                # loss = loss_function(gen_img, input)
                # loss_accum += loss

            gen_classifier, gen_features = discriminator(gen_img_accum / mask_iter)
            d_loss_fake = d_loss_function(gen_classifier, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()


            # gen_classifier_detached, _ = discriminator(gen_img_accum.detach() / mask_iter)
            # g_loss = loss_accum / mask_iter
            Lcon = loss_function(gen_img_accum/mask_iter, input)
            # Ladv = 0.01*d_loss_function(gen_classifier_detached, real_labels)
            # g_loss = Lcon + Ladv
            g_loss = Lcon
            
            optimizer.zero_grad()
            g_loss.backward()
            optimizer.step()
            

            print("i : " + str(i) , ", generator loss(Lcon): ", f'{Lcon}', ', discriminator loss: ', f"{d_loss.item()}")
            total_loss += g_loss.item()
            d_total_loss += d_loss.item()
            
            if i == 0:  # 첫 번째 배치에서만 이미지 캡처
                original_sample_img = input[0].cpu()
                reconstructed_sample_img = (gen_img_accum / mask_iter)[0].cpu()

        avg_loss = total_loss / (i + 1)
        avg_d_loss = d_total_loss / (i + 1)
        losses.append(avg_loss)
        d_losses.append(avg_d_loss)
        
        # 이미지를 저장할 epoch 주기 설정
        if epoch % 1 == 0:
            visualizer.save_image(original_sample_img, f"{epoch}_original_img_0")
            visualizer.save_image(reconstructed_sample_img, f"{epoch}_reconstructed_img_0")
        
        # 모델 및 설정 저장
        # .pt파일 크기가 1기가 이상 -> 주기 최대한 길게
        if epoch % 30 == 0:
            filename = f'ZeroGan_{optimizer.__class__.__name__}_state_epoch_{epoch}.pt'
            filepath = result_path +'/'+ filename
            torch.save({
                'model_state': model.state_dict(),
                'd_model_state': discriminator.state_dict(),
                'image_to_input_state': image_to_input.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'd_optimizer_state': d_optimizer.state_dict(),
                'losses' : losses,
                'd_losses' : d_losses
            }, filepath)
            """"""
        print(f"Epoch : {epoch}, Loss : {avg_loss}")


def test():
    # 테스트 결과를 저장할 경로 설정
    # 모델 불러올 경로와는 다름
    time_now=time.time()
    formatted_time = datetime.datetime.fromtimestamp(time_now).strftime('%Y_%m_%d_%H_%M')
    result_path = 'result/test_' + formatted_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    visualizer = Visualizer(path=result_path + '/images')


    threshold = 0.5
    # 경로와 모델 이름 설정
    model_path = 'result/train_2024_05_30_00_22/'
    model_name = 'ZeroGan_Adam_state_epoch_90.pt'

    opt = None
    with open (model_path + "config.json", "r", encoding='utf-8') as f:
        opt = json.load(f)

    dataloader = load_data(categories=opt['categories'],
                            image_size=(opt['image_size'], opt['image_size']),
                            train_batch_size=1,
                            num_workers=0, ratio=1,
                            mode='test'
                            )


    model = ZeroGanGenerator(opt)
    discriminator = Discriminator(opt)
    image_to_input = ImageToInput(opt)

    checkpoint = torch.load(model_path + model_name)
    model.load_state_dict(checkpoint['model_state'])
    discriminator.load_state_dict(checkpoint['d_model_state'])
    image_to_input.load_state_dict(checkpoint['image_to_input_state'])

    model.to(device)
    discriminator.to(device)
    image_to_input.to(device)

    loss_function = nn.L1Loss().to(device)
    d_loss_function = nn.BCELoss().to(device)

    model.eval()
    discriminator.eval()
    image_to_input.eval()

    
    total_loss = 0
    d_total_loss = 0
    with torch.no_grad():
        total_loss = 0
        for i, batch in enumerate(dataloader):
            input = batch['image'].to(device)
            labels = batch['label'].to(device) # normal=0, abnormal=1

            real_labels = torch.ones(input.shape[0]).to(device)
            fake_labels = torch.zeros(input.shape[0]).to(device)

            input_classifier, input_features = discriminator(input)
            d_loss_real = d_loss_function(input_classifier, real_labels)

            loss_accum = 0
            mask_iter = opt['mask_iterations']
            gen_img_accum = torch.zeros(input.shape[0], opt['in_channels'], opt['image_size'], opt['image_size']).to(device)
            for idx in range(mask_iter):
                embeddings, _ = image_to_input(input)
                latent_i, gen_img, latent_o = model(embeddings)
                gen_img_accum += gen_img

            gen_classifier, gen_features = discriminator(gen_img_accum / mask_iter)
            d_loss_fake = d_loss_function(gen_classifier, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            Lcon = loss_function(gen_img_accum / mask_iter, input)
            g_loss = Lcon

            total_loss += g_loss.item()
            d_total_loss += d_loss.item()
            
            # for idx in range(len(input))

            original_img = input[0]
            generated_img = gen_img_accum[0] / mask_iter


            """
            원본 이미지와 재생성된 이미지 사이의 diff의 embedding
            
            Shape:
            (batchsize, num_patches, embed_dim) -> (batchsize, num_patches * embed_dim)
            
            class_mean_embeddings = [] # 각 클래스별 평균 임베딩 값
            class_mean_embeddings에서 거리가 일정 threshold값보다 작으면 해당 클래스로 분류(+ mean값 갱신)
            가까운 embedding이 없으면 class_mean_embeddings.append
            """
            diff_map = input - gen_img_accum / mask_iter
            diff_map_embedding, _ = image_to_input(diff_map)
            diff_map_embedding = diff_map_embedding.view(1,-1)
            
                
            anomaly_score = loss_function(generated_img, original_img)
            print('anomaly_score : ' ,anomaly_score.item() , ', label : ' , labels[0].item()) # normal=0, abnormal=1
            
            original_img = original_img.cpu()
            generated_img = generated_img.cpu()
            visualizer.save_image(original_img, f'{i}_{0}_original_img')
            visualizer.save_image(generated_img, f'{i}_{0}_generated_img')
                
        avg_loss = total_loss / (i + 1)
        avg_d_loss = d_total_loss / (i + 1)
    print(f"Test Loss: {avg_loss}, Discriminator Loss: {avg_d_loss}")

    report_path = os.path.join(result_path, 'test_report.txt')
    with open(report_path, 'w') as file:
        file.write(f"Test Losses: {avg_loss}\nDiscriminator Loss: {avg_d_loss}")
        print("Test results saved to", report_path)


if __name__ == "__main__":
    # train()

    test()

