# Zero-Gan

## 진행상황

* 기본적인 모델 구조 구현(auto encoder는 완료)
* 데이터 불러오기
    * btad(완료)
* loss함수에서 어떤 값을 쓸지, 어떤 loss함수를 쓸 지 정하기
* 어떤 discriminator를 사용헐지, 입출력 유형(구체적으로), 학습 과정 정의(generator와 독립적으로 등등..)
* backward() 구현
* train, test 스크립트 완성
    * 일정 epoch마다 모델 weights 저장(.pt?)
    * epoch에 따른 loss 값 등 저장(npz)
* 각 parameter의 역할 아키텍쳐에서 그림으로 나타내기
* 

## Parameters
    ---common configs---
    네트워크, 모델과 관계 없이 공통적으로 사용되는 parameter들입니다.
    epoch : 총 train 데이터에 대해 학습을 반복할 횟수 
    lr : learning rate
    
    ---data---
    데이터 관련 parameter들입니다.
    category : 00, 01, 02중에서 선택
    image_size : 이미지(정사각형)의 가로 세로 사이즈(pixel)
    train_batch_size : train 데이터셋에서의 batch size
    eval_batch_size : evaluation 데이터셋에서의 batch size
    num_workers : 아마도 사용할 cpu 코어수?

    ---mae encoder, decoder---
    MAE의 encoder와 decoder에서 공통적으로 사용되는 paramter들입니다.
    embed_dim : encoder의 입력, 출력에서 embedding의 dimension
    patch_size : 한 patch(정사각형)의 가로세로 사이즈 ex)image_size = 256 이고, patch_size = 64라면, patch의 총 개수 = 16 (256/64 * 256/64)
    mlp_ratio : 모르겠음..

    ---mae encoder---
    MAE의 encoder에서만 사용되는 paramter들입니다.
    encoder_depth : encoder의 black 개수, 값이 클수록 네트워크가 복잡해집니다(아마도)
    encoder_num_heads : 뭔진 모르겠지만 값이 커질수록 네트워크가 복잡해집니다(아마도)

    ---mae decoder---
    MAE의 decoder에서만 사용되는 paramter들입니다.
    decoder_embed_dim : decoder의 입력에서 embed_dim -> decoder_embed_dim로 변환됩니다.
    decoder_depth : decoder의 black 개수, 값이 클수록 네트워크가 복잡해집니다(아마도)
    decoder_num_heads : 뭔진 모르겠지만 값이 커질수록 네트워크가 복잡해집니다(아마도)
    in_channels : 입력 채널의 수, 아마 3에서 바꿀 일 없을거같습니다

    ---gan encoder---
    GAN의 encoder에서만 사용되는 paramter들입니다.
    latent_vector_dim : gan encoder의 출력인 latent vector의 차원 수
    discriminator_feature_num : 아직 discriminator가 정해지지 않아서 보류
    n_extra_layers : gan encoder에 추가할 conv layer 수 
    add_final_conv : 마지막에 conv하나 추가하는 옵션인데 어떤 의미가 있는진 모르겠습니다
