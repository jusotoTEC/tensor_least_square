%Experiment 2 of conference paper submitted to TSP 2022. (https://tsp.vutbr.cz/)
%Paper: A Least-Squares Problem of a Linear Tensor Equation of Third-Order for Audio and Color Image Processing
%Author: Pablo Soto-Quiros (https://www.tec.ac.cr/juan-pablo-soto-quiros)
%Institute: Costa Rica Institute Technology
%Email: jusoto@tec.ac.cr

clc; clear; close all

%% Create Tensor X and Y

d=128; %Dimension of each sample image

X=zeros(d^2,1078,3); Y=zeros(d^2,1078,3);
sigma=0.3;
for k=1:1078
    text=['database_images\land_images (' num2str(k) ').jpg'];
    Xk=im2double(imread(text));
    h = fspecial('motion', 5, 5);
    Yk = imfilter(Xk, h)+sigma^2*randn(size(Xk)); %Simulate Noise
    
    % Add Color Image into Tensor X (Data Free-Noise)
    Xk_r=Xk(:,:,1); vXk_r=Xk_r(:);
    Xk_g=Xk(:,:,2); vXk_g=Xk_g(:);
    Xk_b=Xk(:,:,3); vXk_b=Xk_b(:);            
    X(:,k,1)=vXk_r; X(:,k,2)=vXk_g; X(:,k,3)=vXk_b;
    
    % Add Color Image into Tensor Y (Nosiy Data)
    Yk_r=Yk(:,:,1); vYk_r=Yk_r(:);
    Yk_g=Yk(:,:,2); vYk_g=Yk_g(:);
    Yk_b=Yk(:,:,3); vYk_b=Yk_b(:);            
    Y(:,k,1)=vYk_r; Y(:,k,2)=vYk_g; Y(:,k,3)=vYk_b;            
end

%% Samples Images From X and Y

num_samples=sort(randi([1 1078],[3 1]));

figure

for k=1:3
   subplot(2,3,k)   
   AuxXk_r=reshape(X(:,num_samples(k),1),[d d]);
   AuxXk_g=reshape(X(:,num_samples(k),2),[d d]);
   AuxXk_b=reshape(X(:,num_samples(k),3),[d d]);   
   AuxXk=zeros(d,d,3);
   AuxXk(:,:,1)=AuxXk_r;
   AuxXk(:,:,2)=AuxXk_g;
   AuxXk(:,:,3)=AuxXk_b;
   imshow(AuxXk)
   title(['Sample #', num2str(num_samples(k))])
   
   subplot(2,3,k+3)
   AuxYk_r=reshape(Y(:,num_samples(k),1),[d d]);
   AuxYk_g=reshape(Y(:,num_samples(k),2),[d d]);
   AuxYk_b=reshape(Y(:,num_samples(k),3),[d d]);   
   AuxYk=zeros(d,d,3);
   AuxYk(:,:,1)=AuxYk_r;
   AuxYk(:,:,2)=AuxYk_g;
   AuxYk(:,:,3)=AuxYk_b;
   imshow(AuxYk)
   title(['Sample #', num2str(num_samples(k))])    
end

%% Compute Tensor Filter
F=least_square_tensor(X,Y);

%% Denoising Image with Filter F
%Three options to denoising images
%1) Source images: source_image_1.jpg, source_image_2.jpg and 
%                  source_image_3.jpg 
%2) Observed images: observed_image_1.jpg, observed_image_2.jpg and 
%                  observed_image_3.jpg

figure

for j=1:3
    %Read images
    I=im2double(imread(['source_image_',num2str(j),'.jpg']));
    C=im2double(imread(['observed_image_',num2str(j),'.jpg']));

    % Tensor representation of images
    [m,n,p]=size(I);
    X=zeros(m*n,1,p);
    Y=zeros(m*n,1,p);

    for k=1:n
        X(m*(k-1)+1:m*k,1,:)=I(:,k,:);
        Y(m*(k-1)+1:m*k,1,:)=C(:,k,:);
    end

    %Filter image with Noise
    Xe=im2uint8(tprod(F,Y));

    %Tensor to image
    Ie=uint8(zeros(m,n,p));

    for k=1:n
        Ie(:,k,:)=Xe(m*(k-1)+1:m*k,1,:);
    end

    %Plot images

    subplot(3,3,3*(j-1)+1)
    imshow(I)
    title(['Source Image #',num2str(j)])
    subplot(3,3,3*(j-1)+2)
    imshow(C)
    title(['Noisy Observed Image #',num2str(j)])
    subplot(3,3,3*(j-1)+3)
    imshow(Ie)
    title(['Estimation Image #',num2str(j)])

    % Relative Error Estimation
    error=frob_norm_3D(im2double(I)-im2double(Ie))/frob_norm_3D(im2double(I));
    display(['Relative error of estimation #',num2str(j),': = ', num2str(error)])
end