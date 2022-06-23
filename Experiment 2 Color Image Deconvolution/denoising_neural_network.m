%  Image denoising using deep neural network

%Load the pretrained DnCNN network.

net = denoisingNetwork("dncnn");

for j=1:3
    %Read images
    I=im2double(imread(['source_image_',num2str(j),'.jpg']));
    C=im2double(imread(['observed_image_',num2str(j),'.jpg']));

    %Divide in three color channels
    [C_R,C_G,C_B] = imsplit(C);   
    
    %Use the DnCNN network to remove noise from each color channel
    
    denoisedR = denoiseImage(C_R,net);
    denoisedG = denoiseImage(C_G,net);
    denoisedB = denoiseImage(C_B,net);
    
    %Filter image with Noise
    Ie=cat(3,denoisedR,denoisedG,denoisedB);

    
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