clc; clear; close all

%% Create Tensors X and Y from Database

text='database_audio\blue_throated_toucanet  (';

samples=29; size_sample=11026; size_tube=10;

X=zeros(size_sample,1,size_tube); Y=zeros(size_sample,1,size_tube);

fil=1; col=1; fond=1; cond=false; cont=0;

sigma=0.15;

for k=1:samples
    
    [yaux,Fs]=audioread([text num2str(k) ').mp3']);
    if Fs==44100 %Use audios with sample rate of 44100 Hz.
        y=mean(yaux,2);
        m=length(y);        
        j=1;
        Xaux=[];
        Yaux=[];
        while true
            if size_sample*j<=m
                aux_v1=y(size_sample*(j-1)+1:size_sample*j); %Create Free-Noise Samples
                n=randn(size(aux_v1)); aux_v2=aux_v1+sigma^2*n; %Create Nosiy Samples
                Xaux=[Xaux aux_v1(:)];
                Yaux=[Yaux aux_v2(:)];
                j=j+1;
            else
                break
            end
        end

        %Save Data in Tensors X and Y
        for r=1:size(Xaux,2)
            X(:,col,fond)=Xaux(:,r); Y(:,col,fond)=Yaux(:,r);
            fond=fond+1;        
            if fond>size_tube
                display(col)
                col=col+1;
                if col==1001
                    cond=true;
                    break
                end 
                fond=1;
            end
        end   
    
        if cond
            break
        end
    else
        cont=cont+1;
    end
end

%Remove last column
X=X(:,1:end-1,:);
Y=Y(:,1:end-1,:);


%% Show Audio Samples of Database

numSamples=randi([1 343],[1 2]); Fs=44100;

%Figure 1: Audio Free Noise

figure 

for j=1:2 
    auxX=[]; auxY=[];
    for k=1:10
        auxX=[auxX;X(:,numSamples(j),k)];
        auxY=[auxY;Y(:,numSamples(j),k)];
    end
    m1=length(auxX);
    totalTime1=m1./Fs;
    jumps1=totalTime1/m1;
    lengthTime1=totalTime1-jumps1;
    t1=0:jumps1:lengthTime1;
    subplot(4,1,j)
    plot(t1,auxX,'b')
    xlabel('Time','FontSize',9,'FontWeight','bold')
    ylabel('Amplitude','FontSize',9,'FontWeight','bold')
    xlim([0 2.5]) 
    title(['Sample #', num2str(numSamples(j)), ' from Tensor X'])
    
    subplot(4,1,j+2)
    plot(t1,auxY,'r')
    xlabel('Time','FontSize',9,'FontWeight','bold')
    ylabel('Amplitude','FontSize',9,'FontWeight','bold')
    xlim([0 2.5]) 
    title(['Sample #', num2str(numSamples(j)), ' from Tensor Y'])
end

%% Create Tensor Filter F

F=least_square_tensor(X,Y);

%% Denoising Audio

%Load Source and Noisy Audio (Tensor Representation)
load('source_audio.mat','orig_sig');
load('noisy_audio.mat','noise_sig');

%Clean Audio with Filter F (Tensor Representation)
clean_sig=tprod(F,noise_sig);


%Show source audio, noisy audio and cleaned auido
audio_org=[]; audio_noise=[]; audio_clean=[];

for k=1:10
    audio_org=[audio_org; orig_sig(:,:,k)];
    audio_noise=[audio_noise; noise_sig(:,:,k)];
    audio_clean=[audio_clean; clean_sig(:,:,k)];
end

m2=length(audio_org); 
totalTime2=m2./Fs;
jumps2=totalTime2/m2;
lengthTime2=totalTime2-jumps2;
t2=0:jumps2:lengthTime2;

figure

%Plot Source Audio
subplot(3,1,1)
plot(t2,audio_org,'b')
xlabel('Time','FontSize',9,'FontWeight','bold')
ylabel('Amplitude','FontSize',9,'FontWeight','bold')
xlim([0 2.5]) 
title('Source audio')

%Plot Nosiy Audio
subplot(3,1,2)
plot(t2,audio_noise,'r')
xlabel('Time','FontSize',9,'FontWeight','bold')
ylabel('Amplitude','FontSize',9,'FontWeight','bold')
xlim([0 2.5]) 
title('Noisy Recording Audio')

%Plot Estimated Audio
subplot(3,1,3)
plot(t2,audio_clean,'g')
xlabel('Time','FontSize',9,'FontWeight','bold')
ylabel('Amplitude','FontSize',9,'FontWeight','bold')
xlim([0 2.5]) 
title('Estimation Using Algorithm 1')


%Save audio
audiowrite('source_audio.wav',audio_org,Fs)
audiowrite('noisy_audio.wav',audio_noise,Fs)
audiowrite('estimation_audio.wav',audio_clean,Fs)

%% Relative Error Estimation
error=norm(audio_org-audio_clean)/norm(audio_org);
display(['Relative error of estimation: = ', num2str(error)])