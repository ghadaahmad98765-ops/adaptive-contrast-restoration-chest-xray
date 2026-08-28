%% PROJECT 1 - PHASE 2
% Adaptive Contrast Restoration for Low-Dose Chest Radiography
% Enhancement chain evaluated on Phase-1-style Poisson degradation.
% PNG files must be stored in ./Images and named Image (1).png ... Image (300).png

clear; clc; close all; rng(1);

%% SETTINGS
imageFolder = fullfile(pwd,'Images');
outputFolder = fullfile(pwd,'Phase2_Results');
figFolder = fullfile(outputFolder,'Figures');
if ~exist(outputFolder,'dir'), mkdir(outputFolder); end
if ~exist(figFolder,'dir'), mkdir(figFolder); end

files = dir(fullfile(imageFolder,'*.png'));
if isempty(files), error('No PNG images found in: %s',imageFolder); end
numImages = numel(files);
fprintf('Found %d PNG images.\n',numImages);

% Main development dose; final chain is verified at all three doses.
devDose = 0.25;
verifyDoses = [0.50 0.25 0.10];
N0 = 10000;

% Enhancement parameters
AHEgrids = [4 8 16];
CLAHEgrids = [4 8 16];
clipLimits = [0.005 0.01 0.02 0.04];
boxSizes = [3 5 7];
gaussSigmas = [0.5 1.0 1.5];
medianSizes = [3 5];
alphaTrimD = [2 4 6];       % 5x5 neighborhood; d must be even
boostFactors = [1.0 1.25 1.5 1.75 2.0];
repIDs = [1 150 300];

%% BUILD TARGET HISTOGRAM FROM FULL-DOSE IMAGES
avgHist = zeros(256,1);
for k=1:numImages
    I = readGrayDouble(fullfile(imageFolder,files(k).name));
    h = imhist(I,256); h = h/sum(h);
    avgHist = avgHist + h;
end
avgHist = avgHist/numImages;
targetCDF = cumsum(avgHist); targetCDF = targetCDF/targetCDF(end);
writematrix([(0:255)' avgHist targetCDF],fullfile(outputFolder,'TargetHistogram.csv'));

f=figure('Name','Target full-dose histogram','Visible','on');
bar(0:255,avgHist); xlabel('Gray level'); ylabel('Probability'); title('Average Full-Dose Target Histogram'); grid on;
exportgraphics(f,fullfile(figFolder,'TargetHistogram.png'),'Resolution',300);

%% RESULT TABLES
contrastResults = table(); smoothResults = table(); sharpResults = table(); finalResults = table();

%% PROCESS ALL IMAGES AT DEVELOPMENT DOSE
for k=1:numImages
    filename = files(k).name;
    I = readGrayDouble(fullfile(imageFolder,filename));
    noisy = simulatePoisson(I,N0,devDose);
    fprintf('Phase 2: %d/%d %s\n',k,numImages,filename);

    % Baseline
    contrastResults = [contrastResults; metricRow(filename,'Baseline',devDose,noisy,I,NaN,NaN)];

    % Global HE
    J = histeq(noisy,256);
    contrastResults = [contrastResults; metricRow(filename,'GlobalHE',devDose,J,I,NaN,NaN)];

    % Histogram matching to average full-dose target distribution
    J = customHistMatch(noisy,targetCDF);
    contrastResults = [contrastResults; metricRow(filename,'HistMatch',devDose,J,I,NaN,NaN)];

    % AHE grid sweep (unlimited local equalization)
    for g=AHEgrids
        J = customAHE(noisy,g);
        contrastResults = [contrastResults; metricRow(filename,sprintf('AHE_%dx%d',g,g),devDose,J,I,g,NaN)];
    end

    % Custom CLAHE grid + clip sweep
    for g=CLAHEgrids
        for c=clipLimits
            J = customCLAHE(noisy,g,c);
            contrastResults = [contrastResults; metricRow(filename,sprintf('CLAHE_%dx%d_C%.3f',g,g,c),devDose,J,I,g,c)];
        end
    end

    % Smoothing on Poisson noise
    for w=boxSizes
        J = imfilter(noisy,ones(w)/(w*w),'replicate');
        smoothResults = [smoothResults; metricRow(filename,sprintf('Box_%dx%d',w,w),devDose,J,I,w,NaN)];
    end
    for s=gaussSigmas
        J = imgaussfilt(noisy,s,'Padding','replicate');
        smoothResults = [smoothResults; metricRow(filename,sprintf('Gaussian_s%.1f',s),devDose,J,I,s,NaN)];
    end
    for w=medianSizes
        J = medfilt2(noisy,[w w],'symmetric');
        smoothResults = [smoothResults; metricRow(filename,sprintf('Median_%dx%d',w,w),devDose,J,I,w,NaN)];
    end
    J = adaptiveMedianFilter(noisy,7);
    smoothResults = [smoothResults; metricRow(filename,'AdaptiveMedian_max7',devDose,J,I,7,NaN)];
    for d=alphaTrimD
        J = alphaTrimmedMean(noisy,5,d);
        smoothResults = [smoothResults; metricRow(filename,sprintf('AlphaTrim_5x5_d%d',d),devDose,J,I,5,d)];
    end

    % Sharpening directly on degraded image for controlled comparison
    L = imfilter(noisy,[0 1 0;1 -4 1;0 1 0],'replicate');
    J = min(max(noisy-L,0),1);
    sharpResults = [sharpResults; metricRow(filename,'Laplacian',devDose,J,I,NaN,NaN)];

    blur = imgaussfilt(noisy,1,'Padding','replicate');
    mask = noisy-blur;
    J = min(max(noisy+mask,0),1);
    sharpResults = [sharpResults; metricRow(filename,'Unsharp_k1',devDose,J,I,1,NaN)];
    for b=boostFactors
        J = min(max(noisy+b*mask,0),1);
        sharpResults = [sharpResults; metricRow(filename,sprintf('HighBoost_k%.2f',b),devDose,J,I,b,NaN)];
    end
end

%% SUMMARIES (MEAN + STD)
contrastSummary = summarizeMetrics(contrastResults,'Method');
smoothSummary = summarizeMetrics(smoothResults,'Method');
sharpSummary = summarizeMetrics(sharpResults,'Method');
writetable(contrastResults,fullfile(outputFolder,'Contrast_AllImages.csv'));
writetable(contrastSummary,fullfile(outputFolder,'Contrast_Summary.csv'));
writetable(smoothResults,fullfile(outputFolder,'Smoothing_AllImages.csv'));
writetable(smoothSummary,fullfile(outputFolder,'Smoothing_Summary.csv'));
writetable(sharpResults,fullfile(outputFolder,'Sharpening_AllImages.csv'));
writetable(sharpSummary,fullfile(outputFolder,'Sharpening_Summary.csv'));

%% SELECT BEST COMPONENTS BY MEAN SSIM (objective, reproducible rule)
bestSmooth = bestMethodBySSIM(smoothSummary);
bestContrast = bestMethodBySSIM(contrastSummary,{'Baseline'});
bestSharp = bestMethodBySSIM(sharpSummary);
fprintf('\nSelected at 25%% dose by highest mean SSIM:\n  Smoothing: %s\n  Contrast: %s\n  Sharpening: %s\n',bestSmooth,bestContrast,bestSharp);

%% FINAL CHAIN: DENOISE -> CONTRAST -> SHARPEN; VERIFY 50/25/10%
for k=1:numImages
    filename=files(k).name; I=readGrayDouble(fullfile(imageFolder,filename));
    for dose=verifyDoses
        noisy=simulatePoisson(I,N0,dose);
        S=applyNamedMethod(noisy,bestSmooth,targetCDF);
        C=applyNamedMethod(S,bestContrast,targetCDF);
        F=applyNamedMethod(C,bestSharp,targetCDF);
        finalResults=[finalResults; metricRow(filename,'Baseline',dose,noisy,I,NaN,NaN)];
        finalResults=[finalResults; metricRow(filename,'FinalChain',dose,F,I,NaN,NaN)];
    end
end
finalSummary = groupsummary(finalResults,{'Dose_Percent','Method'},{'mean','std'}, ...
    {'MSE','RMSE','PSNR_dB','SSIM','SNR_dB','Entropy','ContrastStd','EdgeRetention_Percent'});
writetable(finalResults,fullfile(outputFolder,'FinalChain_AllImages.csv'));
writetable(finalSummary,fullfile(outputFolder,'FinalChain_Summary.csv'));

%% REPRESENTATIVE FIGURES: IMAGES 1, 150, 300
for id=repIDs
    idx=find(strcmp({files.name},sprintf('Image (%d).png',id)),1);
    if isempty(idx), warning('Representative Image (%d).png not found.',id); continue; end
    I=readGrayDouble(fullfile(imageFolder,files(idx).name)); noisy=simulatePoisson(I,N0,devDose);
    S=applyNamedMethod(noisy,bestSmooth,targetCDF); C=applyNamedMethod(S,bestContrast,targetCDF); F=applyNamedMethod(C,bestSharp,targetCDF);

    f=figure('Name',sprintf('Phase 2 Final Chain Image %d',id),'Visible','on');
    tiledlayout(1,150,'Padding','compact','TileSpacing','compact');
    nexttile; imshow(I,[]); title('Original');
    nexttile; imshow(noisy,[]); title('25% Poisson');
    nexttile; imshow(S,[]); title(['Denoise: ' strrep(bestSmooth,'_',' ')]);
    nexttile; imshow(C,[]); title(['Contrast: ' strrep(bestContrast,'_',' ')]);
    nexttile; imshow(F,[]); title(['Final: ' strrep(bestSharp,'_',' ')]);
    sgtitle(sprintf('Phase 2 Enhancement Chain - Image (%d)',id));
    exportgraphics(f,fullfile(figFolder,sprintf('FinalChain_Image_%d.png',id)),'Resolution',300);
end

fprintf('\nPHASE 2 COMPLETE. Results saved in: %s\n',outputFolder);
disp(finalSummary);

%% ================= LOCAL FUNCTIONS =================
function I=readGrayDouble(path)
Iraw=imread(path); if ndims(Iraw)==3, Iraw=rgb2gray(Iraw); end; I=im2double(Iraw);
end

function J=simulatePoisson(I,N0,dose)
lambda=max(I*N0*dose,0); K=poissrnd(lambda); J=K/(N0*dose); J=min(max(J,0),1);
end

function T=metricRow(filename,method,dose,J,I,p1,p2)
m=immse(J,I); rm=sqrt(m); p=psnr(J,I); s=ssim(J,I); noise=J-I;
sigP=mean(I(:).^2); noiP=mean(noise(:).^2); snrdb=10*log10(sigP/(noiP+eps));
e=entropy(J); cs=std(J(:));
[go,~]=imgradient(I); [gj,~]=imgradient(J); er=100*sum(gj(:).^2)/(sum(go(:).^2)+eps);
T=table(string(filename),string(method),dose*100,p1,p2,m,rm,p,s,snrdb,e,cs,er, ...
'VariableNames',{'Image','Method','Dose_Percent','Param1','Param2','MSE','RMSE','PSNR_dB','SSIM','SNR_dB','Entropy','ContrastStd','EdgeRetention_Percent'});
end

function S=summarizeMetrics(T,groupVar)
S=groupsummary(T,groupVar,{'mean','std'},{'MSE','RMSE','PSNR_dB','SSIM','SNR_dB','Entropy','ContrastStd','EdgeRetention_Percent'});
end

function name=bestMethodBySSIM(S,exclude)
if nargin<2, exclude={}; end
keep=true(height(S),1); for i=1:numel(exclude), keep=keep & S.Method~=string(exclude{i}); end
SS=S(keep,:); [~,ix]=max(SS.mean_SSIM); name=char(SS.Method(ix));
end

function J=customHistMatch(I,targetCDF)
h=imhist(I,256); c=cumsum(h)/sum(h); map=zeros(256,1);
for a=1:256, [~,j]=min(abs(targetCDF-c(a))); map(a)=j-1; end
idx=floor(I*255)+1; J=reshape(map(idx),size(I))/255;
end

function J=customAHE(I,g)
J=tileMap(I,g,Inf);
end

function J=customCLAHE(I,g,clipLimit)
J=tileMap(I,g,clipLimit);
end

function J=tileMap(I,g,clipLimit)
% Custom tile histogram equalization. CLAHE explicitly clips each tile
% histogram and redistributes excess counts uniformly before forming CDF.
[H,W]=size(I); maps=zeros(256,g,g); yc=round(linspace(1,H+1,g+1)); xc=round(linspace(1,W+1,g+1));
for r=1:g
 for c=1:g
  tile=I(yc(r):yc(r+1)-1,xc(c):xc(c+1)-1); h=imhist(tile,256);
  if isfinite(clipLimit)
   thr=max(1,floor(clipLimit*numel(tile))); excess=sum(max(h-thr,0)); h=min(h,thr);
   q=floor(excess/256); h=h+q; remn=excess-q*256;
   if remn>0, pos=round(linspace(1,256,remn)); for z=1:numel(pos), h(pos(z))=h(pos(z))+1; end, end
  end
  cdf=cumsum(h)/sum(h); maps(:,r,c)=cdf;
 end
end
% Bilinear interpolation among neighboring tile mappings at every pixel.
cy=((yc(1:end-1)+yc(2:end)-1)/2); cx=((xc(1:end-1)+xc(2:end)-1)/2); J=zeros(H,W);
for y=1:H
 [r1,r2,wy]=neighbors(cy,y);
 for x=1:W
  [c1,c2,wx]=neighbors(cx,x); b=floor(I(y,x)*255)+1;
  v11=maps(b,r1,c1); v12=maps(b,r1,c2); v21=maps(b,r2,c1); v22=maps(b,r2,c2);
  J(y,x)=(1-wy)*((1-wx)*v11+wx*v12)+wy*((1-wx)*v21+wx*v22);
 end
end
J=min(max(J,0),1);
end

function [i1,i2,w]=neighbors(c,v)
if v<=c(1), i1=1;i2=1;w=0; elseif v>=c(end), i1=numel(c);i2=i1;w=0; else
 i2=find(c>=v,1); i1=i2-1; w=(v-c(i1))/(c(i2)-c(i1)); end
end

function J=adaptiveMedianFilter(I,Smax)
% Classical adaptive median filter (3x3 -> Smax). Included for requested comparison.
pad=floor(Smax/2); P=padarray(I,[pad pad],'symmetric'); J=I; [H,W]=size(I);
for y=1:H
 for x=1:W
  zxy=I(y,x); out=zxy;
  for s=3:2:Smax
   h=floor(s/2); cy=y+pad; cx=x+pad; win=P(cy-h:cy+h,cx-h:cx+h); zmin=min(win(:)); zmax=max(win(:)); zmed=median(win(:));
   if zmed>zmin && zmed<zmax
    if zxy>zmin && zxy<zmax, out=zxy; else, out=zmed; end; break;
   elseif s==Smax, out=zmed; end
  end
  J(y,x)=out;
 end
end
end

function J=alphaTrimmedMean(I,w,d)
if mod(d,2)~=0 || d>=w*w, error('d must be even and smaller than window area'); end
p=floor(w/2); P=padarray(I,[p p],'symmetric'); J=zeros(size(I)); [H,W]=size(I); trim=d/2;
for y=1:H
 for x=1:W
  win=P(y:y+2*p,x:x+2*p); v=sort(win(:)); v=v(trim+1:end-trim); J(y,x)=mean(v);
 end
end
end

function J=applyNamedMethod(I,name,targetCDF)
if strcmp(name,'Baseline'), J=I;
elseif strcmp(name,'GlobalHE'), J=histeq(I,256);
elseif strcmp(name,'HistMatch'), J=customHistMatch(I,targetCDF);
elseif startsWith(name,'AHE_'), g=sscanf(name,'AHE_%dx%d'); J=customAHE(I,g(1));
elseif startsWith(name,'CLAHE_'), a=regexp(name,'CLAHE_(\d+)x\d+_C([0-9.]+)','tokens','once'); J=customCLAHE(I,str2double(a{1}),str2double(a{2}));
elseif startsWith(name,'Box_'), a=sscanf(name,'Box_%dx%d'); w=a(1); J=imfilter(I,ones(w)/(w*w),'replicate');
elseif startsWith(name,'Gaussian_'), a=regexp(name,'Gaussian_s([0-9.]+)','tokens','once'); J=imgaussfilt(I,str2double(a{1}),'Padding','replicate');
elseif startsWith(name,'Median_'), a=sscanf(name,'Median_%dx%d'); J=medfilt2(I,[a(1) a(1)],'symmetric');
elseif strcmp(name,'AdaptiveMedian_max7'), J=adaptiveMedianFilter(I,7);
elseif startsWith(name,'AlphaTrim_'), a=regexp(name,'AlphaTrim_5x5_d(\d+)','tokens','once'); J=alphaTrimmedMean(I,5,str2double(a{1}));
elseif strcmp(name,'Laplacian'), L=imfilter(I,[0 1 0;1 -4 1;0 1 0],'replicate'); J=min(max(I-L,0),1);
elseif strcmp(name,'Unsharp_k1'), B=imgaussfilt(I,1,'Padding','replicate'); J=min(max(I+(I-B),0),1);
elseif startsWith(name,'HighBoost_'), a=regexp(name,'HighBoost_k([0-9.]+)','tokens','once'); b=str2double(a{1}); B=imgaussfilt(I,1,'Padding','replicate'); J=min(max(I+b*(I-B),0),1);
else, error('Unknown method: %s',name); end
end
