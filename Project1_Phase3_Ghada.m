%% PROJECT 1 - PHASE 3
% Adaptive Contrast Restoration for Low-Dose Chest Radiography
% Evaluation phase for the Phase-2 enhancement chain.
%
% Designed for the full working subset of 300 PNG images:
%   Images/Image (1).png ... Images/Image (300).png
% Representative images: 1, 150, 300
%
% IMPORTANT:
% 1) ROI placement is fully automatic. Standardized proportional ROIs are used
%    for left lung, right lung, and mediastinum on representative images.
%    This avoids any manual interaction, but it is a methodological deviation
%    from a brief that explicitly requests manually placed ROIs.
% 2) Core enhancement methods required by the brief are implemented from
%    scratch in local functions below. Library equivalents are used ONLY in
%    the optional correctness-check appendix.

clear; clc; close all; rng(3);

%% ========================================================================
% SETTINGS
% =========================================================================
imageFolder = fullfile(pwd,'Images');
outputFolder = fullfile(pwd,'Phase3_Results');
figFolder = fullfile(outputFolder,'Figures');
assessmentFolder = fullfile(outputFolder,'Visual_Assessment_Pairs');

if ~exist(outputFolder,'dir'), mkdir(outputFolder); end
if ~exist(figFolder,'dir'), mkdir(figFolder); end
if ~exist(assessmentFolder,'dir'), mkdir(assessmentFolder); end

files = dir(fullfile(imageFolder,'*.png'));
if isempty(files)
    error('No PNG images found in: %s',imageFolder);
end

% Natural sort by Image (N).png number.
ids = zeros(numel(files),1);
for k=1:numel(files)
    tok = regexp(files(k).name,'Image \((\d+)\)\.png','tokens','once');
    if isempty(tok), ids(k)=inf; else, ids(k)=str2double(tok{1}); end
end
[~,ord] = sort(ids);
files = files(ord);
ids = ids(ord);

% Use the full numbered working subset: Image (1) through Image (300).
keep = isfinite(ids) & ids>=1 & ids<=300;
files = files(keep);
ids = ids(keep);
numImages = numel(files);
if numImages==0
    error('Expected PNG files named Image (1).png ... Image (300).png');
end
fprintf('Phase 3: found %d working images.\n',numImages);

repIDs = [1 150 300];
verifyDoses = [0.50 0.25 0.10];
N0 = 10000;

% Phase-2 selected chain from your results:
% Gaussian sigma 1.0 -> Histogram Matching -> High-Boost k=1.00
GAUSS_SIGMA = 1.0;
HIGHBOOST_K = 1.00;

% CLAHE sensitivity study at 25% dose, 8x8 tiles.
CLAHE_GRID = 8;
clipSweep = [0.0025 0.005 0.01 0.02 0.04 0.08 0.12];
claheDose = 0.25;

% EME parameters.
EME_BLOCKS = [8 8];

% Stretch objective.
RUN_BONUS = true;

%% ========================================================================
% BUILD FULL-DOSE TARGET HISTOGRAM FROM THE 300 ORIGINAL IMAGES
% =========================================================================
avgHist = zeros(256,1);
for k=1:numImages
    I = readGrayDouble(fullfile(imageFolder,files(k).name));
    h = customHistogram(I,256);
    h = h/sum(h);
    avgHist = avgHist + h;
end
avgHist = avgHist/numImages;
targetCDF = cumsum(avgHist);
targetCDF = targetCDF/targetCDF(end);
writematrix([(0:255)' avgHist targetCDF],fullfile(outputFolder,'TargetHistogram_Phase3.csv'));

%% ========================================================================
% 1. FULL-REFERENCE + NO-REFERENCE METRICS AT EACH DOSE
% =========================================================================
metricResults = table();

for k=1:numImages
    filename = files(k).name;
    I = readGrayDouble(fullfile(imageFolder,filename));

    for dose = verifyDoses
        noisy = simulatePoisson(I,N0,dose);
        enhanced = finalEnhancementChain(noisy,targetCDF,GAUSS_SIGMA,HIGHBOOST_K);

        % Baseline metrics.
        metricResults = [metricResults; evaluationRow(filename,dose,'Baseline',noisy,I,EME_BLOCKS)];

        % Enhanced metrics.
        metricResults = [metricResults; evaluationRow(filename,dose,'Enhanced',enhanced,I,EME_BLOCKS)];
    end
end

metricSummary = groupsummary(metricResults,{'Dose_Percent','Method'},{'mean','std'}, ...
    {'RMSE','PSNR_dB','SSIM','Entropy','EME'});

writetable(metricResults,fullfile(outputFolder,'Evaluation_AllImages.csv'));
writetable(metricSummary,fullfile(outputFolder,'Evaluation_Summary.csv'));

%% ========================================================================
% 2. AUTOMATIC ROI LOCAL-CONTRAST EVALUATION
% =========================================================================
% No manual interaction is required.
%
% Three standardized proportional ROIs are generated automatically on each
% representative image:
%   1) Left lung
%   2) Right lung
%   3) Mediastinum
%
% The same ROI coordinates are then reused for the original, degraded, and
% enhanced versions at every dose level, ensuring a fair comparison.
%
% IMPORTANT FOR THE REPORT:
% These ROIs are standardized geometric sampling regions, NOT anatomical
% segmentation. Their locations should be visually checked in the saved ROI
% figures. This is a methodological deviation if the assignment explicitly
% requires manually placed ROIs.

roiResults = table();
roiCoordinates = table();
roiNames = {'LeftLung','RightLung','Mediastinum'};

fprintf('\n============================================================\n');
fprintf('AUTOMATIC ROI STEP - no manual selection required\n');
fprintf('Representative images: 1, 150, and 300\n');
fprintf('ROIs: left lung, right lung, and mediastinum\n');
fprintf('============================================================\n\n');

for repID = repIDs
    idx = find(ids==repID,1);
    if isempty(idx)
        warning('Image (%d).png not found; skipping automatic ROI evaluation.',repID);
        continue;
    end

    filename = files(idx).name;
    I = readGrayDouble(fullfile(imageFolder,filename));
    [H,W] = size(I);

    % Standardized proportional ROIs [x y width height].
    % Coordinates scale automatically with image dimensions.
    rects = zeros(3,4);
    rects(1,:) = [0.18*W, 0.28*H, 0.20*W, 0.32*H]; % Left lung
    rects(2,:) = [0.62*W, 0.28*H, 0.20*W, 0.32*H]; % Right lung
    rects(3,:) = [0.44*W, 0.32*H, 0.12*W, 0.28*H]; % Mediastinum
    rects = round(rects);

    % Save a visual check of the automatically placed ROIs.
    f = figure('Name',sprintf('Automatic ROIs - Image (%d)',repID),'Visible','on');
    imshow(I,[]); hold on;
    for r=1:3
        rectangle('Position',rects(r,:),'EdgeColor','r','LineWidth',2);
        text(rects(r,1),max(1,rects(r,2)-10),roiNames{r}, ...
            'Color','y','FontWeight','bold','FontSize',10);
    end
    hold off;
    title(sprintf('Automatic standardized ROIs - Image (%d)',repID));
    exportgraphics(f,fullfile(figFolder,sprintf('Automatic_ROIs_Image_%d.png',repID)),'Resolution',300);

    for r=1:3
        roiCoordinates = [roiCoordinates; table(repID,string(roiNames{r}),rects(r,1),rects(r,2),rects(r,3),rects(r,4), ...
            'VariableNames',{'ImageID','ROI','X','Y','Width','Height'})];
    end

    for dose=verifyDoses
        noisy = simulatePoisson(I,N0,dose);
        enhanced = finalEnhancementChain(noisy,targetCDF,GAUSS_SIGMA,HIGHBOOST_K);

        for r=1:3
            roiO = cropByRect(I,rects(r,:));
            roiB = cropByRect(noisy,rects(r,:));
            roiE = cropByRect(enhanced,rects(r,:));

            % Local contrast: coefficient of variation (std / mean).
            cO = localContrastCV(roiO);
            cB = localContrastCV(roiB);
            cE = localContrastCV(roiE);

            roiResults = [roiResults; table(repID,string(filename),dose*100,string(roiNames{r}), ...
                cO,cB,cE,100*(cE-cB)/(cB+eps), ...
                'VariableNames',{'ImageID','Image','Dose_Percent','ROI','Original_LocalContrast','Baseline_LocalContrast','Enhanced_LocalContrast','ContrastGain_Percent'})];
        end
    end
end

writetable(roiCoordinates,fullfile(outputFolder,'Automatic_ROI_Coordinates.csv'));
writetable(roiResults,fullfile(outputFolder,'Automatic_ROI_LocalContrast.csv'));
if ~isempty(roiResults)
    roiSummary = groupsummary(roiResults,{'Dose_Percent','ROI'},{'mean','std'}, ...
        {'Baseline_LocalContrast','Enhanced_LocalContrast','ContrastGain_Percent'});
    writetable(roiSummary,fullfile(outputFolder,'Automatic_ROI_LocalContrast_Summary.csv'));
end

%% ========================================================================
% 3. STRUCTURED VISUAL ASSESSMENT: 20 RANDOMIZED BLINDED PAIRS
% =========================================================================
% Twenty unique image-dose combinations are sampled from the available
% 300 images x 3 dose levels. Enhanced and unenhanced images are randomized
% left/right. Reviewers see only A and B; the answer key is stored separately.

candidates = table();
for k=1:numImages
    for dose=verifyDoses
        candidates = [candidates; table(ids(k),string(files(k).name),dose*100, ...
            'VariableNames',{'ImageID','Image','Dose_Percent'})];
    end
end

nPairs = min(20,height(candidates));
perm = randperm(height(candidates),nPairs);
chosen = candidates(perm,:);
assessmentKey = table();

for p=1:nPairs
    repID = chosen.ImageID(p);
    idx = find(ids==repID,1);
    I = readGrayDouble(fullfile(imageFolder,files(idx).name));
    dose = chosen.Dose_Percent(p)/100;
    baseline = simulatePoisson(I,N0,dose);
    enhanced = finalEnhancementChain(baseline,targetCDF,GAUSS_SIGMA,HIGHBOOST_K);

    enhancedOnLeft = rand < 0.5;
    if enhancedOnLeft
        A = enhanced; B = baseline; enhancedSide = "A";
    else
        A = baseline; B = enhanced; enhancedSide = "B";
    end

    f = figure('Visible','off','Position',[100 100 1000 480]);
    tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
    nexttile; imshow(A,[]); title('A','FontSize',18);
    nexttile; imshow(B,[]); title('B','FontSize',18);
    sgtitle(sprintf('Pair %02d',p),'FontSize',18);
    exportgraphics(f,fullfile(assessmentFolder,sprintf('Pair_%02d.png',p)),'Resolution',220);
    close(f);

    assessmentKey = [assessmentKey; table(p,repID,chosen.Dose_Percent(p),enhancedSide, ...
        'VariableNames',{'PairID','ImageID','Dose_Percent','EnhancedSide'})];
end

writetable(assessmentKey,fullfile(outputFolder,'Visual_Assessment_AnswerKey_DO_NOT_SHOW_REVIEWERS.csv'));

% Blank response template for at least three colleagues.
ReviewerID = strings(nPairs*3,1);
PairID = zeros(nPairs*3,1);
PreferredSide = strings(nPairs*3,1);
Confidence_1to5 = nan(nPairs*3,1);
Comments = strings(nPairs*3,1);
row=0;
for r=1:3
    for p=1:nPairs
        row=row+1;
        ReviewerID(row)=sprintf('Reviewer%d',r);
        PairID(row)=p;
    end
end
responseTemplate = table(ReviewerID,PairID,PreferredSide,Confidence_1to5,Comments);
writetable(responseTemplate,fullfile(outputFolder,'Visual_Assessment_Response_Template.csv'));

%% ========================================================================
% VISUAL-ASSESSMENT ANALYSIS IS NOW A SEPARATE SCRIPT
% =========================================================================
% This main Phase-3 script only creates:
%   1) the 20 randomized A/B pairs,
%   2) the hidden answer key, and
%   3) the blank reviewer response template.
%
% After at least three colleagues complete their responses, save the filled
% file as:
%   Phase3_Results/Visual_Assessment_Responses.csv
%
% Then run ONLY:
%   Project1_Phase3_ReviewerAnalysis.m
%
% There is no need to rerun the expensive 300-image Phase-3 analysis.

fprintf('\nVisual-assessment pairs and response template are ready.\n');
fprintf('After reviewers finish, run Project1_Phase3_ReviewerAnalysis.m only.\n');

%% ========================================================================
% 4. CLAHE CLIP-LIMIT SENSITIVITY STUDY AT 25%% DOSE
% =========================================================================
% The study isolates the contrast stage after the selected Gaussian denoiser.
% Contrast gain is quantified by EME gain. Noise amplification is quantified
% by the increase in high-frequency residual energy relative to the denoised
% input. The crossover is the first clip limit where normalized noise
% amplification exceeds normalized EME contrast gain.

claheSensitivity = table();

for k=1:numImages
    filename = files(k).name;
    I = readGrayDouble(fullfile(imageFolder,filename));
    noisy = simulatePoisson(I,N0,claheDose);
    denoised = customGaussianBlur(noisy,GAUSS_SIGMA);

    baseEME = customEME(denoised,EME_BLOCKS(1),EME_BLOCKS(2));
    baseNoise = highFrequencyNoiseProxy(denoised);

    for c=clipSweep
        J = customCLAHE(denoised,CLAHE_GRID,c);
        eme = customEME(J,EME_BLOCKS(1),EME_BLOCKS(2));
        nproxy = highFrequencyNoiseProxy(J);
        rm = sqrt(mean((J(:)-I(:)).^2));
        ss = ssim(J,I);

        contrastGain = (eme-baseEME)/(abs(baseEME)+eps);
        noiseAmp = (nproxy-baseNoise)/(abs(baseNoise)+eps);

        claheSensitivity = [claheSensitivity; table(string(filename),c,eme,nproxy,rm,ss,contrastGain,noiseAmp, ...
            'VariableNames',{'Image','ClipLimit','EME','NoiseProxy','RMSE','SSIM','ContrastGain_Ratio','NoiseAmplification_Ratio'})];
    end
end

claheSummary = groupsummary(claheSensitivity,'ClipLimit',{'mean','std'}, ...
    {'EME','NoiseProxy','RMSE','SSIM','ContrastGain_Ratio','NoiseAmplification_Ratio'});

% Find first crossover in ascending clip-limit order.
claheSummary = sortrows(claheSummary,'ClipLimit');
crossMask = claheSummary.mean_NoiseAmplification_Ratio > claheSummary.mean_ContrastGain_Ratio;
if any(crossMask)
    ix = find(crossMask,1,'first');
    crossoverClip = claheSummary.ClipLimit(ix);
else
    crossoverClip = NaN;
end

writetable(claheSensitivity,fullfile(outputFolder,'CLAHE_Sensitivity_AllImages.csv'));
writetable(claheSummary,fullfile(outputFolder,'CLAHE_Sensitivity_Summary.csv'));
writematrix(crossoverClip,fullfile(outputFolder,'CLAHE_NoiseOvertakesContrast_ClipLimit.txt'));

f = figure('Name','CLAHE Sensitivity','Visible','on');
plot(claheSummary.ClipLimit,claheSummary.mean_ContrastGain_Ratio,'-o','LineWidth',1.5); hold on;
plot(claheSummary.ClipLimit,claheSummary.mean_NoiseAmplification_Ratio,'-s','LineWidth',1.5);
xlabel('CLAHE clip limit'); ylabel('Normalized change');
title('CLAHE Sensitivity: Contrast Gain vs Noise Amplification');
legend('EME contrast gain','Noise amplification','Location','best'); grid on;
if ~isnan(crossoverClip)
    xline(crossoverClip,'--',sprintf('Crossover %.4f',crossoverClip));
end
exportgraphics(f,fullfile(figFolder,'CLAHE_Sensitivity_Crossover.png'),'Resolution',300);

%% ========================================================================
% 5. STRETCH OBJECTIVE: LOCAL MEAN / LOCAL VARIANCE ENHANCEMENT
% =========================================================================
bonusResults = table();
if RUN_BONUS
    % DIP-style local-statistics parameters.
    localWindow = 7;
    E = 3.0;
    k0 = 0.4;
    k1 = 0.02;
    k2 = 0.4;

    % Compare against a moderate custom CLAHE setting at 25% dose.
    bonusCLAHEClip = 0.01;

    for k=1:numImages
        filename = files(k).name;
        I = readGrayDouble(fullfile(imageFolder,filename));
        noisy = simulatePoisson(I,N0,0.25);
        denoised = customGaussianBlur(noisy,GAUSS_SIGMA);

        Jlocal = localMeanVarianceEnhancement(denoised,localWindow,E,k0,k1,k2);
        Jclahe = customCLAHE(denoised,8,bonusCLAHEClip);

        bonusResults = [bonusResults; bonusRow(filename,'LocalMeanVariance',Jlocal,I,EME_BLOCKS)];
        bonusResults = [bonusResults; bonusRow(filename,'CLAHE_8x8_C0.010',Jclahe,I,EME_BLOCKS)];
    end

    bonusSummary = groupsummary(bonusResults,'Method',{'mean','std'},{'RMSE','PSNR_dB','SSIM','Entropy','EME'});
    writetable(bonusResults,fullfile(outputFolder,'Bonus_LocalAdaptive_AllImages.csv'));
    writetable(bonusSummary,fullfile(outputFolder,'Bonus_LocalAdaptive_Summary.csv'));
end

%% ========================================================================
% 6. LIBRARY-EQUIVALENT CORRECTNESS CHECK (APPENDIX ONLY)
% =========================================================================
% This section does NOT drive any project result. It only compares custom
% implementations against MATLAB library equivalents for one example image.
appendix = table();
idx = find(ids==1,1);
if ~isempty(idx)
    I = readGrayDouble(fullfile(imageFolder,files(idx).name));
    noisy = simulatePoisson(I,N0,0.25);

    % Histogram equalization check.
    C = customHistEq(noisy);
    L = histeq(noisy,256);
    appendix = [appendix; correctnessRow('Histogram Equalization','histeq',C,L,'Same operation; custom mapping used in project')];

    % Histogram matching check, if available.
    C = customHistMatch(noisy,targetCDF);
    if exist('imhistmatch','file')==2
        targetImage = synthTargetImageFromHistogram(avgHist,size(noisy));
        L = imhistmatch(noisy,targetImage,256);
        appendix = [appendix; correctnessRow('Histogram Matching','imhistmatch',C,L,'Library target image approximates the same average target histogram')];
    end

    % CLAHE approximate check.
    C = customCLAHE(noisy,8,0.01);
    if exist('adapthisteq','file')==2
        L = adapthisteq(noisy,'NumTiles',[8 8],'ClipLimit',0.01);
        appendix = [appendix; correctnessRow('CLAHE','adapthisteq',C,L,'Clip-limit conventions are not numerically identical')];
    end

    % High-boost approximate check.
    C = customHighBoost(noisy,GAUSS_SIGMA,HIGHBOOST_K);
    if exist('imsharpen','file')==2
        L = imsharpen(noisy,'Radius',GAUSS_SIGMA,'Amount',HIGHBOOST_K);
        appendix = [appendix; correctnessRow('High-Boost / Unsharp','imsharpen',C,L,'Parameter conventions differ; check is qualitative')];
    end
end
writetable(appendix,fullfile(outputFolder,'Appendix_Library_Correctness_Check.csv'));

%% ========================================================================
% REPRESENTATIVE PHASE-3 FIGURES: IMAGES 1, 150, 300 AT 25%% DOSE
% =========================================================================
for repID = repIDs
    idx = find(ids==repID,1);
    if isempty(idx), continue; end
    I = readGrayDouble(fullfile(imageFolder,files(idx).name));
    noisy = simulatePoisson(I,N0,0.25);
    enhanced = finalEnhancementChain(noisy,targetCDF,GAUSS_SIGMA,HIGHBOOST_K);

    f = figure('Name',sprintf('Phase 3 Evaluation Image %d',repID),'Visible','on');
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
    nexttile; imshow(I,[]); title('Original full-dose');
    nexttile; imshow(noisy,[]); title('25% unenhanced');
    nexttile; imshow(enhanced,[]); title('25% enhanced');
    sgtitle(sprintf('Phase 3 Evaluation - Image (%d)',repID));
    exportgraphics(f,fullfile(figFolder,sprintf('Phase3_Evaluation_Image_%d.png',repID)),'Resolution',300);
end

fprintf('\n============================================================\n');
fprintf('PHASE 3 NUMERICAL PROCESSING COMPLETE\n');
fprintf('Results saved in: %s\n',outputFolder);
fprintf('CLAHE crossover clip limit: ');
if isnan(crossoverClip), fprintf('not reached in tested range\n'); else, fprintf('%.4f\n',crossoverClip); end
fprintf('Visual assessment still requires at least 3 human reviewers.\n');
fprintf('============================================================\n\n');

disp(metricSummary);

%% ========================================================================
% LOCAL FUNCTIONS
% =========================================================================
function I = readGrayDouble(path)
Iraw = imread(path);
if ndims(Iraw)==3, Iraw = rgb2gray(Iraw); end
I = im2double(Iraw);
end

function J = simulatePoisson(I,N0,dose)
lambda = max(I*N0*dose,0);
K = poissrnd(lambda);
J = K/(N0*dose);
J = min(max(J,0),1);
end

function T = evaluationRow(filename,dose,method,J,I,emeBlocks)
err = J-I;
rmse = sqrt(mean(err(:).^2));
ps = 20*log10(1/(rmse+eps));
ss = ssim(J,I);
ent = customEntropy(J);
eme = customEME(J,emeBlocks(1),emeBlocks(2));
T = table(string(filename),dose*100,string(method),rmse,ps,ss,ent,eme, ...
    'VariableNames',{'Image','Dose_Percent','Method','RMSE','PSNR_dB','SSIM','Entropy','EME'});
end

function T = bonusRow(filename,method,J,I,emeBlocks)
err=J-I; rmse=sqrt(mean(err(:).^2)); ps=20*log10(1/(rmse+eps)); ss=ssim(J,I);
T=table(string(filename),string(method),rmse,ps,ss,customEntropy(J),customEME(J,emeBlocks(1),emeBlocks(2)), ...
    'VariableNames',{'Image','Method','RMSE','PSNR_dB','SSIM','Entropy','EME'});
end

function J = finalEnhancementChain(noisy,targetCDF,sigma,k)
S = customGaussianBlur(noisy,sigma);
C = customHistMatch(S,targetCDF);
J = customHighBoost(C,sigma,k);
end

function h = customHistogram(I,nBins)
idx = floor(min(max(I,0),1)*(nBins-1))+1;
h = zeros(nBins,1);
for q=1:numel(idx)
    h(idx(q)) = h(idx(q)) + 1;
end
end

function J = customHistEq(I)
h = customHistogram(I,256);
cdf = cumsum(h)/sum(h);
map = round(255*cdf);
idx = floor(min(max(I,0),1)*255)+1;
J = reshape(map(idx),size(I))/255;
end

function J = customHistMatch(I,targetCDF)
h = customHistogram(I,256);
c = cumsum(h)/sum(h);
map = zeros(256,1);
for a=1:256
    [~,j] = min(abs(targetCDF-c(a)));
    map(a)=j-1;
end
idx=floor(min(max(I,0),1)*255)+1;
J=reshape(map(idx),size(I))/255;
end

function J = customCLAHE(I,g,clipLimit)
[H,W]=size(I);
maps=zeros(256,g,g);
yc=round(linspace(1,H+1,g+1));
xc=round(linspace(1,W+1,g+1));
for r=1:g
    for c=1:g
        tile=I(yc(r):yc(r+1)-1,xc(c):xc(c+1)-1);
        h=customHistogram(tile,256);
        threshold=max(1,floor(clipLimit*numel(tile)));
        excess=sum(max(h-threshold,0));
        h=min(h,threshold);
        q=floor(excess/256);
        h=h+q;
        remainder=excess-q*256;
        if remainder>0
            pos=round(linspace(1,256,remainder));
            for z=1:numel(pos), h(pos(z))=h(pos(z))+1; end
        end
        maps(:,r,c)=cumsum(h)/sum(h);
    end
end
cy=(yc(1:end-1)+yc(2:end)-1)/2;
cx=(xc(1:end-1)+xc(2:end)-1)/2;
J=zeros(H,W);
for y=1:H
    [r1,r2,wy]=neighbors(cy,y);
    for x=1:W
        [c1,c2,wx]=neighbors(cx,x);
        b=floor(min(max(I(y,x),0),1)*255)+1;
        v11=maps(b,r1,c1); v12=maps(b,r1,c2);
        v21=maps(b,r2,c1); v22=maps(b,r2,c2);
        J(y,x)=(1-wy)*((1-wx)*v11+wx*v12)+wy*((1-wx)*v21+wx*v22);
    end
end
J=min(max(J,0),1);
end

function [i1,i2,w]=neighbors(c,v)
if v<=c(1)
    i1=1;i2=1;w=0;
elseif v>=c(end)
    i1=numel(c);i2=i1;w=0;
else
    i2=find(c>=v,1); i1=i2-1;
    w=(v-c(i1))/(c(i2)-c(i1));
end
end

function K = customGaussianKernel(sigma)
r = max(1,ceil(3*sigma));
[x,y] = meshgrid(-r:r,-r:r);
K = exp(-(x.^2+y.^2)/(2*sigma^2));
K = K/sum(K(:));
end

function J = customGaussianBlur(I,sigma)
K = customGaussianKernel(sigma);
% Generic convolution primitive only; the Gaussian kernel itself is built
% explicitly above rather than obtained from a filtering library.
J = conv2(I,K,'same');
J = min(max(J,0),1);
end

function J = customHighBoost(I,sigma,k)
blurred = customGaussianBlur(I,sigma);
mask = I-blurred;
J = I + k*mask;
J = min(max(J,0),1);
end

function ent = customEntropy(I)
h=customHistogram(I,256); p=h/sum(h); p=p(p>0);
ent=-sum(p.*log2(p));
end

function eme = customEME(I,k1,k2)
% Measure of Enhancement (EME): average blockwise log contrast.
[H,W]=size(I);
yEdges=round(linspace(1,H+1,k1+1));
xEdges=round(linspace(1,W+1,k2+1));
s=0; n=0;
for r=1:k1
    for c=1:k2
        B=I(yEdges(r):yEdges(r+1)-1,xEdges(c):xEdges(c+1)-1);
        bmax=max(B(:)); bmin=min(B(:));
        % Small floor avoids division by zero while preserving ranking.
        bmin=max(bmin,1/255);
        bmax=max(bmax,bmin);
        s=s+20*log10((bmax+eps)/(bmin+eps));
        n=n+1;
    end
end
eme=s/max(n,1);
end

function c = localContrastCV(R)
mu=mean(R(:));
c=std(R(:))/(mu+eps);
end

function R = cropByRect(I,rect)
x=max(1,round(rect(1))); y=max(1,round(rect(2)));
w=max(1,round(rect(3))); h=max(1,round(rect(4)));
x2=min(size(I,2),x+w-1); y2=min(size(I,1),y+h-1);
R=I(y:y2,x:x2);
end

function n = highFrequencyNoiseProxy(I)
% High-frequency residual standard deviation using a manually defined 3x3
% averaging kernel. This is a no-reference noise-amplification proxy.
K=ones(3,3)/9;
low=conv2(I,K,'same');
high=I-low;
% Exclude outer one-pixel border affected by zero-padding.
if size(high,1)>2 && size(high,2)>2
    high=high(2:end-1,2:end-1);
end
n=std(high(:));
end

function J = localMeanVarianceEnhancement(I,w,E,k0,k1,k2)
% DIP-style locally adaptive enhancement based on local mean and variance.
K=ones(w,w)/(w*w);
localMean=conv2(I,K,'same');
localMeanSq=conv2(I.^2,K,'same');
localVar=max(localMeanSq-localMean.^2,0);
globalMean=mean(I(:));
globalVar=var(I(:),1);
mask=(localMean<=k0*globalMean) & ...
     (localVar>=k1*globalVar) & ...
     (localVar<=k2*globalVar);
J=I;
J(mask)=E*I(mask);
J=min(max(J,0),1);
end

function T = correctnessRow(customMethod,libraryMethod,C,L,notes)
mad=mean(abs(C(:)-L(:)));
ss=ssim(C,L);
T=table(string(customMethod),string(libraryMethod),mad,ss,string(notes), ...
    'VariableNames',{'CustomMethod','LibraryEquivalent','MeanAbsoluteDifference','SSIM_CustomVsLibrary','Notes'});
end

function targetImage = synthTargetImageFromHistogram(avgHist,sz)
% Construct a deterministic synthetic image whose empirical histogram follows
% the target distribution. Used only for imhistmatch correctness checking.
n=prod(sz); counts=round(avgHist*n); diffn=n-sum(counts);
counts(end)=counts(end)+diffn;
v=zeros(n,1); pos=1;
for b=1:256
    cnt=max(0,counts(b));
    if cnt>0
        stop=min(n,pos+cnt-1);
        v(pos:stop)=(b-1)/255;
        pos=stop+1;
        if pos>n, break; end
    end
end
if pos<=n, v(pos:end)=1; end
targetImage=reshape(v,sz);
end
