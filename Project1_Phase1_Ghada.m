%% PROJECT 1
% Adaptive Contrast Restoration for Low-Dose Chest Radiography
% PHASE 1 - Acquisition Modelling
% Experiments:
%   1. Spatial resolution degradation
%   2. Intensity quantization
%   3. Low-dose Poisson noise simulation
% The images must be stored inside:
%   Images/

clear;
clc;
close all;

rng(1);   % Reproducible Poisson noise


%% ================================================================
%  SETTINGS
% =================================================================

imageFolder = "Images";
outputFolder = "Phase1_Results";

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

% Spatial resolution experiment
scaleFactors = [2 4 8];
interpMethods = {'nearest','bilinear','bicubic'};

% Quantization experiment
bitDepths = [7 6 5 4 3 2];   % Compare each level against original 8-bit image

% Low-dose experiment
doseFactors = [0.50 0.25 0.10];   % Original image is the reference, not re-simulated

% Assumed full-dose maximum photon count.
%
% This is not obtained from the PNG because the NIH images do not contain
% the original detector photon counts.
%
% It is therefore an approximate acquisition model.
N0 = 10000;


%% ================================================================
%  FIND ALL IMAGES
% =================================================================

files = dir(fullfile(imageFolder,'*.png'));

if isempty(files)
    error('No PNG images were found inside the Images folder.');
end

numImages = length(files);

fprintf('\n====================================================\n');
fprintf('PHASE 1 - ACQUISITION MODELLING\n');
fprintf('====================================================\n');
fprintf('Images found: %d\n',numImages);
fprintf('Results folder: %s\n\n',outputFolder);


%% ================================================================
%  CREATE RESULT TABLES
% =================================================================

resolutionResults = table();

quantizationResults = table();

doseResults = table();


%% ================================================================
%  PROCESS ALL IMAGES
% =================================================================

for k = 1:numImages

    filename = files(k).name;
    filepath = fullfile(imageFolder,filename);

    fprintf('Processing image %d / %d : %s\n',...
        k,numImages,filename);

    %% ------------------------------------------------------------
    % READ IMAGE
    % -------------------------------------------------------------

    Iraw = imread(filepath);

    % Convert RGB to grayscale if required
    if ndims(Iraw) == 3
        Iraw = rgb2gray(Iraw);
    end

    % Convert to double precision normalized range [0,1]
    I = im2double(Iraw);

    originalSize = size(I);


    %% ============================================================
    % EXPERIMENT 1
    % SPATIAL RESOLUTION
    % =============================================================

    for s = 1:length(scaleFactors)

        factor = scaleFactors(s);

        for m = 1:length(interpMethods)

            method = interpMethods{m};

            % ------------------------------------------------------
            % Downsample
            % ------------------------------------------------------

            lowResolution = imresize( ...
                I,...
                1/factor,...
                method);

            % ------------------------------------------------------
            % Restore image back to original dimensions
            % ------------------------------------------------------

            restored = imresize( ...
                lowResolution,...
                originalSize,...
                method);

            restored = min(max(restored,0),1);


            % ------------------------------------------------------
            % Quantitative measurements
            % ------------------------------------------------------

            currentMSE = immse(restored,I);

            currentPSNR = psnr(restored,I);

            currentSSIM = ssim(restored,I);


            % ------------------------------------------------------
            % Edge / gradient energy retention
            %
            % This measures how much anatomical high-frequency
            % information remains after downsampling.
            % ------------------------------------------------------

            [GxOriginal,GyOriginal] = imgradientxy(I);
            gradientOriginal = sqrt(GxOriginal.^2 + GyOriginal.^2);

            [GxRestored,GyRestored] = imgradientxy(restored);
            gradientRestored = sqrt(GxRestored.^2 + GyRestored.^2);

            originalEdgeEnergy = sum(gradientOriginal(:).^2);
            restoredEdgeEnergy = sum(gradientRestored(:).^2);

            edgeRetention = ...
                100 * restoredEdgeEnergy / ...
                (originalEdgeEnergy + eps);


            % ------------------------------------------------------
            % Store results
            % ------------------------------------------------------

            newRow = table( ...
                string(filename),...
                factor,...
                string(method),...
                currentMSE,...
                currentPSNR,...
                currentSSIM,...
                edgeRetention,...
                'VariableNames',{ ...
                'Image',...
                'DownsampleFactor',...
                'Interpolation',...
                'MSE',...
                'PSNR_dB',...
                'SSIM',...
                'EdgeEnergyRetention_Percent'});

            resolutionResults = ...
                [resolutionResults; newRow];

        end
    end


    %% ============================================================
    % EXPERIMENT 2
    % INTENSITY QUANTIZATION
    % =============================================================

    for b = 1:length(bitDepths)

        bits = bitDepths(b);

        levels = 2^bits;

        % Uniform quantization
        Iquantized = ...
            round(I*(levels-1))/(levels-1);


        % ----------------------------------------------------------
        % Metrics
        % ----------------------------------------------------------

        currentMSE = immse(Iquantized,I);
        currentPSNR = psnr(Iquantized,I);
        currentSSIM = ssim(Iquantized,I);

        uniqueLevels = numel(unique(Iquantized(:)));


        % ----------------------------------------------------------
        % Zero-gradient fraction
        %
        % Quantization creates flat intensity regions.
        % Increasing flat regions can indicate contouring/banding.
        % ----------------------------------------------------------

        [Gx,Gy] = gradient(Iquantized);

        gradientMagnitude = sqrt(Gx.^2 + Gy.^2);

        zeroGradientFraction = ...
            100 * sum(gradientMagnitude(:) < 1e-6) / ...
            numel(gradientMagnitude);


        % ----------------------------------------------------------
        % Store results
        % ----------------------------------------------------------

        newRow = table( ...
            string(filename),...
            bits,...
            levels,...
            uniqueLevels,...
            currentMSE,...
            currentPSNR,...
            currentSSIM,...
            zeroGradientFraction,...
            'VariableNames',{ ...
            'Image',...
            'Bits',...
            'TheoreticalLevels',...
            'ObservedLevels',...
            'MSE',...
            'PSNR_dB',...
            'SSIM',...
            'FlatPixel_Percent'});

        quantizationResults = ...
            [quantizationResults; newRow];

    end


    %% ============================================================
    % EXPERIMENT 3
    % LOW-DOSE POISSON SIMULATION
    % =============================================================

    for d = 1:length(doseFactors)

        dose = doseFactors(d);


        % ----------------------------------------------------------
        % STEP 1
        % Convert normalized intensity to approximate photon counts
        % ----------------------------------------------------------

        expectedPhotons = I * N0 * dose;


        % ----------------------------------------------------------
        % STEP 2
        % Apply Poisson counting statistics
        % ----------------------------------------------------------

        noisyPhotons = poissrnd(expectedPhotons);


        % ----------------------------------------------------------
        % STEP 3
        % Convert photon counts back to normalized image intensity
        % ----------------------------------------------------------

        lowDoseImage = ...
            noisyPhotons / (N0*dose);

        lowDoseImage = ...
            min(max(lowDoseImage,0),1);


        % ----------------------------------------------------------
        % Metrics
        % ----------------------------------------------------------

        currentMSE = immse(lowDoseImage,I);
        currentRMSE = sqrt(currentMSE);
        currentPSNR = psnr(lowDoseImage,I);
        currentSSIM = ssim(lowDoseImage,I);


        % ----------------------------------------------------------
        % Signal to noise ratio
        % ----------------------------------------------------------

        noiseImage = lowDoseImage - I;

        % ----------------------------------------------------------

        signalPower = mean(I(:).^2);

        noisePower = mean(noiseImage(:).^2);

        estimatedSNR = ...
            10*log10(signalPower/(noisePower + eps));


        % ----------------------------------------------------------
        % Store results
        % ----------------------------------------------------------

        newRow = table( ...
            string(filename),...
            dose*100,...
            N0*dose,...
            currentMSE,...
            currentRMSE,...
            currentPSNR,...
            currentSSIM,...
            estimatedSNR,...
            'VariableNames',{ ...
            'Image',...
            'Dose_Percent',...
            'MaximumPhotonCount',...
            'MSE',...
            'RMSE',...
            'PSNR_dB',...
            'SSIM',...
            'EstimatedSNR_dB'});

        doseResults = ...
            [doseResults; newRow];

    end

end


%% ================================================================
%  SAVE COMPLETE RESULTS
% =================================================================

writetable( ...
    resolutionResults,...
    fullfile(outputFolder,'SpatialResolution_AllImages.csv'));

writetable( ...
    quantizationResults,...
    fullfile(outputFolder,'Quantization_AllImages.csv'));

writetable( ...
    doseResults,...
    fullfile(outputFolder,'LowDose_AllImages.csv'));


%% ================================================================
%  CREATE SUMMARY TABLES
% =================================================================

%% ---------------------------------------------------------------
% Spatial resolution summary
% ----------------------------------------------------------------

resolutionSummary = groupsummary( ...
    resolutionResults,...
    {'DownsampleFactor','Interpolation'},...
    {'mean','std'},...
    {'MSE','PSNR_dB','SSIM','EdgeEnergyRetention_Percent'});

writetable( ...
    resolutionSummary,...
    fullfile(outputFolder,'SpatialResolution_Summary.csv'));


%% ---------------------------------------------------------------
% Quantization summary
% ----------------------------------------------------------------

quantizationSummary = groupsummary( ...
    quantizationResults,...
    'Bits',...
    {'mean','std'},...
    {'MSE','PSNR_dB','SSIM','FlatPixel_Percent'});

writetable( ...
    quantizationSummary,...
    fullfile(outputFolder,'Quantization_Summary.csv'));


%% ---------------------------------------------------------------
% Dose summary
% ----------------------------------------------------------------

doseSummary = groupsummary( ...
    doseResults,...
    'Dose_Percent',...
    {'mean','std'},...
    {'MSE','RMSE','PSNR_dB','SSIM','EstimatedSNR_dB'});

writetable( ...
    doseSummary,...
    fullfile(outputFolder,'LowDose_Summary.csv'));


%% ================================================================
%  CREATE VISUAL EXAMPLES
% =================================================================
%
% Representative figures are generated for Image (1), Image (150),
% and Image (300).  If one of these files is missing, the script skips it
% and prints a warning.

representativeIDs = [1 150 300];

for r = 1:length(representativeIDs)

    repID = representativeIDs(r);
    targetName = sprintf('Image (%d).png',repID);

    idx = find(strcmp({files.name},targetName),1);

    if isempty(idx)
        warning('Representative image not found: %s',targetName);
        continue;
    end

    exampleFile = fullfile(imageFolder,files(idx).name);
    I = imread(exampleFile);

    if ndims(I) == 3
        I = rgb2gray(I);
    end

    I = im2double(I);
    originalSize = size(I);


    %% ------------------------------------------------------------
    % FIGURE A - SPATIAL RESOLUTION
    % -------------------------------------------------------------

    f = figure('Name',sprintf('Spatial Resolution - Image %d',repID), ...
               'Visible','on');

    tiledlayout(3,4);

    for s = 1:length(scaleFactors)

        factor = scaleFactors(s);

        nexttile;
        imshow(I,[]);
        title(sprintf('Original - Factor %d',factor));

        for m = 1:length(interpMethods)

            method = interpMethods{m};

            down = imresize(I,1/factor,method);
            restored = imresize(down,originalSize,method);

            nexttile;
            imshow(restored,[]);
            title(sprintf('%s - x%d',method,factor));

        end
    end

    sgtitle(sprintf('Effect of Spatial Resolution - Image (%d)',repID));

    exportgraphics( ...
        f,...
        fullfile(outputFolder, ...
        sprintf('Figure_SpatialResolution_Image_%d.png',repID)),...
        'Resolution',300);

    % close(f); % Keep figure open automatically


    %% ------------------------------------------------------------
    % FIGURE B - QUANTIZATION
    % Original 8-bit reference + 7, 6, 5, 4, 3 and 2-bit versions
    % -------------------------------------------------------------

    f = figure('Name',sprintf('Quantization - Image %d',repID), ...
               'Visible','on');

    tiledlayout(2,4);

    nexttile;
    imshow(I,[]);
    title('Original (8-bit)');

    for b = 1:length(bitDepths)

        bits = bitDepths(b);
        levels = 2^bits;

        Iq = round(I*(levels-1))/(levels-1);

        nexttile;
        imshow(Iq,[]);
        title(sprintf('%d-bit (%d levels)',bits,levels));

    end

    sgtitle(sprintf('Effect of Intensity Quantization - Image (%d)',repID));

    exportgraphics( ...
        f,...
        fullfile(outputFolder, ...
        sprintf('Figure_Quantization_Image_%d.png',repID)),...
        'Resolution',300);

    % close(f); % Keep figure open automatically


    %% ------------------------------------------------------------
    % FIGURE C - LOW-DOSE SIMULATION
    % Original reference + simulated 50%, 25% and 10% dose
    % -------------------------------------------------------------

    f = figure('Name',sprintf('Low Dose - Image %d',repID), ...
               'Visible','on');

    tiledlayout(1,4);

    nexttile;
    imshow(I,[]);
    title('Original');

    for d = 1:length(doseFactors)

        dose = doseFactors(d);

        expectedPhotons = I*N0*dose;
        noisyPhotons = poissrnd(expectedPhotons);
        lowDose = noisyPhotons/(N0*dose);
        lowDose = min(max(lowDose,0),1);

        nexttile;
        imshow(lowDose,[]);
        title(sprintf('Simulated %.0f%%',dose*100));

    end

    sgtitle(sprintf('Simulated Low-Dose Chest Radiographs - Image (%d)',repID));

    exportgraphics( ...
        f,...
        fullfile(outputFolder, ...
        sprintf('Figure_LowDose_Image_%d.png',repID)),...
        'Resolution',300);

    % close(f); % Keep figure open automatically

end


%% ================================================================
%  SUMMARY GRAPHS
% =================================================================

%% ---------------------------------------------------------------
% QUANTIZATION SSIM
% ----------------------------------------------------------------

f = figure('Name','Quantization SSIM','Visible','on');

plot( ...
    quantizationSummary.Bits,...
    quantizationSummary.mean_SSIM,...
    '-o',...
    'LineWidth',1.5);

xlabel('Bit Depth');
ylabel('Mean SSIM');
title('Image Structure vs Quantization Bit Depth');
grid on;
set(gca,'XDir','reverse');

exportgraphics( ...
    f,...
    fullfile(outputFolder,'Graph_Quantization_SSIM.png'),...
    'Resolution',300);

% close(f); % Keep figure open automatically


%% ---------------------------------------------------------------
% LOW DOSE SSIM
% ----------------------------------------------------------------

f = figure('Name','Dose SSIM','Visible','on');

plot( ...
    doseSummary.Dose_Percent,...
    doseSummary.mean_SSIM,...
    '-o',...
    'LineWidth',1.5);

xlabel('Dose (%)');
ylabel('Mean SSIM');
title('Structural Similarity vs Radiation Dose');
grid on;

exportgraphics( ...
    f,...
    fullfile(outputFolder,'Graph_Dose_SSIM.png'),...
    'Resolution',300);

% close(f); % Keep figure open automatically


%% ---------------------------------------------------------------
% LOW DOSE SNR
% ----------------------------------------------------------------

f = figure('Name','Dose SNR','Visible','on');

plot( ...
    doseSummary.Dose_Percent,...
    doseSummary.mean_EstimatedSNR_dB,...
    '-o',...
    'LineWidth',1.5);

xlabel('Dose (%)');
ylabel('Mean Estimated SNR (dB)');
title('Signal-to-Noise Ratio vs Radiation Dose');
grid on;

exportgraphics( ...
    f,...
    fullfile(outputFolder,'Graph_Dose_SNR.png'),...
    'Resolution',300);

% close(f); % Keep figure open automatically


%% ================================================================
%  DISPLAY RESULTS
% =================================================================

fprintf('\n====================================================\n');
fprintf('PROCESSING COMPLETE\n');
fprintf('====================================================\n\n');

fprintf('SPATIAL RESOLUTION SUMMARY (MEAN + STD):\n');
disp(resolutionSummary);

fprintf('\nQUANTIZATION SUMMARY (MEAN + STD):\n');
disp(quantizationSummary);

fprintf('\nLOW-DOSE SUMMARY (MEAN + STD):\n');
disp(doseSummary);

fprintf('\nRepresentative figures generated for Image (1), Image (150), and Image (300) when available.\n');
fprintf('Results saved inside: %s\n',outputFolder);
