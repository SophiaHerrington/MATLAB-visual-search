%% EEE3032 - Computer Vision and Pattern Recognition (ee3.cvpr)
%%
%% cvpr_visualsearch.m
%% Skeleton code provided as part of the coursework assessment
%%
%% This code will load in all descriptors pre-computed (by the
%% function cvpr_computedescriptors) from the images in the MSRCv2 dataset.
%%
%% It will pick a descriptor at random and compare all other descriptors to
%% it - by calling cvpr_compare.  In doing so it will rank the images by
%% similarity to the randomly picked descriptor.  Note that initially the
%% function cvpr_compare returns a random number - you need to code it
%% so that it returns the Euclidean distance or some other distance metric
%% between the two descriptors it is passed.
%%
%% (c) John Collomosse 2010  (J.Collomosse@surrey.ac.uk)
%% Centre for Vision Speech and Signal Processing (CVSSP)
%% University of Surrey, United Kingdom

close all;
clear all;

%% Edit the following line to the folder you unzipped the MSRCv2 dataset to
DATASET_FOLDER = 'G:\CV and PR labs\cvprlab\MSRC_ObjCategImageDatabase_v2';

%% Folder that holds the results...
DESCRIPTOR_FOLDER = 'G:\CV and PR labs\cvprlab\descriptors';

%% Add SIFT library to the MATLAB path
addpath('path_to_sift_library');  % Replace with the actual path to your SIFT library
%% and within that folder, another folder to hold the descriptors
%% we are interested in working with
DESCRIPTOR_SUBFOLDER='globalRGBhisto';


%% 1) Load all the descriptors into "ALLFEAT"
%% each row of ALLFEAT is a descriptor (is an image)

ALLFEAT=[];
ALLFILES=cell(1,0);
ctr=1;
allfiles=dir (fullfile([DATASET_FOLDER,'/Images/*.bmp']));
for filenum=1:length(allfiles)
    fname=allfiles(filenum).name;
    imgfname_full=([DATASET_FOLDER,'/Images/',fname]);
    img=double(imread(imgfname_full))./255;

    
    thesefeat=[];
    featfile=[DESCRIPTOR_FOLDER,'/',DESCRIPTOR_SUBFOLDER,'/',fname(1:end-4),'.mat'];%replace .bmp with .mat
    load(featfile,'gridFeatures');
    
    ALLFILES{ctr}=imgfname_full;
    ALLFEAT=[ALLFEAT ; gridFeatures];
    ctr=ctr+1;
end

%% 2) Pick an image at random to be the query

% Set a new random seed based on the current time
rng('shuffle');
NIMG=size(ALLFEAT,1);           % number of images in collection
queryimg=floor(rand() * NIMG);    % index of a random image
% Display the query image
query_image = imread(ALLFILES{queryimg});
figure (1); imshow(query_image);
title('Query Image');

% Get the full file name of the query image
query_image_filename = ALLFILES{queryimg};
% Extract the base file name without the extension
[~, base_filename, ~] = fileparts(query_image_filename);
% Split the base file name based on underscores
parts = strsplit(base_filename, '_');
% Extract the desired part
query_class = parts{1};  
disp(['Query Image File : ' query_class]);

%% Calculate mean vector and covariance matrix for Mahalanobis distance
mean_vec = mean(ALLFEAT);
covariance_mat = cov(ALLFEAT);

%% 3) Compute the distance of image to the query
dst=[];
for i=1:NIMG
    candidate=ALLFEAT(i,:);
    query=ALLFEAT(queryimg,:);
    %thedst=cvpr_compare(query,candidate);
    thedst = cvpr_Mahalanobis(query, candidate, mean_vec, covariance_mat);
    dst=[dst ; [thedst i]];
end

dst=sortrows(dst,1);  % sort the results 

%% PCA for Dimensionality Reduction
% Perform PCA for dimensionality reduction
[coeff, score, ~, ~, explained] = pca(ALLFEAT);

% Choose the number of principal components based on explained variance
num_components = find(cumsum(explained) >= 95, 1);

% Project descriptors onto the reduced-dimensional space
reduced_feats = score(:, 1:num_components);




%% 4) Visualise the results


SHOW=15; % Show top 15 results
dst=dst(1:SHOW,:);
outdisplay=[];
for i=1:size(dst,1)
   img=imread(ALLFILES{dst(i,2)});
   img=img(1:2:end,1:2:end,:); % make image a quarter size
   img=img(1:81,:,:); % crop image to uniform size vertically (some MSVC images are different heights)
   outdisplay=[outdisplay img];
end
figure(2);
imshow(outdisplay);
axis off



%% plot pr
precision = zeros(1, 30);
recall = zeros(1, 30);

 % Determine the number of available top-ranked images
NIMG = size(dst, 1);

for n = 1:30
    % Calculate true positives and false positives
    true_positive = 0;
    false_positive = 0;

    % % Ensure that the number of images to consider (n) does not exceed the available images
    n = min(n, NIMG);

    for i = 1:n
        % Get the index of the i-th top-ranked image
        rank_i_index = dst(i, 2);

        % Get the filename of the i-th top-ranked image
        rank_i_image_filename = ALLFILES{rank_i_index};

        % Extract the base filename without the extension
        [~, base_filename, ~] = fileparts(rank_i_image_filename);

        % Split the base filename based on underscores
        parts = strsplit(base_filename, '_');

        % Extract the desired part, which is the label
        image_label = parts{1};


        % If the image label matches the query class label, increment true_positive
        if strcmp(image_label, query_class)
            true_positive = true_positive + 1;
        else
            false_positive = false_positive + 1;
        end
        % % Update the confusion matrix
        % true_class_index = find(strcmp(query_class, class_labels)); % Modify class_labels accordingly
        % predicted_class_index = find(strcmp(image_label, class_labels));
        % confusion_matrix(true_class_index, predicted_class_index) = confusion_matrix(true_class_index, predicted_class_index) + 1;
    end

    % Calculate precision and recall for this iteration
    precision(n) = true_positive / (true_positive + false_positive);
    recall(n) = true_positive / 30; 

end

% Create and plot the PR curve
figure;
plot(recall, precision, '-o');
xlabel('Recall');
ylabel('Precision');
title('Precision-Recall Curve');

%% Bag of Visual words

% Step 1: Extract SIFT features from all images
siftFeaturesCell = cell(1, length(allfiles));  % Cell array to store SIFT features
for filenum = 1:length(allfiles)
    fname = allfiles(filenum).name;
    imgfname_full = fullfile([DATASET_FOLDER, '/Images/', fname]);
    img = imread(imgfname_full);

    % Convert the image to grayscale
    grayImg = rgb2gray(img);

    % Detect SIFT features
    points = detectSURFFeatures(grayImg);

    % Extract SIFT descriptors
    [siftFeatures, ~] = extractFeatures(grayImg, points, 'Method', 'SIFT');

    % Store SIFT features in cell array
    siftFeaturesCell{filenum} = siftFeatures % Extract Features property
end

% Concatenate SIFT features from all images

siftFeatures = cat(1, siftFeaturesCell{:});

% Step 2: Perform PCA for dimensionality reduction
siftFeaturesNorm = zscore(siftFeatures);

[coeff, score, ~, ~, explained] = pca(siftFeaturesNorm);

% Choose the number of principal components based on explained variance
num_components = find(cumsum(explained) >= 95, 1);

% Project descriptors onto the reduced-dimensional space
siftFeaturesReduced = score(:, 1:num_components);

% Perform k-Means clustering on the reduced features
num_clusters = 100;  % Adjust based on your dataset and computational resources
max_iterations = 500;  % Reduced the maximum number of iterations

% Use 'plus' for k-means++ initialization
opts = statset('MaxIter', max_iterations, 'UseParallel', 'always', 'UseSubstreams', 'always', 'Streams', RandStream('mlfg6331_64', 'Seed', 0));
[~, codebook] = kmeans(siftFeaturesReduced, num_clusters, 'Options', opts, 'Start', 'plus');


% Step 3: Assign visual words to each image and create histograms
histograms = zeros(length(allfiles), num_clusters);
for filenum = 1:length(allfiles)
    fname = allfiles(filenum).name;
    imgfname_full = fullfile([DATASET_FOLDER, '/Images/', fname]);
    img = single(rgb2gray(imread(imgfname_full)));

    % Detect SIFT features
    points = detectSURFFeatures(img);

    % Extract SIFT descriptors
    siftFeaturesForImage = extractFeatures(img, points, 'Method', 'SIFT');

    % Calculate mean and standard deviation
    meanSiftFeatures = mean(siftFeatures);
    stdSiftFeatures = std(siftFeatures);

    % Project onto the reduced-dimensional space
    siftFeaturesForImageReduced = bsxfun(@rdivide, bsxfun(@minus, siftFeaturesForImage, meanSiftFeatures), stdSiftFeatures) * coeff(:, 1:num_components);

    % Assign visual words to descriptors using the codebook
    [~, indices] = pdist2(codebook, siftFeaturesForImageReduced, 'euclidean', 'Smallest', 1);

    % Create a histogram for the current image
    histograms(filenum, :) = histcounts(indices, 1:num_clusters+1);
end

figure;
histogram(indices, 1:num_clusters+1);
title('Histogram for Image 1');

%% SVM classsification

labels = cellfun(@(x) strsplit(x, '_'), ALLFILES, 'UniformOutput', false);
categories = cellfun(@(x) x{1}, labels, 'UniformOutput', false);
unique_categories = unique(categories);

% Create a label vector
label_vector = zeros(NIMG, 1);
for i = 1:length(unique_categories)
    label_vector(strcmp(categories, unique_categories{i})) = i;
end

% Split data into training and testing sets (e.g., 80% for training, 20% for testing)
split_ratio = 0.8;
num_train = round(split_ratio * NIMG);
train_indices = randperm(NIMG, num_train);
test_indices = setdiff(1:NIMG, train_indices);

train_data = reduced_feats(train_indices, :);
train_labels = label_vector(train_indices);

test_data = reduced_feats(test_indices, :);
test_labels = label_vector(test_indices);

% Normalize the data
train_data = zscore(train_data);
test_data = zscore(test_data);


%% Train SVM model
svm_model = fitcecoc(train_data, train_labels);

%% predict and evaluate

% Predict labels for test data
predicted_labels = predict(svm_model, test_data);

disp(['Total number of images: ' num2str(length(ALLFILES))]);
disp(['Number of unique classes: ' num2str(length(unique_categories))]);


disp(['Number of training samples: ' num2str(length(train_labels))]);
disp(['Number of testing samples: ' num2str(length(test_labels))]);


% Evaluate the accuracy
accuracy = sum(predicted_labels == test_labels) / length(test_labels);
disp(['Accuracy: ' num2str(accuracy)]);

