clear; clc; 
addpath('./tools');

dataset = 'nus_wide';       % nus_wide, coco, wider_att
threshold = 0.5;   % threshold for generate positive predictions

switch dataset
    case 'nus_wide'
        % load ground-truth and predictions
        labels = (importdata('./datasets/nus_wide/nus_wide_test_label.txt')==1);       % num_sample * num_cls
        probs = importdata('./results/reference_nus_wide_predictions.txt');
        probs = 1./(1+exp(-probs)); 
        % for nus-wide, select valid images for evaluation
        load('./datasets/nus_wide/nus_test_accessible.mat');        % idx_final_test
        labels = labels(idx_final_test,:);
        probs = probs(idx_final_test,:);
        % evaluate
        mAP_voc = AP_VOC(labels, probs);
        [P_C, R_C, F1_C] = precision_recall_f1(labels, (probs>threshold));
        [P_O, R_O, F1_O] = precision_recall_f1(labels(:), reshape((probs>threshold),[],1));
        fprintf('mAP\tF1-C\tP-C\tR-C\tF1-O\tP-O\tR-O\n');
        metrics = 100*[mean(mAP_voc), mean(F1_C), mean(P_C), mean(R_C), F1_O, P_O, R_O];
        fprintf('%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n',metrics(1),metrics(2),metrics(3),metrics(4),metrics(5),metrics(6),metrics(7));
    case 'coco'
        % load ground-truth and predictions
        labels = (importdata('./datasets/coco/coco_test_label.txt')==1);       % num_sample * num_cls
        probs = importdata('./results/reference_coco_predictions.txt');
        probs = 1./(1+exp(-probs)); 
        % evaluate
        mAP_voc = AP_VOC(labels, probs);
        [P_C, R_C, F1_C] = precision_recall_f1(labels, (probs>threshold));
        [P_O, R_O, F1_O] = precision_recall_f1(labels(:), reshape((probs>threshold),[],1));
        fprintf('mAP\tF1-C\tP-C\tR-C\tF1-O\tP-O\tR-O\n');
        metrics = 100*[mean(mAP_voc), mean(F1_C), mean(P_C), mean(R_C), F1_O, P_O, R_O];
        fprintf('%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n',metrics(1),metrics(2),metrics(3),metrics(4),metrics(5),metrics(6),metrics(7));
    case 'wider_att'
        % load ground-truth and predictions
        labels = importdata('./datasets/wider_att/wider_att_test_label.txt');   % num_sample * num_cls
        probs = importdata('./results/reference_wider_att_predictions.txt');
        probs = 1./(1+exp(-probs));
        % evaluate
        AP_eccv16 = zeros(size(labels,2), 1);  
        for m = 1:size(labels,2)
            [AP_eccv16(m),~,~] = get_mAP_eccv16(probs(:,m),labels(:,m));
        end
        [~, ~, F1_C] = precision_recall_f1(labels==1, (probs>threshold));
        [~, ~, F1_O] = precision_recall_f1(labels(:)==1, reshape((probs>threshold),[],1));
        metrics = 100*[mean(AP_eccv16), mean(F1_C), F1_O];
        fprintf('mAP\tF1-C\tF1-O\n');
        fprintf('%.1f\t%.1f\t%.1f\n',metrics(1),metrics(2),metrics(3));    
    otherwise
        error('unknown dataset !');     
end
