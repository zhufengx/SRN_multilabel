function [P_class, R_class, F1_class] = precision_recall_f1(labels, label_pred)
% calc P, R and F1 score for each column (class) of inputs
%   labels:         ground truth labels, type: logical, size: num_im*num_class
%   label_pred:     predicted labels, type: logical, size: num_im*num_class
    tp = (labels & label_pred);
    num_tp = sum(tp, 1)+eps;
    num_pred = sum(label_pred, 1)+eps;
    num_p = sum(labels, 1)+eps;
    P_class = num_tp./num_pred;
    R_class = num_tp./num_p+eps;
    F1_class = 2*P_class.*R_class./(P_class + R_class);
end