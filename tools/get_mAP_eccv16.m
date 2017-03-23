% mAP used in "Human Attribute Recognition by Deep Hierarchical Contexts, ECCV 2016."
% code: https://github.com/facebook/pose-aligned-deep-networks/blob/master/matlab/get_precision_recall.m
function [ap,rec,prec] = get_mAP_eccv16(scores,labels,num_truths)
    if nargin==2
        num_truths=sum(labels==1);
    end
    [srt1,srtd]=sort(scores,'descend');
    fp=cumsum(labels(srtd)==-1);
    tp=cumsum(labels(srtd)==1);
    rec=tp/num_truths;
    prec=tp./(fp+tp);

    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end