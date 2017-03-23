function AP_voc12 = AP_VOC(labels, probs)
% calc AP for each column of inputs
    num_cls = size(labels, 2);
    AP_voc12 = zeros(num_cls, 1);
    for m = 1:num_cls
        gt = labels(:,m);
        out = probs(:,m);
        % compute precision/recall
        [~,si]=sort(out, 'descend');
        tp = gt(si);
        fp = (~gt(si));
        fp = cumsum(fp);
        tp = cumsum(tp);
        rec = tp/sum(gt);
        prec = tp./(fp+tp);
        % compute voc12 style average precision
        ap = VOCap(rec,prec);
        AP_voc12(m) = ap;
    end
end
function ap = VOCap(rec,prec)
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end