%% Evaluate the pix-level AP on the reasonable subset in the KAIST test set

clc; clear; close all;

load ('gt_reasonable.mat')

load ('HMFFN320.mat') 
% load ('HMFFN640.mat')

scene = 'all';
% scene = 'day';
% scene = 'night';

im_size=[512,640];
num_images = length(gt_reasonable);
x_total=cell(1,num_images); y_total=cell(1,num_images);
ap_total = zeros(1,num_images);
for idx=1:num_images
seg_gt = single(zeros(im_size));
seg_gt_ignores = single(zeros(im_size));
bbs = gt_reasonable{1,idx};
bbs(:,3) = bbs(:,3)+bbs(:,1); bbs(:,4) = bbs(:,4)+bbs(:,2);
for gtind=1:size(bbs,1)
    ignore = bbs(gtind,5);
    gt = bbs(gtind,1:4);
    x1 = min(max(round(gt(1)),1),size(seg_gt,2));
    y1 = min(max(round(gt(2)),1),size(seg_gt,1));
    x2 = min(max(round(gt(3)),1),size(seg_gt,2));
    y2 = min(max(round(gt(4)),1),size(seg_gt,1));
    if ignore==0; seg_gt(y1:y2,x1:x2) = 1; 
    else seg_gt_ignores(y1:y2,x1:x2) = 1; end
end
seg_mask_dt = seges{idx,1}';
seg_dt = imresize(seg_mask_dt, im_size,'bilinear');

x = seg_dt(:); y = seg_gt(:); 
ignores = logical(seg_gt_ignores(:));
x(ignores)=-1; y(ignores)=-1;
x_total{1,idx}=x; y_total{1,idx}=y;
end

x=cell2mat(x_total); x=x(:); y=cell2mat(y_total); y=y(:); 
if strcmp(scene,'all'); 
elseif strcmp(scene,'day'); seed=length(x_total{1,1}); x=x(1:seed*1455,:); y=y(1:seed*1455,:);
elseif strcmp(scene,'night'); seed=length(x_total{1,1}); x=x((1+seed*1455):end,:); y=y((1+seed*1455):end,:);
end

tic;
ap = prec_rec(x, y, 'plotPR', 1, 'plotROC', 0, 'plotBaseline', 0); ylim([0 1]);
fprintf('Pix-level Average Precision = %0.3f \n', ap);
toc;


