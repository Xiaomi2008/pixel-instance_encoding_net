

hd5_file = '../tempdata/seg_fina_distance_only';
hd5_file ='../tempdata/seg_final_plus_distance.h5'
distance = hdf5read(hd5_file,'/Set_A_d2');
gt       = hdf5read(hd5_file,'/Set_A_t1');
h =22;
slcie = 80
d1 = distance(:,:,100:125);
g1 = gt(:,:,100:125);

h = [6:0.5:30];
%h = [22.5];
for i=1:length(h) 
    imhmin_d1 = imhmin(1-d1,h(i));
    ws = watershed(imhmin_d1);
    display(sprintf('watershed threshold h = %d, metric = %d', h(i), SNEMI3D_metrics(g1,ws)));
end
%figure, imagesc(label2rgb(ws,'jet','c','shuffle'));

subplot(1,2,1), imshow(label2rgb(ws,'jet','c','shuffle'));
title('ws')
subplot(1,2,2), imshow(label2rgb(g1,'jet','c','shuffle'))
title('gt')
