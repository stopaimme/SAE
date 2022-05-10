function patches = sampleIMAGES()
% sampleIMAGES
% Returns 10000 patches for training

%load IMAGES;    % load images from disk 
file_path =  'C:\Users\ss\Desktop\CrackForest-dataset-master\train\train\';          % 要转化的图片的文件夹
img_path_list = dir(strcat(file_path,'*.jpg'));        % 要转化的图片的详细信息
img_num = length(img_path_list);                       % 要转化的图片的个数
% 批量生成灰度图像并保存到创建的文件夹下
IMAGES=zeros(120,120,10);
for j = 1:10
    image_name = img_path_list(j).name;                               % 选择第j个图片
    originalimg =  imread(strcat(file_path,image_name));              % 读入第j个图片
    imgtemp=im2gray(originalimg);
    %imgtemp=rgb2gray(originalimg);                                    % 将第j个转化为灰度图
    %imgtemp2 = imbinarize(imgtemp);
    %imwrite(imgtemp,[new_folder,image_name]);% 灰度图像批量保存在文件夹下
    IMAGES(:,:,j) = imgtemp;
end

patchsize = 8;  % we'll use 8x8 patches 
numpatches = 10000;

% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Fill in the variable called "patches" using data 
%  from IMAGES.  
%  
%  IMAGES is a 3D array containing 10 images
%  For instance, IMAGES(:,:,6) is a 512x512 array containing the 6th image,
%  and you can type "imagesc(IMAGES(:,:,6)), colormap gray;" to visualize
%  it. (The contrast on these images look a bit off because they have
%  been preprocessed using using "whitening."  See the lecture notes for
%  more details.) As a second example, IMAGES(21:30,21:30,1) is an image
%  patch corresponding to the pixels in the block (21,21) to (30,30) of
%  Image 1
for imageNum = 1:10%在每张图片中随机选取1000个patch，共10000个patch
    [rowNum colNum] = size(IMAGES(:,:,imageNum));
    for patchNum = 1:1000%实现每张图片选取1000个patch
        xPos = randi([1, rowNum-patchsize+1]);
        yPos = randi([1, colNum-patchsize+1]);
        patches(:,(imageNum-1)*1000+patchNum) = reshape(IMAGES(xPos:xPos+patchsize-1,yPos:yPos+patchsize-1,...
                                                        imageNum),64,1);
    end
end


%% ---------------------------------------------------------------

patches = normalizeData(patches); %规范化

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)
 
patches = bsxfun(@minus, patches, mean(patches));%均值归0

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;%因为根据3sigma法则，95%以上的数据都在该区域内
                                                % 这里转换后将数据变到了-1到1之间

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end