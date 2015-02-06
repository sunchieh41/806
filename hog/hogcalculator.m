            %Matlab版HOG代?
function F = hogcalculator(img, cellpw, cellph, nblockw, nblockh,...
    nthet, overlap,issigned, normmethod);
 
% HOG特征由Dalal在2005 cvpr 的一篇?文中提出
 
% NORMMETHOD：重??中的特征?准化函?的方法
%       e?一??定的很小的?使分母不?0
%       v??准化前的特征向量
%       'none', which means non-normalization;
%       'l1', which means L1-norm normalization; V=V/(V+e)
%       'l2', which means L2-norm normalization; V=V/根?(V平方+e平方)
%       'l1sqrt',V=根?(V/(V+e))
%       'l2hys',l2的省略形式。?V最大值限制?0.2
 
if nargin < 2
    % 在DALAL?文中指出的在rows:128*columns:64情?下的最佳值，?定?DEFAULT
    cellpw = 8;
    cellph = 8;
    nblockw = 2;
    nblockh = 2;
    nthet = 9;
    overlap = 0.5;
    issigned = 'unsigned';
    normmethod = 'l2hys';
else
    if nargin < 9
        error('?入??不足.');
    end
end
 
[M, N, K] = size(img);  %M?行?，N?列?，K???
if mod(M,cellph*nblockh) ~= 0   %行?必???的高度的整?倍
    error('?片行?必???的高度的整?倍.');
end
if mod(N,cellpw*nblockw) ~= 0   %列?必???的?度的整?倍
    error('?片列?必???的?度的整?倍.');
end                            
if mod((1-overlap)*cellpw*nblockw, cellpw) ~= 0 ||...  %要使滑步后左?是整?
        mod((1-overlap)*cellph*nblockh, cellph) ~= 0
    error('滑步的像素??必???胞?元尺寸的整?倍');
end
 
%?置高斯空??值窗口的方差
delta = cellpw*nblockw * 0.5;
 
 
%?算梯度矩?  梯度的?算【-1，0，1】效果是很好的，而3*3的sobel算子或者2*2的?角矩?反而?系?的降低效果
hx = [-1,0,1];
hy = -hx';   %?置
gradscalx = imfilter(double(img),hx);  %imfilter是?波器，hx表示?波掩膜
gradscaly = imfilter(double(img),hy);
 
if K > 1
    gradscalx = max(max(gradscalx(:,:,1),gradscalx(:,:,2)), gradscalx(:,:,3));  %取RGB中最大值
    gradscaly = max(max(gradscaly(:,:,1),gradscaly(:,:,2)), gradscaly(:,:,3));
end
gradscal = sqrt(double(gradscalx.*gradscalx + gradscaly.*gradscaly));  %梯度矩? gradscal
 
% ?算梯度方向矩?
gradscalxplus = gradscalx+ones(size(gradscalx))*0.0001;  %防止?0，所以gradscalx加了0.0001
gradorient = zeros(M,N);                                 %初始化梯度方向矩?
% unsigned situation: orientation region is 0 to pi.
if strcmp(issigned, 'unsigned') == 1                     %?向的情?
    gradorient =...
        atan(gradscaly./gradscalxplus) + pi/2;           %加pi/2因?atan的??取值?-pi/2?始
    or = 1;
else
    % signed situation: orientation region is 0 to 2*pi. %有向的情?
    if strcmp(issigned, 'signed') == 1
        idx = find(gradscalx >= 0 & gradscaly >= 0);
        gradorient(idx) = atan(gradscaly(idx)./gradscalxplus(idx));
        idx = find(gradscalx < 0);
        gradorient(idx) = atan(gradscaly(idx)./gradscalxplus(idx)) + pi;
        idx = find(gradscalx >= 0 & gradscaly < 0);
        gradorient(idx) = atan(gradscaly(idx)./gradscalxplus(idx)) + 2*pi;
        or = 2;
    else
     %  error('Incorrect ISSIGNED parameter.');
        error('??ISSIGNED?入有?');
    end
end
 
% ?算?的滑步
xbstride = cellpw*nblockw*(1-overlap);   %x方向的滑步
ybstride = cellph*nblockh*(1-overlap);
xbstridend = N - cellpw*nblockw + 1;     %x方向?左角能?到的最大值
ybstridend = M - cellph*nblockh + 1;
 
% ???=ntotalbh*ntotalbw
ntotalbh = ((M-cellph*nblockh)/ybstride)+1; %除了第一?，后面每?都是只需要ybstride就可以加一?
ntotalbw = ((N-cellpw*nblockw)/xbstride)+1;
 
% hist3dbig存?三?直方?，其中外面加了一?包裹以方便?算
      hist3dbig = zeros(nblockh+2, nblockw+2, nthet+2);
        F = zeros(1, ntotalbh*ntotalbw*nblockw*nblockh*nthet);
        glbalinter = 0;
    
% 生成存?一??的特征值的向量
sF = zeros(1, nblockw*nblockh*nthet);
 
% 生成高斯?值的模板
[gaussx, gaussy] = meshgrid(0:(cellpw*nblockw-1), 0:(cellph*nblockh-1));   %生成一??的网格
weight = exp(-((gaussx-(cellpw*nblockw-1)/2)...
    .*(gaussx-(cellpw*nblockw-1)/2)+(gaussy-(cellph*nblockh-1)/2)...
    .*(gaussy-(cellph*nblockh-1)/2))/(delta*delta));
 
% ?值投票，三?插值
for btly = 1:ybstride:ybstridend
    for btlx = 1:xbstride:xbstridend
        for bi = 1:(cellph*nblockh)
            for bj = 1:(cellpw*nblockw)
                 
                i = btly + bi - 1;       %在整?坐?系中的坐?
                j = btlx + bj - 1;
                gaussweight = weight(bi,bj);
                 
                gs = gradscal(i,j);   %梯度值
                go = gradorient(i,j); %梯度方向
                           
                % calculate bin index of hist3dbig
                % ?算八?????中心?的坐?
                binx1 = floor((bj-1+cellpw/2)/cellpw) + 1;
                biny1 = floor((bi-1+cellph/2)/cellph) + 1;
                binz1 = floor((go+(or*pi/nthet)/2)/(or*pi/nthet)) + 1;
                 
                if gs == 0
                    continue;
                end
                 
                binx2 = binx1 + 1;
                biny2 = biny1 + 1;
                binz2 = binz1 + 1;
                 
                x1 = (binx1-1.5)*cellpw + 0.5;
                y1 = (biny1-1.5)*cellph + 0.5;
                z1 = (binz1-1.5)*(or*pi/nthet);
                 
                % trilinear interpolation.三?插值
                hist3dbig(biny1,binx1,binz1) =...
                    hist3dbig(biny1,binx1,binz1) + gs*gaussweight...
                    * (1-(bj-x1)/cellpw)*(1-(bi-y1)/cellph)...
                    *(1-(go-z1)/(or*pi/nthet));
                hist3dbig(biny1,binx1,binz2) =...
                    hist3dbig(biny1,binx1,binz2) + gs*gaussweight...
                    * (1-(bj-x1)/cellpw)*(1-(bi-y1)/cellph)...
                    *((go-z1)/(or*pi/nthet));
                hist3dbig(biny2,binx1,binz1) =...
                    hist3dbig(biny2,binx1,binz1) + gs*gaussweight...
                    * (1-(bj-x1)/cellpw)*((bi-y1)/cellph)...
                    *(1-(go-z1)/(or*pi/nthet));
                hist3dbig(biny2,binx1,binz2) =...
                    hist3dbig(biny2,binx1,binz2) + gs*gaussweight...
                    * (1-(bj-x1)/cellpw)*((bi-y1)/cellph)...
                    *((go-z1)/(or*pi/nthet));
                hist3dbig(biny1,binx2,binz1) =...
                    hist3dbig(biny1,binx2,binz1) + gs*gaussweight...
                    * ((bj-x1)/cellpw)*(1-(bi-y1)/cellph)...
                    *(1-(go-z1)/(or*pi/nthet));
                hist3dbig(biny1,binx2,binz2) =...
                    hist3dbig(biny1,binx2,binz2) + gs*gaussweight...
                    * ((bj-x1)/cellpw)*(1-(bi-y1)/cellph)...
                    *((go-z1)/(or*pi/nthet));
                hist3dbig(biny2,binx2,binz1) =...
                    hist3dbig(biny2,binx2,binz1) + gs*gaussweight...
                    * ((bj-x1)/cellpw)*((bi-y1)/cellph)...
                    *(1-(go-z1)/(or*pi/nthet));
                hist3dbig(biny2,binx2,binz2) =...
                    hist3dbig(biny2,binx2,binz2) + gs*gaussweight...
                    * ((bj-x1)/cellpw)*((bi-y1)/cellph)...
                    *((go-z1)/(or*pi/nthet));
            end
        end
        
        %F生成
            if or == 2   %有向的?候，BINZ=nthet+2要返回?BINZ=2，BINZ=1要??BINZ=nthet+1
                         %因??似一?首尾相接的?
                hist3dbig(:,:,2) = hist3dbig(:,:,2)...
                    + hist3dbig(:,:,nthet+2);
                hist3dbig(:,:,(nthet+1)) =...
                    hist3dbig(:,:,(nthet+1)) + hist3dbig(:,:,1);
            end
            hist3d = hist3dbig(2:(nblockh+1), 2:(nblockw+1), 2:(nthet+1));
             
         
            for ibin = 1:nblockh     %???每??胞?元
                for jbin = 1:nblockw
                    idsF = nthet*((ibin-1)*nblockw+jbin-1)+1;
                    idsF = idsF:(idsF+nthet-1);
                    sF(idsF) = hist3d(ibin,jbin,:);  %每??胞?元的nthet?BIN
                end
            end
            iblock = ((btly-1)/ybstride)*ntotalbw +...
                ((btlx-1)/xbstride) + 1;
            idF = (iblock-1)*nblockw*nblockh*nthet+1;
            idF = idF:(idF+nblockw*nblockh*nthet-1);
            F(idF) = sF;
            hist3dbig(:,:,:) = 0;
         
    end
end
 
F(F<0) = 0;   %?值清0
 
%?一化方法
e = 0.001;  %?了防止分母出?0，?定一??小的值e
l2hysthreshold = 0.2;
fslidestep = nblockw*nblockh*nthet;
switch normmethod
    case 'none'
    case 'l1'        %l1-norm
        for fi = 1:fslidestep:size(F,2)
            div = sum(F(fi:(fi+fslidestep-1)));
            F(fi:(fi+fslidestep-1)) = F(fi:(fi+fslidestep-1))/(div+e);
        end
    case 'l1sqrt'    %l1-sqrt
        for fi = 1:fslidestep:size(F,2)
            div = sum(F(fi:(fi+fslidestep-1)));
            F(fi:(fi+fslidestep-1)) = sqrt(F(fi:(fi+fslidestep-1))/(div+e));
        end
    case 'l2'        %l2-norm
        for fi = 1:fslidestep:size(F,2)
            sF = F(fi:(fi+fslidestep-1)).*F(fi:(fi+fslidestep-1));
            div = sqrt(sum(sF)+e*e);
            F(fi:(fi+fslidestep-1)) = F(fi:(fi+fslidestep-1))/div;
        end
    case 'l2hys'     %l2-Hys 限定最大不超?0.2
        for fi = 1:fslidestep:size(F,2)
            sF = F(fi:(fi+fslidestep-1)).*F(fi:(fi+fslidestep-1));
            div = sqrt(sum(sF)+e*e);
            sF = F(fi:(fi+fslidestep-1))/div;
            sF(sF>l2hysthreshold) = l2hysthreshold;
            div = sqrt(sum(sF.*sF)+e*e);
            F(fi:(fi+fslidestep-1)) = sF/div;
        end
    otherwise
        error('??NORMMETHOD?入不正确');
end
