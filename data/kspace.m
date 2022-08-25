addpath( genpath( [pwd,  filesep, 'NIfTI_20140122' ]));
rootdir='/media/user_home0/cdforigua/data/HCP/data';
fileFolder=fullfile(rootdir);
datadir=dir(fullfile(fileFolder));

for i=1:length(datadir)-2
    cd (rootdir)
    sub_id = datadir(i+2).name;
    cd ([rootdir,'/',sub_id,'/unprocessed/3T/T1w_MPR1/']);
    obj = load_untouch_nii('*.nii.gz');
    Data = obj.img;
    Data_float = double(Data);
    Data_norm = Data_float./4095.0;
    kData = fftshift(fftn(ifftshift(Data_norm)));
    Factor_truncate=4.0;
    x_range = round(size(kData,2)/Factor_truncate);
    y_range = round(size(kData,3)/Factor_truncate);
    kData_truncate=kData;
    kData_truncate(:,1:x_range,:)=0;
    kData_truncate(:,:,1:y_range)=0;
    kData_truncate(:,end-x_range+1:end,:)=0;
    kData_truncate(:,:,end-y_range+1:end)=0;
    
    kout = kData_truncate;
    out = fftshift(ifftn(ifftshift(kout)));
    out_real = abs(out);
    
    nz=size(Data_norm,1);
    nx=size(Data_norm,2);
    ny=size(Data_norm,3);
    
    out_norm=out_real./max(out_real(:));
    for j=1:nz
        out_float(j,:,:)=imresize(squeeze(out_norm(j,:,:)),[nx ny],'bilinear');
    end
    
    out_final = round(out_float.*4095.0);
    out_final = int16(out_final);
    save([sub_id,'_4LR.mat'],'out_final');
end
    
    