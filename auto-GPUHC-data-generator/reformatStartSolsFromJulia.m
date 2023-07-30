sympref('FloatingPointOutput',false);
clear;

fileFolder = '/oscar/data/bkimia/cchien3/Trifocal_Relative_Pose_Estimator/auto-GPUHC-data-generator/';
problem_name = "";
inputSolName = strcat(problem_name, 'julia-start-sols-raw');
outputSolName = strcat(problem_name, 'start-sols.txt');
fullInputFileName = fullfile(fileFolder, inputSolName);
fullOutputFileName = fullfile(fileFolder, outputSolName);

solFileRd = fopen(fullInputFileName, 'r');
solFileWr = fopen(fullOutputFileName, 'w');

result = [];
ldata = textscan(solFileRd, '%s', 'Delimiter', {'im', ','});
line = ldata{1,1};

real_parts = strings(size(line, 1), 1);
imag_parts = strings(size(line, 1), 1);

indx = 1;
for i = 1:size(line, 1)
    
    cvt_str = string(line{i,1});
    clean_str = strrep(cvt_str, '[', '');
    
    cplx_str = strcat(clean_str,'i');
    cplx_num = double(cplx_str);
    
    real_parts(indx,1) = num2str(real(cplx_num), 15);
    imag_parts(indx,1) = num2str(imag(cplx_num), 15);
    indx = indx + 1;
    wr_str = strcat(real_parts(i,1), '\t', imag_parts(i,1), '\n');
    fprintf(solFileWr, wr_str);
    
    if mod(i,1000) == 0
        fprintf('. ');
    end
end
fprintf('\n');

fclose(solFileWr);