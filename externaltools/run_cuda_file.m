function [] = run_cuda_file(fname, cpcuda, rmcuda)
    if nargin < 3
        rmcuda = 2;
    end
    if nargin < 2
        cpcuda = 0;
    end
    if nargin < 1 || ~fname
        fname = 'run_cuda';
    end

    fid = fopen('externaltools\\run_cuda_file.bat','w');
    if cpcuda == 0 && exist(strcat('externaltools\\',fname,'.exe'),'file')
        fprintf(fid,strcat('externaltools\\',fname,'.exe'));
    else
        fprintf(fid,strcat('VCVARS32.BAT&&nvcc externaltools\\',fname,'.cu  -Xcompiler "/wd4819" -o externaltools\\',fname,'.exe&&externaltools\\',fname,'.exe'));
    end
    fclose(fid);

    system('externaltools\\run_cuda_file.bat');
    delete('externaltools\\run_cuda_file.bat');

    if rmcuda
        if rmcuda ~= 2
            delete(sprintf('externaltools\\%s.exe',fname));
        end
        delete(sprintf('externaltools\\%s.exp',fname));
        delete(sprintf('externaltools\\%s.lib',fname));
    end
end
