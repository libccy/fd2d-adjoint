function [] = compile_cuda(fname, rmlink)
    if nargin < 2
        rmlink = 1;
    end
    if nargin < 1
        fname = 'run_cuda';
    end

    fid = fopen('externaltools\\compile_cuda.bat','w');
    fprintf(fid,strcat('VCVARS32.BAT&&nvcc externaltools\\',fname,'.cu -arch=sm_50  -Xcompiler "/wd4819" -o externaltools\\',fname,'.exe'));
    fclose(fid);

    system('externaltools\\compile_cuda.bat');
    delete('externaltools\\compile_cuda.bat');

    if rmlink
        expname = sprintf('externaltools\\%s.exp',fname);
        libname = sprintf('externaltools\\%s.lib',fname);
        if exist(expname, 'file')
            delete(expname);
        end
        if exist(libname, 'file')
            delete(libname);
        end
    end
end
