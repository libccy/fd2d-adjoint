function [] = run_cuda(fname, varargin)
    if ~exist('externaltools\\run_cuda.exe','file')
        run_cuda_file(0,1,2);
    end
    n = length(varargin);
    for i = 1:n
        var = varargin(i);
        var = var{1};
        if exist(var, 'var')
            var = eval(var);
            disp(var);
        end
    end
    system(sprintf('externaltools\\run_cuda.exe %s',fname));
end
