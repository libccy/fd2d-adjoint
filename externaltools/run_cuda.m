function varargout = run_cuda(varargin)
    if ~exist('externaltools\run_cuda.exe','file')
        compile_cuda;
    end
    export_data;
    
    system('externaltools\\run_cuda.exe');
    n = length(varargin);
    varargout = cell(1,n);
    for i = 1:n
        fpath = sprintf('externaltools\\%s',varargin{i});
        if exist(fpath, 'file')
            fid = fopen(fpath, 'rb');
            varargout{i} = fread(fid, inf, 'real*4');
            fclose(fid);
            delete(fpath);
        else
            varargout{i} = 0;
        end
    end
    
    if exist('externaltools\config','file')
        delete('externaltools\config');
    end
end
