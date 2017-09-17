function varargout = run_cuda(fname, onames, expdata)
    if ~exist('externaltools\run_cuda.exe','file')
        compile_cuda;
    end
    
    if nargin < 3 
        expdata = 2;
    end
    if expdata
        export_data;
    end
    
    system(sprintf('externaltools\\run_cuda.exe %s',fname));
    n = length(onames);
    varargout = cell(1,n);
    for i = 1:n
        fpath = sprintf('externaltools\\%s',onames{i});
        if exist(fpath, 'file')
            fid = fopen(fpath, 'r');
            varargout{i} = textscan(fid, '%f', 'delimiter', '\n');
            varargout{i} = varargout{i}{1};
            fclose(fid);
            delete(fpath);
        else
            varargout{i} = 0;
        end
    end
    
    if expdata == 2 && exist('externaltools\config','file')
        delete('externaltools\config');
    end
end
