function [] = run_cuda(fname, varargin)
    if ~exist('externaltools\\run_cuda.exe','file')
        run_cuda_file(0,1,2);
    end
    input_parameters;
    n = length(varargin);
    nout = 0;
    oname = cell(1,n);
    for i = 1:n
        varname = varargin(i);
        varname = varname{1};
        if exist(varname, 'var')
            nout = nout + 1;
            oname{nout} = varname;
            var = eval(varname);
            fid = fopen(strcat('externaltools\\',varname),'w');
            for j=1:length(var)
                fprintf(fid,'%f\n',var(j));
            end
            fclose(fid);
        end
    end
    oname = oname(1:nout);
    disp(oname);
    system(sprintf('externaltools\\run_cuda.exe %s',fname));
end
