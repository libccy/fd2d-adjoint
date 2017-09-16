function output = run_cuda(fname, inames, onames)
    if ~exist('externaltools\\run_cuda.exe','file')
        compile_cuda;
    end
    input_parameters;
    for i = 1:length(inames)
        if exist(inames{i}, 'var')
            argv = eval(inames{i});
            fid = fopen(strcat('externaltools\\',inames{i}),'w');
            for j=1:length(argv)
                fprintf(fid,'%f\n',argv(j));
            end
            fclose(fid);
        else
            inames{i} = [];
        end
    end
    inames(cellfun(@isempty,inames))=[];
    system(sprintf('externaltools\\run_cuda.exe %s',fname));
    for i = 1:length(inames)
        delete(sprintf('externaltools\\%s',inames{i}));
    end
    output = struct();
    for i = 1:length(onames)
        output.(onames{i}) = i;
    end
end
