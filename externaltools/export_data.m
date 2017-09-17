function [] = export_data(key, value)

    if nargin == 0
        input_parameters;
        fid = fopen('externaltools\config','w');
        list = {'nx','nz','nt'};
        for i=1:length(list)
            if exist(list{i},'var')
                fprintf(fid, '%s\n%f\n', list{i}, eval(list{i}));
            end
        end
        fclose(fid);
    elseif nargin == 2
        fid = fopen(sprintf('externaltools\\%s',key),'w');
        for i = 1:length(value)
            fprintf(fid, '%f\n', value(i));
        end
        fclose(fid);
    end

end

