function [] = export_data(key, value)

    if nargin == 0
        input_parameters;
        fid = fopen('externaltools\config','w');
        data = struct();
        data.nx = nx;
        data.nz = nz;
        data.nt = nt;
        data.dt = dt;
        data.Lx = Lx;
        data.Lz = Lz;
%         list = {'nx', 'nz', 'nt', 'Lx', 'Lz','dt'};
%         for i=1:length(list)
%             if exist(list{i},'var')
%                 fprintf(fid, '%s\n%f\n', list{i}, eval(list{i}));
%             end
%         end
        fprintf(fid, '%s', jsonencode(data));
        fclose(fid);
    elseif nargin == 2
        fid = fopen(sprintf('externaltools\\%s',key),'w');
        for i = 1:length(value)
            fprintf(fid, '%f\n', value(i));
        end
        fclose(fid);
    end

end
