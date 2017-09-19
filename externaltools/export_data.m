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

        data.src_info = src_info;
        data.rec_x = rec_x;
        data.rec_z = rec_z;
        data.sfe = store_fw_every;
        data.model_type = model_type;
        data.source_amplitude = source_amplitude;

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
