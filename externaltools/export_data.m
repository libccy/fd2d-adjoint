function [] = export_data(key, value)
    if nargin == 2 && ischar(key)
        fid = fopen(sprintf('externaltools\\%s',key),'w');
        for i = 1:length(value)
            fprintf(fid, '%f\n', value(i));
        end
        fclose(fid);
    else
        input_parameters;
        fid = fopen('externaltools\config','w');

        data.nx = nx;
        data.nz = nz;
        data.nt = nt;
        data.dt = dt;
        data.Lx = Lx;
        data.Lz = Lz;
        
        data.use_given_model = 0;
        data.use_given_stf = 0;
        data.model_type = model_type;
        data.order = order;
        dat.obs_type = 0;
        dat.optimization_method = 0;
        
        data.sfe = store_fw_every;
        data.src_info = src_info;
        data.rec_x = rec_x;
        data.rec_z = rec_z;
        data.source_amplitude = source_amplitude;
        data.wave_propagation_type = wave_propagation_type;
        
        data.width = width;
        data.absorb_left = absorb_left;
        data.absorb_right = absorb_right;
        data.absorb_top = absorb_top;
        data.absorb_bottom = absorb_bottom;

        if nargin > 0 && isstruct(key)
            if isfield(key,'lambda') && isfield(key,'mu') && isfield(key,'rho')
                data.use_given_model = 1;
                export_data('lambda', key.lambda);
                export_data('rho', key.rho);
                export_data('mu', key.mu);
            end
            if isfield(key,'stf')
                data.use_given_stf = 1;
                for i = 1:length(key.stf)
                    export_data(sprintf('stf_x%d',i), key.stf(i).x);
                    export_data(sprintf('stf_y%d',i), key.stf(i).y);
                    export_data(sprintf('stf_z%d',i), key.stf(i).z);
                end
            end
        end
        
        fprintf(fid, '%s', jsonencode(data));
        fclose(fid);
    end
end
