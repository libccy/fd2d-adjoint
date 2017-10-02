function [] = run_test()
    type = 1;
    clc;
    
    if type < 0
        if type == -1
            [v_rec,t,u_fw,v_fw]=run_forward; %#ok<ASGLU>
        end
    elseif type > 0
        if type == 1
            compile_cuda;
        end
        [t,vx,vz] = run_cuda('t','vx','vz');
        
        input_parameters;
        vx = spanarr(vx, nrec, nt);
        vz = spanarr(vz, nrec, nt);
        v_rec = cell(1,nrec);
        for i=1:nrec
           v_rec{i} = struct();
           v_rec{i}.x = vx(i,1:nt);
           v_rec{i}.z = vz(i,1:nt);
        end
    end

    if abs(type) < 3
        n = length(v_rec);
        for i=1:n
            subplot(n,2,i*2-1)
            plot(t,v_rec{i}.x);
            xlabel('vx');
            
            subplot(n,2,i*2)
            plot(t,v_rec{i}.z);
            xlabel('vz');
        end
    end
end

function oarr = spanarr(iarr, m, n)
    oarr = zeros(m, n);
    for i = 1:m
        oarr(i,1:n) = iarr((i-1)*n+1:i*n);
    end
end