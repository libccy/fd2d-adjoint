cfg = 1; clc;

if cfg < 0
    if cfg == -1
        [v_rec,t,u_fw,v_fw]=run_forward;
    elseif cfg == -2
        [v_rec,t,u_fw,v_fw]=run_forward;
        stf = prepare_stf;
        stf = {stf.stf};
        K = run_adjoint(u_fw,v_fw,stf);
    elseif cfg == -3
        v_rec = cell(1,nrec);
         for i=1:nrec
           v_rec{i} = struct();
           v_rec{i}.x = sEventRecIter(2).vel{i}.x;
           v_rec{i}.z = sEventRecIter(2).vel{i}.z;
         end
        
         disp(norm( v_rec{1}.x));
    end
elseif cfg >= 0
    compile_cuda;
    [nx, nz, nt, nrec, nsfe]=getn; 
    
    if abs(cfg) < 3
        [vx_rec, vz_rec, vx, vz, t] = run_cuda('vx_rec', 'vz_rec', 'vx', 'vz', 't');
        vx_rec = spanarr(vx_rec, nrec, nt);
        vz_rec = spanarr(vz_rec, nrec, nt);
        vx = spanarr(vx, nsfe, nx, nz);
        vz = spanarr(vz, nsfe, nx, nz);
        v_rec = cell(1,nrec);
        for i=1:nrec
           v_rec{i} = struct();
           v_rec{i}.x = vx_rec(i,1:nt);
           v_rec{i}.z = vz_rec(i,1:nt);
        end
        v_fw = struct();
        v_fw.x = vx;
        v_fw.z = vz;
        clear('vx_rec', 'vz_rec', 'vx', 'vz');
    else
        [rho, mu, lambda, t] = run_cuda('rho', 'mu', 'lambda','t');
        rho = spanarr(rho,nx,nz);
        mu = spanarr(mu,nx,nz);
        lambda = spanarr(lambda,nx,nz);
        imagesc(lambda);
    end
end

if abs(cfg) ==1 ||cfg ==-3
    n = length(v_rec);
    for i=1:n
        subplot(n,2,i*2-1)
        plot(t,v_rec{i}.x);
        xlabel('vx');

        subplot(n,2,i*2)
        plot(t,v_rec{i}.z);
        xlabel('vz');
    end
elseif abs(cfg) == 2
    animate_output(v_fw);
end
clear('i','n','cfg');

function [nx, nz, nt, nrec, nsfe] = getn() %#ok<STOUT>
    input_parameters;
    nsfe = nt / store_fw_every;
end

function [oarr] = spanarr(iarr, m, n, p)
    if nargin == 3
        oarr = zeros(m, n);
        if length(iarr) < m * n
            iarr = zeros(1, m * n);
        end
        for i = 1:m
            for j = 1:n
                oarr(i,j) = iarr((i-1)*n + j);
            end
        end
    elseif nargin == 4
        oarr = zeros(m, n, p);
        if length(iarr) < m * n * p
            iarr = zeros(1, m * n * p);
        end
        for i = 1:m
            for j = 1:n
                for k = 1:p
                   oarr(i,j,k) = iarr((i-1)*n*p + (j-1)*p + k);
                end
            end
        end
    end
end


