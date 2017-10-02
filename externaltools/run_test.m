type = 2;
clc;

if type < 0
    if type == -1
        [v_rec,t,u_fw,v_fw]=run_forward;
    end
elseif type > 0
    if type <3
        compile_cuda;
    else
        type = type - 2;
    end
    [vx_rec, vz_rec, vx, vz, t] = run_cuda('vx_rec', 'vz_rec', 'vx', 'vz', 't');

    [nx, nz, nt, nrec, nsfe]=getn;
    vx_rec = spanarr(vx_rec, nrec, nt);
    vz_rec = spanarr(vz_rec, nrec, nt);
    vx = spanarr(vx, nsfe, nx, nz);
    vz = spanarr(vz, nsfe, nx, nz);
    v_rec = cell(1,nrec);
    for i=1:nrec
       v_rec{i} = struct();
       v_rec{i}.x = vx_rec(i,1:nt);
       v_rec{i}.z = vx_rec(i,1:nt);
    end
    v_fw = struct();
    v_fw.x = vx;
    v_fw.z = vz;
    clear('vx_rec', 'vz_rec', 'vx', 'vz');
end

if abs(type) ==1
    n = length(v_rec);
    for i=1:n
        subplot(n,2,i*2-1)
        plot(t,v_rec{i}.x);
        xlabel('vx');

        subplot(n,2,i*2)
        plot(t,v_rec{i}.z);
        xlabel('vz');
    end
    clear('i','n','type');
elseif abs(type) == 2
    animate_output(v_fw);
end

function [nx, nz, nt, nrec, nsfe] = getn() %#ok<STOUT>
    input_parameters;
    nsfe = nt / store_fw_every;
end

function oarr = spanarr(iarr, m, n, p)
    if nargin == 3
        oarr = zeros(m, n);
        for i = 1:m
            for j = 1:n
                oarr(i,j) = iarr((i-1)*n + j);
            end
        end
    elseif nargin == 4
        oarr = zeros(m, n, p);
        for i = 1:m
            for j = 1:n
                for k = 1:p
                   oarr(i,j,k) = iarr((i-1)*n*p + (j-1)*p + k);
                end
            end
        end
    end
end