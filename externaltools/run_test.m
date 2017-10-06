clc;
cfg = 0;

[nx, nz, nt, nrec, nsfe, dt] = getn;
if cfg >= 0
    compile_cuda;
    if cfg == 1
        [vx_rec, vz_rec, t] = run_cuda('vx_rec', 'vz_rec', 't');
        vx_rec = spanarr(vx_rec, nrec, nt);
        vz_rec = spanarr(vz_rec, nrec, nt);

        v_rec = cell(1,nrec);
        for i=1:nrec
            v_rec{i}.x = vx_rec(i,1:nt);
            v_rec{i}.z = vz_rec(i,1:nt);
        end
        clear('i', 'vx_rec', 'vz_rec');
    elseif cfg == 2
        [vx_rec, vz_rec, vx, vz, t] = run_cuda('vx_rec', 'vz_rec', 'vx', 'vz', 't');
        vx = spanarr(vx, nsfe, nx, nz);
        vz = spanarr(vz, nsfe, nx, nz);
        v_fw.x = vx;
        v_fw.z = vz;
        clear('vx', 'vz');
    elseif cfg == 3
        [rho, mu, lambda, t] = run_cuda('rho', 'mu', 'lambda','t');
        rho = spanarr(rho,nx,nz);
        mu = spanarr(mu,nx,nz);
        lambda = spanarr(lambda,nx,nz);
        K.lambda.total = lambda;
        K.mu.total = mu;
        K.rho.total = rho;
        plot_kernels(K);
    elseif cfg == 4
        [vx_syn, vx_obs, vx_stf, vz_syn, vz_obs, vz_stf, t] = run_cuda('vx_syn', 'vx_obs', 'vx_stf', 'vz_syn', 'vz_obs', 'vz_stf', 't');
         vx_syn = spanarr(vx_syn, nrec, nt);
         vx_obs = spanarr(vx_obs, nrec, nt);
         vx_stf = spanarr(vx_stf, nrec, nt);
         
         vz_syn = spanarr(vz_syn, nrec, nt);
         vz_obs = spanarr(vz_obs, nrec, nt);
         vz_stf = spanarr(vz_stf, nrec, nt);
        for i=1:nrec
            subplot(nrec,2,i*2-1)
            hold on
            plot(t, vx_syn(i, 1:nt), 'r');
            plot(t ,vx_obs(i, 1:nt), 'b');
            plot(t, vx_stf(i, 1:nt), 'color',[1 0.5 0]);
            xlabel('vx');
            hold off

            subplot(nrec,2,i*2)
            hold on
            plot(t, vz_syn(i, 1:nt), 'r');
            plot(t ,vz_obs(i, 1:nt), 'b');
            plot(t, vz_stf(i, 1:nt), 'color',[1 0.5 0]);
            xlabel('vz');
            hold off
        end
    else
        [t] = run_cuda('t');
    end
elseif cfg < 0
    if cfg >=-3
        [v_rec,t,u_fw,v_fw]=run_forward;
        if cfg == -3
            stf = prepare_stf;
            stf = {stf.stf};
            K = run_adjoint(u_fw, v_fw,stf);
        end
    elseif cfg == -4
        stf = sEventAdstf{1};
        stf = stf.adstf;
        rec = sEventRecIter(1);
        rec = rec.vel;
        obs = sEventObs(1);
        obs = obs.vel;
        for i=1:3
            subplot(nrec,2,i*2-1)
            hold on
            plot(t,cumsum(rec{i}.x,2)*dt, 'r');
            plot(t,cumsum(obs{i}.x,2)*dt, 'b');
            plot(t, fliplr(stf{i}.x)/2, 'color',[1 0.5 0]);
            xlabel('vx');
            hold off

            subplot(nrec,2,i*2)
            hold on
            plot(t,cumsum(rec{i}.z,2)*dt, 'r');
            plot(t,cumsum(obs{i}.z,2)*dt, 'b');
            plot(t, fliplr(stf{i}.z)/2, 'color',[1 0.5 0]);
            xlabel('vz');
            hold off
        end
    end
end

if abs(cfg) == 1
    plotv(v_rec,t);
elseif abs(cfg) == 2
    animate_output(v_fw);
end

function [] = plotv(v_rec,t)
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

function [nx, nz, nt, nrec, nsfe, dt] = getn() %#ok<STOUT>
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


