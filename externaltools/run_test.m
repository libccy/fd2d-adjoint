type = 1;
clc;
tic;
if type == 0
    [v_rec,t,u_fw,v_fw,rec_x,rec_z]=run_forward;
else
    if type == 1
        compile_cuda;
    end
    [stf_z, t, vx] = run_cuda('runForward',{'stf_z', 't','vx'});
end
toc;
if type == 0
    plot(1:length(v_rec{1}.x),v_rec{1}.x);
else
    plot(1:length(vx),vx);
end

% subplot(2,1,1)
% plot(t, stf_z);
% xlabel('vx')
% subplot(2,1,2)
% plot(1:length(vx),vx);
% xlabel('vz')

% stf = prepare_stf;
% stf = {stf.stf};
% K = run_adjoint(u_fw,v_fw,stf);

% subplot(2,1,1)
% plot(0.1:0.1:500,v_rec{1}.x);
% xlabel('vx')
% subplot(2,1,2)
% plot(1:length(v_rec{1}.z),v_rec{1}.z);
% xlabel('vz')
