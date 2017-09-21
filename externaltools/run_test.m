tic;
% [v_rec,t,u_fw,v_fw,rec_x,rec_z]=run_forward;
[stf_z, t, abs] = run_cuda('runForward',{'stf_z', 't','abs'});
toc;

subplot(2,1,1)
plot(t, stf_z);
xlabel('vx')
subplot(2,1,2)
plot(1:length(abs),abs);
xlabel('vz')

% stf = prepare_stf;
% stf = {stf.stf};
% K = run_adjoint(u_fw,v_fw,stf);

% subplot(2,1,1)
% plot(0.1:0.1:500,v_rec{1}.x);
% xlabel('vx')
% subplot(2,1,2)
% plot(0.1:0.1:500,v_rec{1}.z);
% xlabel('vz')
