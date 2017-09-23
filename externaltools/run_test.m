tic;
% [v_rec,t,u_fw,v_fw,rec_x,rec_z]=run_forward;
[stf_z, t, abs] = run_cuda('runForward',{'stf_z', 't','abs'});
toc;

% a=zeros(8,8);
% b=zeros(8,8);
% e=zeros(8,8);
% for i=1:8
%     for j=1:8
%         a(i,j)=(i+4)*(j+6)-(i+1)/(j+5);
%         b(i,j)=(i+0)*(j+8)+(i+2)/(j+3);
%         e(i,j)=(i+10)*(j+18)+(i+12)/(j+13);
%     end
% end
% [c,d]=div_s_PSV(a,e,b,1,1,8,8,4);
% o=c;
% disp(o(3,3:6));
% disp(o(4,3:6));
% disp(o(5,3:6));
% disp(o(6,3:6));

% subplot(2,1,1)
% plot(t, stf_z);
% xlabel('vx')
% subplot(2,1,2)
% plot(1:length(abs),abs);
% xlabel('vz')

% stf = prepare_stf;
% stf = {stf.stf};
% K = run_adjoint(u_fw,v_fw,stf);

% subplot(2,1,1)
% plot(0.1:0.1:500,v_rec{1}.x);
% xlabel('vx')
% subplot(2,1,2)
% plot(0.1:0.1:500,v_rec{1}.z);
% xlabel('vz')
