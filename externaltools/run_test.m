type = 1;
clc;tic;
if type == 0
    [v_rec,t,u_fw,v_fw,rec_x,rec_z]=run_forward;
elseif type > 0
    if type == 1
        compile_cuda;
    end
    [t,vx0,vz0,vx1,vz1,vx2,vz2] = run_cuda('runForward',{'t','vx0','vz0','vx1','vz1','vx2','vz2'});
end
toc;
if type <= 0
    pvx0 = v_rec{1}.x;
    pvz0 = v_rec{1}.z;
    pvx1 = v_rec{2}.x;
    pvz1 = v_rec{2}.z;
    pvx2 = v_rec{3}.x;
    pvz2 = v_rec{3}.z;
else
    pvx0 = vx0;
    pvz0 = vz0;
    pvx1 = vx1;
    pvz1 = vz1;
    pvx2 = vx2;
    pvz2 = vz2;
end

if type < 3
    subplot(3,2,1)
    plot(1:length(pvx0), pvx0);
    xlabel('vx')
    subplot(3,2,2)
    plot(1:length(pvz0), pvz0);

    subplot(3,2,3)
    plot(1:length(pvx1), pvx1);
    xlabel('vx')
    subplot(3,2,4)
    plot(1:length(pvz1), pvz1);

    subplot(3,2,5)
    plot(1:length(pvx2), pvx2);
    xlabel('vx')
    subplot(3,2,6)
    plot(1:length(pvz2), pvz2);
    xlabel('vz')
end