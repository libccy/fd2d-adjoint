function [] = animate_output(v_fw)
    input_parameters;
    nsfe = size(v_fw.x);
    nsfe = nsfe(1);
    [X,Z,dx,dz]=define_computational_domain(Lx,Lz,nx,nz);
    set_figure_properties;
    position_figures;
    figure(fig_vel)

    for n = 1:nsfe
        vx = squeeze(v_fw.x(nsfe-n+1,:,:));
        vz = squeeze(v_fw.z(nsfe-n+1,:,:));
        plot_velocity_field;
    end
end
