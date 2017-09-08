[v_rec,t,u_fw,v_fw,rec_x,rec_z]=run_forward;
stf = prepare_stf;
stf = {stf.stf};
K = run_adjoint(u_fw,v_fw,stf);