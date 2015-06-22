 
function [gm] = eval_grad_objective(m, ModRandString, usr_par)
% EVAL_GRAD_OBJECTIVE function to compute the gradient of the objective 
% function at a given model m.
%
% Input:
% m : model -- should be a 1d vector of numbers
% usr_par : auxiliary user defined parameters (optional)
%
% Output:
% gm : gradient (vector of same size as m)
%
% See also EVAL_OBJECTIVE and EVAL_OBJECTIVE_AND_GRADIENT.

% [jm, gm] = eval_objective_and_gradient(m, usr_par);

disp('----evaluating gradient only');

%% initialise stuff
input_parameters; 

rec_g       = usr_par.rec_g;
misfit_init = usr_par.misfit_init;
whichFrq    = usr_par.whichFrq;
Model_bg    = usr_par.Model_bg;
parametrisation = usr_par.parametrisation;
use_grav    = usr_par.use_grav;

% inversion parameters
output_path = usr_par.output_path;

% iter-specific parameters
% g_src           = usr_par.g_src;
% sEventAdstfIter = usr_par.sEventAdstfIter;

%% convert variable structures InvTbx -> my structures
[Model1] = map_m_to_parameters(m, usr_par);

%% retrieve forward fields and seismic / grav sources for correct model
TempFolder = [output_path,'/fwd_temp/'];
ModFolder = [output_path,'/fwd_temp/',ModRandString,'/'];

% move the mat-files from the model subfolder to the temp folder
blips = dir([ModFolder,'*.mat']);
for ii = 1:numel(blips)
    bestand = blips(ii).name;
    oldfile = [ModFolder,bestand];
    newfile = [TempFolder,bestand];
    movefile(oldfile,newfile);
end; 

% load the non-fwd field files (fwd fields are loaded per src in
% run_adjoint_persource)
load([TempFolder,'model-adstf.mat']);
% for ii = 1:numel(blips)
%     bestand = blips(ii).name;
%     load([TempFolder,bestand]);
% end

% make sure that the actual input model is used.
% -> there should be a test here to check if the models from file & from
%    function call are actually the same.
Model = Model1;

%% calculate gradients

% gravity
if strcmp(use_grav,'yes')
    %- calculate gravity kernels
    disp ' ';
    disp(['calculating gravity kernel']);
    
    % calculating the gravity kernel
    [Kg_temp] = compute_kernels_gravity(g_src,rec_g,'no'); % 'no' is for plotting gravity kernel update

    % normalising the gravity kernel
    Kg = norm_kernel(Kg_temp, normalise_misfits, ...
        misfit_init(whichFrq).grav);
    clearvars Kg_temp;

end

% seismic
disp ' ';
disp(['calculating seismic kernels']);
[Kseis_temp, sEventKnls_iter] = run_adjoint_persource(Model, sEventAdstfIter);

% normalise kernels
Kseis = norm_kernel(Kseis_temp, normalise_misfits, ...
    misfit_init(whichFrq).seis);


%% combine gradients to Ktotal

if strcmp(use_grav,'yes')
    % determine weight of respective kernels
    w_Kseis = 1;
    w_Kg = 1;
    
    % combine seismic and gravity kernels
    disp ' '; disp('combining gravity and seismic kernels');
    
    % add kernels in appropriate parametrisation
    Ktest = change_parametrisation_kernels('rhomulambda',parametrisation,Kseis,Model);
    switch parametrisation
        case 'rhomulambda'
            Ktest.rho.total = w_Kseis * Ktest.rho.total + w_Kg * Kg;
        case 'rhovsvp'
            Ktest.rho2.total = w_Kseis * Ktest.rho2.total  +  w_Kg * Kg;
        otherwise
            error('the parametrisation in which kernels are added was unknown');
    end
    
    % saving the total kernel in rho-mu-lambda
    K_total = change_parametrisation_kernels(parametrisation,'rhomulambda', Ktest,Model);

    clearvars('Ktest', 'Ktest1');
else
    K_total = Kseis(iter);
end

% reparametrising Kernel to inversion parametrisation
K_reparam = change_parametrisation_kernels('rhomulambda', parametrisation, K_total, Model);

% calculate relative kernels
K_rel = calculate_relative_kernels(K_reparam, Model_bg);

% filter kernels
K_rel = filter_kernels(K_rel, parametrisation, smoothgwid);

%% move matfiles back to model folder

for ii = 1:numel(blips)
    bestand = blips(ii).name;
    oldfile = [TempFolder,bestand];
    newfile = [ModFolder,bestand];
    movefile(oldfile,newfile);
end; 

%% convert the obtained gradient back to same struc as m
gm = map_gradparameters_to_gradm(K_rel, usr_par);

% usr_par.Kseis(iter)   = Kseis;
% usr_par.Kg{iter}      = Kg;
% usr_par.K_total(iter) = K_total;

end