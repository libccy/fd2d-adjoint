 
function [jm] = eval_objective(m, ModRandString, usr_par)
% EVAL_OBJECTIVE function to evaluate the objective function at a given
% model m.
%
% Input:
% m : model
% usr_par : auxiliary user defined parameters (optional)
%
% Output:
% jm : objective value (double)
%
% See also EVAL_GRAD_OBJECTIVE and EVAL_OBJECTIVE_AND_GRADIENT.

% disp('----evaluating objective only');

%% initialise stuff
misfit_init = usr_par.misfit_init;
whichFrq    = usr_par.whichFrq;
g_obs       = usr_par.g_obs;
sEventInfo  = usr_par.sEventInfo;
sEventObs   = usr_par.sEventObs;
% InvProps    = usr_par.InvProps;

% inversion stuff
output_path = usr_par.output_path;


%% convert variable structures InvTbx -> my stuff
[Model] = map_m_to_parameters(m, usr_par);

%% calculate misfits
disp(['calculating current misfit']);

[misfit_total, misfit_seis, misfit_grav, ...
        g_recIter, g_src, sEventRecIter, sEventAdstfIter] = calc_misfits(Model, ...
                  g_obs, misfit_init(whichFrq).grav , ...
                  sEventInfo, sEventObs, misfit_init(whichFrq).seis, ...
                  'yessavefields','noplot', 'nosaveplots');

% save model and forward field info to file
TempFolder = [output_path,'/fwd_temp/'];
ModFolder = [output_path,'/fwd_temp/',ModRandString,'/'];
mkdir(ModFolder)
save([ModFolder,'model-adstf.mat'], ...
    'ModRandString', 'Model', 'sEventAdstfIter', 'g_src', '-v6');
save([ModFolder,'iter-rec.mat'], ...
    'ModRandString', 'Model', 'sEventRecIter', 'g_recIter', '-v6');

%% move saved matfiles to model specific folder
blips = dir([TempFolder,'*.mat']);
for ii = 1:numel(blips)
    bestand = blips(ii).name;
    oldfile = [TempFolder,bestand];
    newfile = [ModFolder,bestand];
    movefile(oldfile,newfile);
end; clearvars blips;


%% OUTPUT to Inversion Toolbox structure
jm = misfit_total;

%% save variables of current iteration to file
currentMisfits.misfit      = misfit_total;
currentMisfits.misfit_seis = misfit_seis;
currentMisfits.misfit_grav = misfit_grav;
save([ModFolder,'currentIter.misfits.mat'], 'currentMisfits', '-v6');


end