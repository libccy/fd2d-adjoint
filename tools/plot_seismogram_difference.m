function seisdif = plot_seismogram_difference(v_obs, v_rec, t)

% This function is (just like @ make_adjoint_sources) to plot the observed
% and recorded seismograms, and the difference between them.
%
% INPUT:
% - v_obs:  struct containing x and/or y and/or z traces of seismograms
% - v_rec:  struct containing x and/or y and/or z traces of seismograms. At
%           the very least, the same components as are present in v_obs
%           must be present in this struct, else errors will ensue.
% - t:      time axis.
%
% OUTPUT:
% - figure  plotting both sets of seismograms, plus the difference traces of
%   (v_rec - v_obs).
% - seisdif: figure handle of this figure

seisdif = figure;
set_figure_properties_bothmachines;
set(seisdif, 'OuterPosition', pos_seis);


% number of receivers for which we have seismograms
nrec = length(v_obs);

% number of components (x,y,z) for which seismograms have been recorded
ncomp = size(fieldnames(v_obs{1}), 1);

for irec = 1:nrec
    comp = fieldnames(v_obs{irec});
    for icomp = 1:length(comp);
        
%         subplot(ncomp,1,icomp);
        subplot(nrec,ncomp,(irec-1)*ncomp + icomp)
%         hold on
        plot(t,v_rec{irec}.(comp{icomp}),'k', ...
             t,v_obs{irec}.(comp{icomp}),'r--', ...
             t,v_rec{irec}.(comp{icomp}) - v_obs{irec}.(comp{icomp}), 'b');
        %     plot(t,v_rec.(comp{1}) - v_obs.(comp{1}), 'b', 'LineWidth',2)
%         hold off
        
        if irec==1
        title({[comp{icomp},' component:']; 'synth - black, obs - red, diff - blue'})
        end
        if irec==nrec
        xlabel('t [s]')
        end
        ylabel('v [m/s]')
        
%         icomp=icomp+1;
        
    end
end


end