function K = import_data(varargin)
    n = length(varargin);
    k = 1;
    for i = 1:n
        if isnumeric(varargin{i})&&length(varargin{i}) == 2
            dim = varargin{i};
            for j = k:i-1
                if ischar(varargin{j})
                    if length(K.(varargin{j})) == dim(1) * dim(2)
                        K.(varargin{j}) = spanarr(K.(varargin{j}), dim(1), dim(2));
                    end
                end
            end
            k = i + 1;
        elseif(isnumeric(varargin{i})&&length(varargin{i}) == 3)
            dim = varargin{i};
            for j = k:i-1
                if ischar(varargin{j})
                    if length(K.(varargin{j})) == dim(1) * dim(2) * dim(3)
                        K.(varargin{j}) = spanarr(K.(varargin{j}), dim(1), dim(2), dim(3));
                    end
                end
            end
            k = i + 1;
        else
            fpath = sprintf('externaltools\\%s',varargin{i});
            if exist(fpath, 'file')
                fid = fopen(fpath, 'rb');
                K.(varargin{i}) = fread(fid, inf, 'real*4');
                fclose(fid);
                % delete(fpath);
            else
                K.(varargin{i}) = 0;
            end
        end
    end
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
