function dst=cvpr_compare(F1, F2)

% This function should compare F1 to F2 - i.e. compute the distance
% between the two descriptors

% % For now it just returns a random number
% 
% x=F1-F2;
% x=x.^2; 
% x=sum(x); 
% dst=sqrt(x); 


 % This function compares F1 to F2 by computing the Bhattacharyya distance

    % Ensure the inputs are column vectors
    F1 = F1(:);
    F2 = F2(:);

    % Normalize histograms
    F1 = F1 / sum(F1);
    F2 = F2 / sum(F2);

    % Compute Bhattacharyya coefficient
    BC = sum(sqrt(F1 .* F2));

    % Compute Bhattacharyya distance
    dst = -log(BC);

    % Handle the case where BC is very close to 1 (due to rounding errors)
    if isnan(dst) || isinf(dst)
        dst = 0;
    end


return;
