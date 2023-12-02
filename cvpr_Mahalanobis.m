function dst = cvpr_Mahalanobis(F1, F2, mean_vec, covariance_mat)
    % Ensure the inputs are column vectors
    F1 = F1(:);
    F2 = F2(:);

    % Compute Mahalanobis distance
    diff_vec = F1 - F2;
    inv_covariance_mat = inv(covariance_mat);
    dst = sqrt(diff_vec' * inv_covariance_mat * diff_vec);

    % Handle special cases if needed
    if isnan(dst) || isinf(dst)
        dst = 0;
    end
end