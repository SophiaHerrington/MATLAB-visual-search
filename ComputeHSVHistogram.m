function H = ComputeHSVHistogram(img, Q)
    % INPUT: img, an RGB image where pixels have RGB values in the range 0-255
    % INPUT: Q, the level of quantization of the HSV space

    % Convert the RGB image to HSV color space
    img_hsv = rgb2hsv(img);

    % Quantize the HSV channels
    h_values = floor(img_hsv(:, :, 1) * Q);
    s_values = floor(img_hsv(:, :, 2) * Q);
    v_values = floor(img_hsv(:, :, 3) * Q);

    % Calculate the number of bins based on the quantization level
    num_bins = Q;

    % Create histograms for each channel
    h_hist = histcounts(h_values, 0:num_bins);
    s_hist = histcounts(s_values, 0:num_bins);
    v_hist = histcounts(v_values, 0:num_bins);

    % Concatenate the histograms to form the final HSV histogram
    H = [h_hist, s_hist, v_hist];
    % Normalize the histogram using L1 normalization
    %H = H / sum(H);
    
    % Alternatively, you can use L2 normalization
    H = H / sqrt(sum(H.^2));
end