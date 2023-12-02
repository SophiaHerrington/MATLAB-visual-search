% function H = ComputeRGBHistogram(img, Q)
%     % INPUT: img, an RGB image where pixels have RGB values in range 0-255
%     % INPUT: Q, the level of quantization of the RGB space e.g. 4
% 
%     % First, create qimg, an image where RGB values are normalized in the range 0 to (Q-1).
%     qimg = double(img) / 256;
%     qimg = floor(qimg * Q);
% 
%     % Now, create a single integer value for each pixel that summarizes the RGB value.
%     % We will use this as the bin index in the histogram.
%     bin = qimg(:, :, 1) * Q^2 + qimg(:, :, 2) * Q + qimg(:, :, 3);
% 
%     % 'bin' is a 2D image where each 'pixel' contains an integer value in
%     % the range 0 to Q^3-1 inclusive.
% 
%     % We will now use Matlab's histcounts function to build a frequency histogram
%     % from these values. First, we have to reshape the 2D matrix into a column vector.
%     vals = reshape(bin, [], 1);
% 
%     % Now we can use histcounts to create a histogram of Q^3 bins.
%     H = histcounts(vals, 0:Q^3);
% 
%     % Normalize the histogram so that the sum of its elements equals 1.
%     H = H / sum(H);
% end

function H = ComputeRGBHistogram(img, Q)
    % INPUT: img, an RGB image where pixels have RGB values in the range 0-255
    % INPUT: Q, the level of quantization of the RGB space

    % Quantize the RGB channels
    r_values = floor(img(:, :, 1) / (256 / Q));
    g_values = floor(img(:, :, 2) / (256 / Q));
    b_values = floor(img(:, :, 3) / (256 / Q));

    % Calculate the number of bins based on the quantization level
    num_bins = Q;

    % Create histograms for each channel
    r_hist = histcounts(r_values, 0:num_bins);
    g_hist = histcounts(g_values, 0:num_bins);
    b_hist = histcounts(b_values, 0:num_bins);

    % Concatenate the histograms to form the final RGB histogram
    H = [r_hist, g_hist, b_hist];

    % % Normalize the histogram so that the sum of its elements equals 1.
    H = H / sum(H);
end
