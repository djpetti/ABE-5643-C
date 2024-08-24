%% Load images
left = imread('zed_left.png');
right = imread('zed_right.png');
figure, imshow(left);
title("Left Input");
figure, imshow(right);
title("Right Input");

%% Convert to grayscale, double, [0, 1] range for easier computation
left_gray = double(rgb2gray(left)) / 255.0;
right_gray = double(rgb2gray(right)) / 255.0;

%% Define image patch location (topleft [row col]) and size
patch_loc = [120 170];
patch_size = [100 100];

%% Extract patch (from left image)
patch_left = left_gray(patch_loc(1):(patch_loc(1) + patch_size(1) - 1), patch_loc(2):(patch_loc(2) + patch_size(2) - 1));
figure, imshow(patch_left);
title("Original Patch (Left)");

%% Extract strip (from right image)
strip_right = right_gray(patch_loc(1):(patch_loc(1) + patch_size(1) - 1), :);
figure, imshow(strip_right);
title("Extracted Strip (Right)");

%% Now look for the patch in the strip and report the best position (column index of topleft corner)
best_x = find_best_match(patch_left, strip_right);
disp(best_x);
patch_right = right_gray(patch_loc(1):(patch_loc(1) + patch_size(1) - 1), best_x:(best_x + patch_size(2) - 1));
figure, imshow(patch_right);
title("Matching Patch (Right)");

% Find best match
function best_x = find_best_match(patch, strip)
    % Find patch in strip and return column index (x value) of topleft corner
    smallest_diff = Inf;
    for x = 1:(size(strip, 2) - size(patch, 2) + 1)
        candidate = strip(1:end, x:(x + size(patch, 2) - 1));
        candidate_diff = norm(candidate - patch);
        if candidate_diff < smallest_diff
            smallest_diff = candidate_diff;
            best_x = x;
        end
    end
end