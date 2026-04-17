%% 1. Show original image (Section 3)
figure(1);

I = dicomread('knee1');
knee = mat2gray(I);

imshow(knee);

P = phantom(256);

subplot(1,2,1);
imshow(P, []);
title('Phantom');

subplot(1,2,2);
imshow(knee, []);
title('Knee MRI');


%% 2. Perform Radon Transform (Section 3)
theta = 0:179;
[R_knee, xp] = radon(knee, theta);

figure(2);
imagesc(theta, xp, R_knee);
colormap(hot);
colorbar;
title('Sinogram of Knee');


%% 3. Simple Backprojection Reconstruction (Section 4)
% Here you can try without applying a filter ('none')
I_simple = iradon(R_knee, theta, 'linear', 'none');

figure(3);
imshow(I_simple, []);
title('Simple Backprojection');


%% 4. Filtered Backprojection Reconstruction (Section 5)
% Using the default 'Ram-Lak' filter
I_fbp = iradon(R_knee, theta, 'linear', 'Ram-Lak');

figure(4);
imshow(I_fbp, []);
title('Filtered Backprojection (Ram-Lak)');


%% 5. Effect of Number of Projections
% High, medium, and low sampling (as required in the experiment)
angles_list = [180, 60, 20];

figure(5);

for i = 1:length(angles_list)
    % Redefine sampling angles
    theta_test = linspace(0, 179, angles_list(i));

    % Generate projections (Radon transform)
    R_test = radon(knee, theta_test);

    % Reconstruction
    I_test = iradon(R_test, theta_test, 'linear', 'Ram-Lak');

    subplot(1, 3, i);
    imshow(I_test, []);
    title(['Projections: ', num2str(angles_list(i))]);
end


%% 6 & 7. Noise Effect and Filter Robustness
% Add random noise to the sinogram
noise_level = 5;
R_noisy = R_knee + noise_level * randn(size(R_knee));

% Compare two filters: Ram-Lak (standard) and Hamming (smoother)
I_noisy_ramlak = iradon(R_noisy, theta, 'linear', 'Ram-Lak');
I_noisy_hamming = iradon(R_noisy, theta, 'linear', 'Hamming');

figure(6);
subplot(1, 2, 1);
imshow(I_noisy_ramlak, []);
title('Noisy + Ram-Lak');

subplot(1, 2, 2);
imshow(I_noisy_hamming, []);
title('Noisy + Hamming');


%% 8. Iterative Reconstruction (ART) - Physically Scaled Version
I_art = zeros(size(knee));

n_iter = 10; % Number of iterations

% Key point: lambda must be very small or scaled by image size
N = size(knee, 1);
lambda = 0.05 / N; % Properly scaled step size

fprintf('--- Running physically scaled ART reconstruction ---\n');

for k = 1:n_iter
    for j = 1:length(theta)
        % 1. Compute estimated projection
        p_est = radon(I_art, theta(j));

        % 2. Compute error
        error_vec = R_knee(:, j) - p_est;

        % 3. Backproject the error
        back_err = iradon(error_vec, theta(j), 'linear', 'none', 1, N);

        % 4. Update image (with scaling)
        I_art = I_art + lambda * back_err;
    end

    % Constraints: remove NaNs and negative values
    I_art(isnan(I_art)) = 0;
    I_art(I_art < 0) = 0;

    fprintf('Iteration %d completed, max value: %.2f\n', k, max(I_art(:)));
end

figure(7);
imshow(I_art, []);
colormap(gray);
title('ART Reconstruction (Normalized)');