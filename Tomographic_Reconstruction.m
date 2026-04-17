I = dicomread('knee1');
knee = mat2gray(I);
imshow (knee)



%% 1. Generate Shepp-Logan phantom
N = 256;
P = phantom(N);

figure;
imshow(P, []);
title('Shepp-Logan Phantom');

%% 2. Load knee MRI image
I = dicomread('knee1');
knee = mat2gray(I);

figure;
imshow(knee, []);
title('Knee MRI');

%% 3. Define projection angles
theta = 0:179;   % 180 projections

%% 4. Radon transform
[Rp, xp] = radon(P, theta);
[Rk, xk] = radon(knee, theta);

figure;
imagesc(theta, xp, Rp);
colormap hot; colorbar;
xlabel('Angle (degrees)');
ylabel('Detector position');
title('Sinogram of Phantom');

figure;
imagesc(theta, xk, Rk);
colormap hot; colorbar;
xlabel('Angle (degrees)');
ylabel('Detector position');
title('Sinogram of Knee MRI');

%% 5. Simple backprojection (no filter)
BP = iradon(Rp, theta, 'linear', 'none', 1, N);

figure;
imshow(BP, []);
title('Simple Backprojection (No Filter)');

%% 6. Filtered backprojection (Ram-Lak)
FBP = iradon(Rp, theta, 'linear', 'Ram-Lak', 1, N);

figure;
imshow(FBP, []);
title('Filtered Backprojection (Ram-Lak)');

%% 7. Influence of number of projections
theta_high = linspace(0,179,180);
theta_med  = linspace(0,179,60);
theta_low  = linspace(0,179,20);

R_high = radon(P, theta_high);
R_med  = radon(P, theta_med);
R_low  = radon(P, theta_low);

rec_high = iradon(R_high, theta_high, 'linear', 'Ram-Lak', 1, N);
rec_med  = iradon(R_med, theta_med, 'linear', 'Ram-Lak', 1, N);
rec_low  = iradon(R_low, theta_low, 'linear', 'Ram-Lak', 1, N);

figure;
subplot(1,3,1); imshow(rec_high, []); title('180 projections');
subplot(1,3,2); imshow(rec_med, []); title('60 projections');
subplot(1,3,3); imshow(rec_low, []); title('20 projections');

%% 8. Influence of noise
noise_level = 10;
noise_level2 = 3;
Rp_noisy = Rp + noise_level * randn(size(Rp));
Rp_noisy2 = Rp + noise_level2 * randn(size(Rp));

rec_noisy = iradon(Rp_noisy, theta, 'linear', 'Ram-Lak', 1, N);
rec_noisy2 = iradon(Rp_noisy2, theta, 'linear', 'Ram-Lak', 1, N);

figure;
subplot(1,3,1); imshow(FBP, []); title('FBP without noise');
subplot(1,3,2); imshow(rec_noisy, []); title('FBP High Noise');
subplot(1,3,3); imshow(rec_noisy2, []); title('FBP Low Noise');

%% 9. Filter comparison
rec_ramlak   = iradon(Rp_noisy, theta, 'linear', 'Ram-Lak', 1, N);
rec_shepp    = iradon(Rp_noisy, theta, 'linear', 'Shepp-Logan', 1, N);
rec_cosine   = iradon(Rp_noisy, theta, 'linear', 'Cosine', 1, N);
rec_hamming  = iradon(Rp_noisy, theta, 'linear', 'Hamming', 1, N);
rec_hann     = iradon(Rp_noisy, theta, 'linear', 'Hann', 1, N);

figure;
subplot(2,3,1); imshow(rec_ramlak, []); title('Ram-Lak');
subplot(2,3,2); imshow(rec_shepp, []); title('Shepp-Logan');
subplot(2,3,3); imshow(rec_cosine, []); title('Cosine');
subplot(2,3,4); imshow(rec_hamming, []); title('Hamming');
subplot(2,3,5); imshow(rec_hann, []); title('Hann');


% 
%ART
clc; clear; close all;

%% Parameters
N = 64;                        % image size (start small for speed)
num_iter = 10;                 % number of ART iterations
lambda = 0.2;                  % relaxation parameter
theta = 0:3:177;               % projection angles

%% Generate phantom
img_true = phantom(N);

figure;
imshow(img_true, []);
title('Original Shepp-Logan Phantom');

%% Generate sinogram
[R, xp] = radon(img_true, theta);

figure;
imagesc(theta, xp, R);
colormap gray; colorbar;
xlabel('Angle (degrees)');
ylabel('Detector position');
title('Sinogram of Phantom');

%% Build system matrix A
% A maps image vector x to projection vector b
%
% For each pixel basis image, compute its Radon transform
% and store as one column of A

disp('Building system matrix A... this may take some time.');

num_detectors = size(R,1);
num_angles = length(theta);
M = num_detectors * num_angles;     % number of equations
P = N * N;                          % number of unknowns

A = zeros(M, P);

for j = 1:P
    basis_img = zeros(N, N);
    basis_img(j) = 1;   % pixel basis

    proj = radon(basis_img, theta);
    A(:, j) = proj(:);
end

disp('System matrix built.');

%% Measured projection vector
b = R(:);

%% Initialize reconstruction
x = zeros(P, 1);

%% Precompute row norms for stability
row_norms = sum(A.^2, 2);
row_norms(row_norms == 0) = eps;

%% ART iterations
disp('Running ART reconstruction...');

for it = 1:num_iter
    for i = 1:M
        ai = A(i, :);                 % i-th row of A
        projection_est = ai * x;      % current estimate for ray i
        error_i = b(i) - projection_est;

        % ART update
        x = x + lambda * (error_i / row_norms(i)) * ai';
    end

    % Display progress every iteration
    recon_img = reshape(x, [N, N]);
    recon_img = max(recon_img, 0);    % optional non-negativity

    figure(100);
    imshow(recon_img, []);
    title(['ART Reconstruction - Iteration ', num2str(it)]);
    drawnow;
end

%% Final reconstruction
recon_art = reshape(x, [N, N]);
recon_art = max(recon_art, 0);

figure;
imshow(recon_art, []);
title('Final ART Reconstruction');

%% Compare with FBP
fbp = iradon(R, theta, 'linear', 'Ram-Lak', 1, N);

figure;
subplot(1,3,1);
imshow(img_true, []);
title('Original Phantom');

subplot(1,3,2);
imshow(fbp, []);
title('FBP Reconstruction');

subplot(1,3,3);
imshow(recon_art, []);
title('ART Reconstruction');