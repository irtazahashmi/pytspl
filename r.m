% this is the script for chebyshev implementation of the simplicial filters

b1 = [-1 -1 -1 0 0 0 0 0 0 0; ...
          1 0 0 -1 0 0 0 0 0 0; ...
          0 1 0 1 -1 -1 0 0 0 0; ...
          0 0 1 0 1 0 -1 0 0 0; ...
          0 0 0 0 0 0 1 -1 -1 0; ...
          0 0 0 0 0 1 0 1 0 -1; ...
          0 0 0 0 0 0 0 0 1 1];

b2 = zeros(10, 3);
b2(1, 1) = 1; b2(2, 1) = -1; b2(4, 1) = 1;
b2(2, 2) = 1; b2(3, 2) = -1; b2(5, 2) = 1;
b2(8, 3) = 1; b2(9, 3) = -1; b2(10, 3) = 1;

L1l = b1' * b1; L1u = b2 * b2'; L1 = L1l + L1u;
%% compute the eigendecomposition
[U, Lam] = eig(L1); Lam = diag(Lam);
Lam(Lam(:) < 1e-3) = 0; lam = uniquetol(Lam, 0.06);

% the harmonic space
a_h = [1; zeros(9, 1)]
U_H = U(:, (a_h == 1)); lam_h = Lam(a_h == 1);
% % the gradient space
[Ul, Lam_l] = eig(L1l); Lam_l = diag(Lam_l);
Lam_l(Lam_l(:) < 1e-3) = 0; lam_l = uniquetol(Lam_l, 0.03);
a_g = [0; 1; 0; 1; 0; 1; 1; 0; 1; 1];
U_G = U(:, (a_g == 1)); lam_g = Lam(a_g == 1);
% % % the curl space
% [Uu,Lam_u] = eig(L1u); Lam_u = diag(Lam_u);
% Lam_u(Lam_u(:)<1e-3) = 0; lam_u = uniquetol(Lam_u,0.06);
% a_c = [0; 0; 1; 0; 1; 0; 0; 1; 0; 0]
% U_C = Uu(:,(a_c==1)); lam_c = Lam_u(a_c==1);

%% flow generation
f = U * 1 * ones(10, 1);

% % analyze its frequency component
f_h_tilde = U_H' * f; f_h = U_H * f_h_tilde;
f_g_tilde = U_G' * f; f_g = U_G * f_g_tilde;
% f_c_tilde = U_C'*f; f_c = U_C*f_c_tilde;
% f_tilde = U'*f;

% % check the divergence and curl
% div = B1*f; curl = B2t*f;
% f - (f_h + f_g + f_c);

% % projection operator
% N_G = 545; N_C = 431; N_H = 112;
% norm(B1'*B1 - U_G*diag(lam_g)*U_G')
% P_g = B1'*pinv(B1*B1')*B1; P_g = U_G*U_G'; trace(P_g) == N_G;
% f_g_proj = (B1'*pinv(B1*B1')*B1)*f;
% f_c_proj = (B2*pinv(B2'*B2)*B2')*f;
% f_h_proj = (eye(num_edges) - L1*pinv(L1))*f;

%% use the chebyshev implementation to construct the filter to perform the
% subcomponent extraction

% build the continuous subcomponent extraction function
lam_g_cut = 0.01; % cut off frequency gradient
steep = 100;
% a logistic function to perform the gradeint component extraction
g_g = @(lam) 1 ./ (1 + exp(-steep .* (lam - lam_g_cut)));
% g_g = 'sign(x)'; % if we consider a sign function approximated by
% Chebyshev polynomial, the performance is really bad when the filter order
% is small

% estimate the largest eigenvalue
% power iteration algorithm
v = ones(size(L1l, 1), 1);

for k = 1:50
    v = L1l * v;
    v = v / norm(v);
end

lam_g_max = mean((L1l * v) ./ v);

% plot the continuous extraction function
%figure; plot(0:0.01:lam_g_max, g_g(0:0.01:lam_g_max));
% obtain the chebyshev approx and coefficients
g_g_cheb = chebfun(g_g, [0, lam_g_max], 100);
coeff_g = chebcoeffs(g_g_cheb);

lam_in = Lam_l;
% plot the continuous frequency response on the gradeint eigenvalues and
% the chebyshev approxi. one
figure;
scatter(lam_in, g_g(lam_in)); hold on;
scatter(lam_in, g_g_cheb(lam_in)); hold on;

alpha_g = lam_g_max / 2;
in = (lam_in - alpha_g) ./ alpha_g;
cheb_approx(in, coeff_g);

% the truncated order
K_trnc = 5:5:100;
% the ideal frequency response
num_edges = 10
h_g_ideal = zeros(num_edges, 1);
%h_g_ideal(544:end) = 1;
h_g_ideal = a_g
H_g_ideal = Ul * diag(h_g_ideal) * Ul';
norm(H_g_ideal - U_G * U_G'); % diff of the ideal filter and the grad projector
scatter(lam_in, h_g_ideal); hold on;

for i = 1:length(K_trnc)
    H_g_cheb_approx_out(i, :, :) = filter_cheb_approx(L1l, coeff_g, alpha_g, K_trnc(i));
end

figure;
% plot the true and estimated gradient component in the frequency domain
scatter(Lam, U' * f_g); hold on;

for i = 1:length(K_trnc)
    g_g_cheb_approx_out = diag(Ul' * squeeze(H_g_cheb_approx_out(i, :, :)) * Ul);
    % compute the error w.r.t. the true gradient extraction filter
    err_g_response(i) = norm(g_g_cheb_approx_out - h_g_ideal) / norm(h_g_ideal);
    err_filter_g(i) = norm(squeeze(H_g_cheb_approx_out(i, :, :)) - H_g_ideal, 2);

    f_g_cheb(:, i) = squeeze(H_g_cheb_approx_out(i, :, :)) * f;
    % compute the SFT of the approx.ed component
    f_g_cheb_tilde(:, i) = U' * f_g_cheb(:, i);
    err_g_tilde(i) = norm(f_g_cheb_tilde(:, i) - U' * f_g) / norm(U' * f_g);
    err_g(i) = norm(f_g_cheb(:, i) - f_g) / norm(f_g);
end

scatter(Lam, f_g_cheb_tilde(:, i)); hold on;
% identity term coefficient
sum(coeff_g(1:2:end)) - sum(coeff_g(2:2:end));
%
figure;
% plot the 8th one, of order 40
H_cheb_40 = squeeze(H_g_cheb_approx_out(8, :, :));
c3 = [0.4660 0.6740 0.1880];
scatter(Lam_l, diag(Ul' * H_cheb_40 * Ul), 'filled', 'MarkerFaceColor', c3, ...
    'MarkerEdgeColor', c3); hold on;
%% spectral design with approximations
L = 10;
Phi_G = [];
Phi_G = [Phi_G lam_l .^ (0:1:L - 1)];
% solve the LS problem
% for a gradient preserving filter, the parameter h = 1, preserving the rest
% alpha = pinv(0.2*eye(L)+Phi_G)*[1;zeros(length(lam_l)-1,1)];
alpha = pinv(Phi_G) * [0; ones(length(lam_l) - 1, 1)];
% build the filter
H1 = zeros(num_edges);

for l = 1:L
    H1 = H1 + alpha(l) * L1l ^ (l - 1);
end

h_h1 = diag(U_H' * H1 * U_H);
h_g1 = diag(U_G' * H1 * U_G);
h_c1 = diag(U_C' * H1 * U_C);
h_1 = diag(Ul' * H1 * Ul);

f_filtered1 = H1 * f;
err_1_filter = norm(H1 - H_g_ideal, 2);
err_1 = norm(f_filtered1 - f_g) / norm(f_g);
% analyze its frequency component
f_h_tilde_o1 = U_H' * f_filtered1;
f_g_tilde_o1 = U_G' * f_filtered1;
f_c_tilde_o1 = U_C' * f_filtered1;
f_tilde_o1 = U' * f_filtered1;
f_tilde_o1_rest = U' * (f - f_filtered1);

c1 = [0 0.4470 0.7410];
scatter(Lam_l, h_1, 'filled', 'MarkerFaceColor', c1, ...
    'MarkerEdgeColor', c1);
%% consider the universal design

% grid into 100 points
num_universal = 100;
lam_G_universal = [linspace(0, lam_g_max, num_universal)]';
L = 10;
Phi_G = lam_G_universal .^ (0:1:L - 1);
g_uni = g_g(lam_G_universal);

% compute the filter coefficient
alpha = pinv(Phi_G) * g_uni;

% build the filter
H_uni = zeros(num_edges);

for l = 1:L
    H_uni = H_uni + alpha(l) * L1l ^ (l - 1);
end

h_h = diag(U_H' * H_uni * U_H);
h_g_uni = diag(U_G' * H_uni * U_G);
h_c = diag(U_C' * H_uni * U_C);
h_uni = diag(Ul' * H_uni * Ul);
f_g_uni = H_uni * f;
% analyze its frequency component
f_h_tilde_o = U_H' * f_g_uni;
f_g_tilde_o = U_G' * f_g_uni;
f_c_tilde_o = U_C' * f_g_uni;

err_uni_filter = norm(H_uni - H_g_ideal, 2);
err_uni = norm(f_g_uni - f_g) / norm(f_g);
hold on;

c2 = [0.6350 0.0780 0.1840];
scatter(Lam_l, h_uni, 'filled', 'MarkerFaceColor', c2, ...
    'MarkerEdgeColor', c2);
xlim([0 lam_g_max])
ylim([0 1.2]);

grid on;
set(gca, 'fontsize', 14)
set(gca, 'box', 'on')
legend('Chebyshev of order 40', 'Approx. spectral design of order 10', ...
'Universal design of order 10 with 100 samples')

%% plot the extracted gradient component in the frequency domain
make_it_tight = true;
subplot = @(m, n, p) subtightplot (m, n, p, [0.00 0.05], [0.13 0.03], [0.05 0.02]);
if ~make_it_tight, clear subplot; end

figure;
subplot(1, 2, 1);
% only check the first 200 frequencies, as they are quite hard to obtain a
% frequency response of 1
num_check = 50; % upto freq 1
scatter(lam_g(1:num_check), f_g_tilde(1:num_check), 100, 'd', 'LineWidth', 1.5); hold on;

%scatter(lam_g(1:num_check),f_g_tilde_o1(1:num_check),50,'filled'); hold on; % this is the spectral design result

scatter(lam_g(1:num_check), f_g_tilde_o(1:num_check), 80, c2, '+'); hold on; % this is the universal design result

% compute the SFT of the approx.ed component
f_g_cheb_tilde_8th = U_G' * (H_cheb_40 * f);
scatter(lam_g(1:num_check), f_g_cheb_tilde_8th(1:num_check), 50, c3, 'filled'); hold on; % this is the chebyshev result
set(gca, 'YScale', 'log')
grid on;
set(gca, 'fontsize', 14)
legend('true grad comp', 'grad comp (universal)', 'grad comp (Chebyshev)')
xlabel('Frequency')
ylim([1e2, 3e4])

% plot the error of the chebyshev design
subplot(1, 2, 2);
plot(K_trnc(1:10), err_g(1:10), 'LineWidth', 2);
hold on;
plot(K_trnc(1:10), err_g_response(1:10), 'LineWidth', 2);
set(gca, 'YScale', 'log')
grid on;
set(gca, 'fontsize', 14)
xlabel('Order')
legend('$$||(\mathbf{P}_{\rm G}-\mathbf{H}_{1,\rm c}||_2)/||\mathbf{P}_{\rm G}||_2$$', ...
    '$$||(\mathbf{f}_{\rm G}-\mathbf{H}_{1,\rm c}\mathbf{f}||_2)/||\mathbf{f}_{\rm G}||_2$$', 'interpreter', 'latex')
