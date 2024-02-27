% clear all; close all;
% to use topological filter to extract the sub-component from a given
% unknonw edge flow, then do comparison between different approaches and
% different filter form and different lengths of filter orders

% the code can be used also for denoising, for instance, the ideal low-pass
% filter in edge space would be the harmonic component extraction, or in
% the non-ideal case, the low-pass filter should
%% read me
% 1. run code section by section
% 2. change the compared component and the line setting

%% build the topology, i.e., B1 and B2 incidence matrix
b1 = [-1 -1 -1 0 0 0 0 0 0 0;...
       1 0 0 -1 0 0 0 0 0 0 ;...
       0 1 0 1 -1 -1 0 0 0 0;...
       0 0 1 0 1 0 -1 0 0 0 ;...
       0 0 0 0 0 0 1 -1 -1 0;...
       0 0 0 0 0 1 0 1 0 -1;...
       0 0 0 0 0 0 0 0 1 1];

b2 = zeros(10,3);
b2(1,1) = 1; b2(2,1) = -1; b2(4,1) = 1;
b2(2,2) = 1; b2(3,2) = -1; b2(5,2) = 1;
b2(8,3) = 1; b2(9,3) = -1; b2(10,3) = 1;

% build hodge laplacian matrix
Le = b1'*b1; L1u = b2*b2'; L1 = Le + L1u;

% eigendecomposition of laplacians
[u1l ~]  = eig(Le); eig_Le = eig(Le);
[u1 ~]= eig(L1); eig_L1 = eig(L1);
[u1u ~]= eig(b2*b2'); eig_L1u = eig(L1u);

% generate an edge from the spectral domain which has all components with 1
f = u1 * 1*ones(10,1);
f_tilde = u1' * f; % perform fourier transform

norm_f = norm(f)

%% add some noise for robustness
% noise = randn(10,1);
% f = f + 0.5*noise;
% f_tilde = u1' * f; % perform fourier transform
% n_tilde = u1' * noise
% err_ori = norm(f-u1 * 1*ones(10,1))
%% generate the pure component from spectral domain for computing the error
% pure harmonic component
f_h = u1 * [1;zeros(9,1)];
% pure gradient component
f_g = u1 * [0;1;0;1;0;1;1;0;1;1];
% pure curl component
f_c = u1 * [0;0;1;0;1;0;0;1;0;0];

%% regularization method for harmonic and least squares method for grad & curl
% the regularization method for extracting the certain component, i.e.,
% three regularizers
I = eye(10);
%mu1 = 0.5; mu2 = 0.5;
mu3 = 5;
% least squares based grad extraction
H_g_r = b1' * pinv(b1*b1') * b1; h_g_r = u1'*H_g_r*u1;
f_g_r = H_g_r * f;
err_g_r = norm(f_g_r-f_g)/norm(f_g)

% least squares based curl extraction
H_c_r = b2 * pinv(b2'*b2) * b2'; h_c_r = u1'*H_c_r*u1;
f_c_r = H_c_r * f;
err_c_r = norm(f_c_r-f_c)/norm(f_c)

% harmonic component, can be done by deletion or regularization
f_h_r = f - f_g_r - f_c_r;
f_h_r2 = (I + mu3*L1)\f; % regularization is not a good strategy
err_h_r = norm(f_h_r - f_h)/norm(f_h)
err_h_r2 = norm(f_h_r2 - f_h)/norm(f_h)

%% subcomponent extraction based on filtering
a_h = [1;zeros(9,1)]; % since we know there is only one harmonic frequency
a_g = [0;1;0;1;0;1;1;0;1;1]; % for extracting the gradient
a_c = [0;0;1;0;1;0;0;1;0;0]; % for extracting the curl
% let the user choose what component to be extracted
addpath /Users/maosheng/Documents/subtightplot
make_it_tight = true;
subplot = @(m,n,p) subtightplot (m, n, p, [0.025 0.00], [0.1 0.02], [0.06 0.02]);
if ~make_it_tight,  clear subplot;  end

comp = input('Enter the component to be extracted (1:grad, 2:curl, 3:harmonic): ');
switch(comp)
    case 1
        a = a_g; f_true = f_g;
        fig = subplot(3,1,1);
    case 2
        a = a_c; f_true = f_c;
        fig = subplot(3,1,2);
    case 3
        a = a_h; f_true = f_h;
        fig = subplot(3,1,3);
end

%% topological filter method with form 1
% initialization
L_set = (1:1:15); % the length of filter
A = []; % the system matrix to be updated
err_tf1 = [];
H_fr = [];
H_fr_leak = [];
for L = L_set-1
    % build the system matrix
    A = [A, eig_L1.^L];
    % rank(A) should be full rank, but if L is too large, then it can be
    % singular
    % from the least squares solution to obtain the filter coefficients
    %     if size(A,1) > size(A,2) % if the system matrix is tall
    %         h = (A'*A)\A'*a;
    %     elseif size(A,1) == size(A,2) % if the system matrix is square
    %         h = A\a;
    %     else % if the system matrix is fat
    %         h = A'*inv(A*A')*a;
    %     end
    h = pinv(A)*a;
    H = zeros(size(L1));
    % build the topological filters
    for l = 1:length(h)
        H = H + h(l) * L1^(l-1);
    end
    % filtering process
    f_est = H * f;
    % compute the error % remember to change the exact component variable
    err_tf1(L+1) = norm(f_est - f_true)/norm(f_true);
    % check the filter frequency response
    H_fr(:,L+1) = diag(u1' * H * u1);
    % calculate the frequency leak
    H_fr_leak(L+1) = norm(H_fr(:,L+1).* not(a));
end


% gradient
% plot(L_set,err_tf1(L_set(1):end),'o-','LineWidth',2,'Color',[0.8500 0.3250 0.0980]);
% curl
% plot(L_set,err_tf1(L_set(1):end),'o-','LineWidth',2,'Color',[0.4660 0.6740 0.1880]);
% harmonic
% plot(L_set,err_tf1(L_set(1):end),'o-','LineWidth',2,'Color',[0 0.4470 0.7410]);
% xlabel('Filter length (L)')
% ylabel('Normalized RMSE')
%title('Curl component extraction')
% grid on;
% set(gca,'fontsize',12)
% xlim([1 12])
%%
% h1 = hgload('./Topological Neural Network/Topological filter/experiments/figures/grad-extract');
% h2 = hgload('./Topological Neural Network/Topological filter/experiments/figures/curl-extract');
% h3 = hgload('./Topological Neural Network/Topological filter/experiments/figures/harmonic-extract');
% h1l = findobj(h1,'Type','line'); y1 = get(h1l,'YData');
% h2l = findobj(h2,'Type','line'); y2 = get(h2l,'YData');
% h3l = findobj(h3,'Type','line'); y3 = get(h3l,'YData');
% h4 = figure;
% plot(L_set,y1,'LineWidth',2); hold on;
% plot(L_set,y2,'LineWidth',2); hold on;
% plot(L_set,y3,'LineWidth',2); hold on;
% legend('Gradient component','Curl component','Harmonic component');
% xlabel('Filter length (L)')
% ylabel('Normalized RMSE')
% title('sub-component extraction')
% grid on;
% set(gca,'fontsize',12)
% xlim([1 12])
%% for eusipco no need for the rest
%% topological filter method with form 2 for gradient and curl
% since there are two sets of filter coefficients, we consider two lengths


L_set1 = 1:8; L_set2 = 1:8;
% set the -inf number as 0 in eigenvalues
eig_Le(eig_Le(:)<1e-6) = 0;
% keep distinct eigenvalues, remove the repeated ones and 0
eig_Le = nonzeros(unique(eig_Le));
eig_L1u(eig_L1u(:)<1e-6) = 0;
eig_L1u = nonzeros(unique(eig_L1u));
% since in this filter form, the eigenvalues are not ordered based on the
% values but on the types, thus we need to rewrite the targeted frequency
% response vectors as follows
% pure grad
a_g2 = ones(length(eig_Le),1); % only the grad frequencies response, not including 0
% pure curl
a_c2 = ones(length(eig_L1u),1); % only the curl frequencies response, not including 0
% pure harmonic
a_h2 = 1;
switch(comp)
    case 1
        a2 = [~a_h2;a_g2;~a_c2];
    case 2
        a2 = [~a_h2;~a_g2;a_c2];
    case 3
        a2 = [a_h2;~a_g2;~a_c2];
end
method_choice = input('Choose the SV filter design method (1:joint, 2:serparate-approx): ');
switch(method_choice)
    case 1
        err_tf2 = []; H_fr2 = []; H_fr2_leak = [];
        Al = [];
        %%
        for l1 = L_set1
            % update the system submatrix
            Al = [Al, eig_Le.^l1];
            Au = [];
            for l2 = L_set2
                % update the system sub-matrix
                Au = [Au, eig_L1u.^l2];
                % build the system matrix
                A2 = [ones(length(eig_L1),1) [zeros(1,size(Au,2)+size(Al,2));...
                    [Al; zeros(size(Au,1),size(Al,2))],...
                    [zeros(size(Al,1),size(Au,2));Au]]];
                % obtain the filter coefficients
                h2 = pinv(A2) * a2;
                % build the filter
                H2 = h2(1)*I;
                for i = 1:l1
                    H2 = H2 + h2(i+1) * Le^(i);
                end
                for i = 1:l2
                    H2 = H2 + h2(i+1+l1) * L1u^(i);
                end
                % filtering
                f_est2 = H2 * f;
                % compute the error
                err_tf2(l1,l2) = norm(f_est2 - f_true)/norm(f_true);
                % check the filter frequency response
                H_fr2(:,l1,l2) = diag(u1' * H2 * u1);
                % calculate the frequency leak
                H_fr2_leak(l1,l2) = norm(H_fr2(:,l1,l2).* not(a2));
            end
        end
    case 2
        %% separate way of solving the joint filter design for SV filter
        err_tf2 = []; H_fr2 = []; H_fr2_leak = [];
        Al = [];
        for l1 = L_set1
            % update the system submatrix
            Al = [Al, eig_Le.^l1];
            Au = [];
            for l2 = L_set2
                % update the system sub-matrix
                Au = [Au, eig_L1u.^l2];
                % look for the solution separately
                h2 = zeros(1+l1+l2,1);
                h2(1) = a2(1);
                h2(2:1+l1) = pinv(Al)*(a2(2:1+length(eig_Le))-a2(1));
                h2(2+l1:1+l1+l2) = pinv(Au)*(a2(2+length(eig_Le):end)-a2(1));
                % build the filter
                H2 = h2(1)*I;
                for i = 1:l1
                    H2 = H2 + h2(i+1) * Le^(i);
                end
                for i = 1:l2
                    H2 = H2 + h2(i+1+l1) * L1u^(i);
                end
                % filtering
                f_est2 = H2 * f;
                % compute the error
                err_tf2(l1,l2) = norm(f_est2 - f_true)/norm(f_true);
                % check the filter frequency response
                H_fr2(:,l1,l2) = diag(u1' * H2 * u1);
                % calculate the frequency leak
                H_fr2_leak(l1,l2) = norm(H_fr2(:,l1,l2).* not(a2));
            end
        end

end




%% in the form 2, for subcomponent extraction, for instance, if we care the gradien
% component, the solution will have zero coefficients on the \beta;
% similarly, the filter coefficients \alpha will be zero for the curl
% extraction. Thus, let us only consider the lower or upper part of the
% filter to do so
% depends on the requirement for grad or curl
% the following code should be updated each time

L_set3 = (1:1:8); % the length of filter
A3 = []; % the system matrix to be updated
err_tf3 = [];
H_fr3 = [];
H_fr_leak3 = [];
eig_Le = eig(Le); eig_Le(eig_Le(:)<1e-6) = 0; eig_Le = unique(eig_Le);
eig_L1u = eig(L1u);eig_L1u(eig_L1u(:)<1e-6) = 0; eig_L1u = unique(eig_L1u);
% the desired frequency response
switch(comp)
    case 1
        a3 = [0;ones(length(eig_Le)-1,1)];
    case 2
        a3 = [0;ones(length(eig_L1u)-1,1)];
    case 3
        %
end
%a3 = [0;ones(length(eig_Le)-1,1)]; % the zero frequency corresponds to the curl and harmonic part
%a3 = [0;ones(length(eig_L1u)-1,1)];% the zero frequency corresponds to the grad and harmonic part
for L = L_set3-1
    % build the system matrix
    switch(comp)
        case 1
            A3 = [A3, eig_Le.^L];
        case 2
            A3 = [A3, eig_L1u.^L];
    end
    h3 = pinv(A3)*a3;
    H3 = zeros(size(L1));
    % build the topological filters
    switch(comp)
        case 1
            for l = 1:length(h3)
                H3 = H3 + h3(l) * Le^(l-1);
            end
        case 2
            for l = 1:length(h3)
                H3 = H3 + h3(l) * L1u^(l-1);
            end
    end
    % filtering process
    f_est3 = H3 * f;
    % compute the error % remember to change the exact component variable
    err_tf3(L+1) = norm(f_est3 - f_true)/norm(f_true);
    % check the filter frequency response
    H_fr3(:,L+1) = diag(u1' * H3 * u1);
    % calculate the frequency leak
    H_fr_leak3(L+1) = norm(H_fr3(:,L+1).* not(a));
end

%% for illustration subcomponents
% f_g = f_est3;
% f_c = f_est3;
% f_h = f - f_g - f_c;
% write three flows in a file through another m.file
%%
% gradient
% plot(L_set3,err_tf3(L_set3(1):end),'o--','LineWidth',2,'Color',[0.8500 0.3250 0.0980]);
% curl
% plot(L_set3,err_tf3(L_set3(1):end),'o--','LineWidth',2,'Color',[0.4660 0.6740 0.1880]);
%%
hold on;
switch(comp)
    case 1
        switch(method_choice)
            case 1
                line1 = plot(L_set-1,err_tf1(L_set(1):end),'o-','LineWidth',2,...
                    'Color',[0.8500 0.3250 0.0980],'MarkerSize',10);
                legend('$$L_1=L_2$$')
            case 2
                %
        end
    case 2
        switch(method_choice)
            case 1
                line1 =  plot(L_set-1,err_tf1(L_set(1):end),'o-','LineWidth',2,...
                    'Color',[0.4660 0.6740 0.1880],'MarkerSize',10);
                legend('$$L_1=L_2$$')
            case 2 
                %
        end
    case 3
        switch(method_choice)
            case 1
                line1 = plot(L_set-1,err_tf1(L_set(1):end),'o-','LineWidth',2,...
                    'Color',[0 0.4470 0.7410],'MarkerSize',10');
                legend('$$L_1=L_2$$')
            case 2
                %
        end
end
hold on
% SV filter with nonzero
switch(comp)
    case 1
        switch(method_choice)
            case 1
                % joint design
                line2 = plot(L_set1,err_tf2(L_set1(1):end,3),'x-.','LineWidth',...
                    2,'Color',[0.8500 0.3250 0.0980],'MarkerSize',10,'DisplayName',['joint','$$(L_2=3)$$']);
                hold on;
            case 2
                % slip design
                line4 = plot(L_set1,err_tf2(L_set1(1):end,3),'d-.','LineWidth',...
                    2,'Color',[0.8500 0.3250 0.0980],'MarkerSize',10,'DisplayName',['decoupled','$$(L_2=3)$$']');
        end
    case 2
        switch(method_choice)
            case 1
                % joint design
                line2 = plot(L_set2,err_tf2(6,L_set2(1):end),'x-.','LineWidth',...
                    2,'Color',[0.4660 0.6740 0.1880],'MarkerSize',10, 'DisplayName',['joint','$$(L_1=6)$$']);
            case 2
                % slip design
                line4 = plot(L_set2,err_tf2(6,L_set2(1):end),'d-.','LineWidth',...
                    2,'Color',[0.4660 0.6740 0.1880],'MarkerSize',10,'DisplayName',['decoupled','$$(L_1=6)$$']);
        end
    case 3
        switch(method_choice)
            case 1
                % joint design
                line2 = plot(L_set1,err_tf2(L_set1(1):end,3),'x-.','LineWidth',...
                    2,'Color',[0 0.4470 0.7410],'MarkerSize',10,'DisplayName',['joint','$$(L_2=3)$$']);
            case 2
                % split design
                line4 = plot(L_set1,err_tf2(L_set1(1):end,3),'d-.','LineWidth',...
                    2,'Color',[0 0.4470 0.7410],'MarkerSize',10,'DisplayName',['decoupled','$$(L_2=3)$$']);

        end
end
% gradient
% plot(L_set1,err_tf2(L_set1(1):end,2),'*-.','LineWidth',2,'Color',[0.8500 0.3250 0.0980]);
% curl
% plot(L_set2,err_tf2(2,L_set2(1):end),'*--','LineWidth',2,'Color',[0.4660 0.6740 0.1880]);
% harmonic
% plot(L_set1,err_tf2(L_set1(1):end,3),'o--','LineWidth',2,'Color',[0 0.4470 0.7410]);

hold on;
% SV filter with only part of the filter
switch(comp)
    case 1
        switch(method_choice)
            case 1
                line3 = plot(L_set3-1,err_tf3(L_set3(1):end),'s--','LineWidth',2,...
                    'Color',[0.8500 0.3250 0.0980],'MarkerSize',10,'DisplayName','$$L_2=0$$');
            case 2
                % nothing
        end
    case 2
        switch(method_choice)
            case 1
                line3 = plot(L_set3-1,err_tf3(L_set3(1):end),'s--','LineWidth',2,...
                    'Color',[0.4660 0.6740 0.1880],'MarkerSize',10,'DisplayName','$$L_1=0$$');
            case 2
                %
        end
    case 3
        % does nothing
end

grid on;
set(gca,'fontsize',14)
set(gca,'box','on')
xlim([1 12])
ylim([0 1])
xticks(L_set)
xlabel(['Filter length ','$$L_1$$', ' or ', '$$L_2$$'],'Interpreter','latex'); 
%ylabel('NRMSE','Interpreter','latex')
%% put legend
% leg = legend('$$ \text{Harmonic comp with} \mathbf{H}_1$$','Grad comp with H_1', 'Curl comp with H_1',...
%    'Grad comp with H_1^{SV} (L_2=0)','Curl comp with H_1^{SV} (L_1=0)');
% legend(['Harmonic comp ','$$\mathbf{H}_1 $$'],...
%     ['Grad comp with ', '$$\mathbf{H}_1 $$'],...
%     ['Curl comp with ', '$$ \mathbf{H}_1 $$'],...
%     ['Grad comp with ','$$ \mathbf{H}_1^{SV}(L_2=0)$$'],...
%     ['Curl comp with ','$$ \mathbf{H}_1^{SV}(L_1=0)$$'],...
%     ['Grad comp with ','$$ \mathbf{H}_1^{SV}(L_2=2)$$'],...
%     ['Curl comp with ','$$ \mathbf{H}_1^{SV}(L_1=2)$$'],...
%     'Interpreter','latex')
% leg = [line1(1) line2(1) line3(1) line4(1)];
% switch(comp)
%     case 1
%         legend(leg,['joint','$$(L_1=L_2,\boldmath{\alpha}=\boldmath{\beta})$$'],...
%             ['joint','$$(L_2=2)$$'],...
%             ['decoupled','$$(L_2=0)$$'],...
%             ['decoupled','$$(L_2=2)$$'],...
%             'Interpreter','latex')
%     case 2
%         legend(leg,['joint','$$(L_1=L_2,\boldmath{\alpha}=\boldmath{\beta})$$'],...
%             ['joint','$$(L_1=2)$$'],...
%             ['decoupled','$$(L_1=0)$$'],...
%             ['decoupled','$$(L_1=2)$$'],...
%             'Interpreter','latex')
%     case 3
%         legend(leg([1 2 4]),['joint','$$(L_1=L_2,\boldmath{\alpha}=\boldmath{\beta})$$'],...
%             ['joint','$$(L_2=3)$$'],...
%             ['deoupled','$$(L_2=3)$$'],...
%             'Interpreter','latex')
% end
%
% 
%% make plots of a gradient preserving frequency response
% figure;
% stem(eig_L1,H_fr(:,end),'o','MarkerSize',10,'LineWidth',2,'Color',[0 0.4470 0.7410]); hold on;
% stem(eig_L1,H_fr2(:,end,end),'d','MarkerSize',10,'LineWidth',2,'Color',[0.4660 0.6740 0.1880]); hold on;
% stem(eig_L1,H_fr3(:,end),'*','MarkerSize',10,'LineWidth',2,'Color',[0.9290 0.6940 0.1250]);
% xlabel('Frequency (\lambda)')
% ylabel('Frequency response')
% ylim([0 1.2])
% grid on
% set(gca,'fontsize',16)
% xticks(eig_L1)
% xtickformat('%.1f')
% legend('$$ \mathbf{H}_1, L = 5 $$',...
%     '$$ \mathbf{H}_1^{SV}, L_1 = 3, L_2 = 2$$',...
%     '$$ \mathbf{H}_1^{SV}, L_1 = 5, L_2 = 0$$',...
%     'Interpreter','latex')