% Liuzhaoming
% 新松重载机器人SR210A补偿代码
%

% 加入文件夹路径，启动GPML
addpath(genpath(pwd));
startup

% 输入我们的数据集
SR210A_position = readtable('dataset\SR210A\50点激光跟踪仪采集姿态值.xlsx');
SR210A_joint = readtable("dataset\SR210A\50点关节值.xlsx");
SR210A_mean = readtable("dataset\SR210A\50点姿态值.xlsx");

% 设置GP模型
meanfunc_x = SR210A_position.X;      % 用理论计算值作为均值函数
covfunc_x = @covSEiso;               % 使用 Squared Exponential 协方差函数
likfunc_x = @likGauss;               % 高斯似然函数
hyp.cov_x = [0; 0];                  % 初始超参数（log(ell), log(sf))
hyp.lik_x = log(0.1);                % 初始噪声参数 log(sn)
hyp = struct('mean', meanfunc_x, 'cov', covfunc_x, 'lik', hyp.lik_x);

% 训练模型
hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc_x, covfunc_x, likfunc_x, SR210A_joint, SR210A_position.X);
