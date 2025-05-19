% 让我们考虑一个简单的一维非线性回归示例，数据受到高斯噪声的干扰。我们的数据集是  
x = gpml_randn(0.8, 20, 1);                 % 20 training inputs
y = sin(3*x) + 0.1*gpml_randn(0.9, 20, 1);  % 20 noisy training targets
xs = linspace(-3, 3, 61)';                  % 61 test inputs 

% 指定均值函数、协方差函数和似然函数
meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

% 初始化超参数结构体
hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

% 通过优化（对数）边际似然来设置超参数。具体步骤如下
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

% 使用这些超参数进行预测
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

% 测试点绘制预测均值以及预测的 95%置信区间和训练数据
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
hold on; plot(xs, mu); plot(x, y, '+')