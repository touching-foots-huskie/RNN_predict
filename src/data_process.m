% this file is going to transfer the data into 10 useful mat:
load('data.mat');
f = 5000;
N = size(data, 1);
T = size(data, 2)/f;
t = 0:1/5000:T-1/5000;
for i=1:1:N
    this_data = data(i, :);
    this_data = [t;this_data]';
    save(sprintf("data_%d.mat",i), 'this_data');
end