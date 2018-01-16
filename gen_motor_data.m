% Using file to generate  motor data:
function out_data = gen_motor_data(entry)
    data = load('data.mat');
    y_data = data(1).data(entry, :);
    t_data = 0:1/5000:5-1/5000;
    out_data = [y_data; t_data];
end
