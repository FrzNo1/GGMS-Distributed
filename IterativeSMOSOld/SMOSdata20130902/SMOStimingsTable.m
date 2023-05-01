filedate = '20130902';
fileprefix = 'Summary';
type=cell(3,1); type{1}='F'; type{2}='D'; type{3}='U';
typstr=cell(3,1); typstr{1}='Floats'; typstr{2}='Doubles'; typstr{3}='Uints';
vec=cell(4,1); vec{1}='U'; vec{2}='N'; vec{3}='H'; vec{4}='C';
vecstr=cell(4,1); vecstr{1}='Uniform'; vecstr{2}='Normal'; vecstr{3}='Half Normal'; vecstr{4}='Cauchy';
OS=cell(5,1); OS{1}='U'; OS{2}='R'; OS{3}='N'; OS{4}='C'; OS{5}='S';

p=24:2:28;
nlist=2.^p;
OSlist=[5, 11, 101, 1001];
OSdistr=OS{1};

data=cell(3,4);

for v=1:4
    vecdistr=vec{v};
    switch v
        case 1
            for t=1:3
                vectype=type{t};
                filesuffix = [vectype vecdistr OSdistr filedate];
                fname = [fileprefix filesuffix '.csv'];
                temp=csvread(fname);
                temp=temp(ismember(temp(:,1),nlist),:);
                temp=temp(ismember(temp(:,2),OSlist),:);
                data{t,v}=temp;
                [t v], size(data{t,v})
            end
        case 2
            for t=1:2
                vectype=type{t};
                filesuffix = [vectype vecdistr OSdistr];
                fname = [fileprefix filesuffix '.csv'];
                temp=csvread(fname);
                temp=temp(ismember(temp(:,1),nlist),:);
                temp=temp(ismember(temp(:,2),OSlist),:);
                data{t,v}=temp;
                [t v], size(data{t,v})
            end
        otherwise
            t=1;
            vectype=type{t};
            filesuffix = [vectype vecdistr OSdistr];
            fname = [fileprefix filesuffix '.csv'];
            temp=csvread(fname);
            temp=temp(ismember(temp(:,1),nlist),:);
            temp=temp(ismember(temp(:,2),OSlist),:);
            data{t,v}=temp;
                [t v], size(data{t,v})
    end
end
    






fprintf('\n\nLatex table....\n\n');
fprintf('\\begin{table}\\centering\n');
fprintf(['\\tbl{Mean timings and acceleration ratios for selecting quartiles, deciles, percentiles, and $1/10$-percentiles, C2070.\\label{tab:timings}}{%%\n']);
fprintf('\\begin{tabular}{c} \n');
fprintf('\\begin{tabular}{||c|r||ccc|ccc||}\\hline\n');
fprintf('  \\multicolumn{2}{||r||}{Vector Type} & \\multicolumn{6}{|c||}{Float}  \\\\ \n');
fprintf('\\hline\n');
fprintf('  \\multicolumn{2}{||r||}{Vector Distribution} & \\multicolumn{3}{|c|}{Uniform} & \\multicolumn{3}{|c||}{Normal} \\\\ \n');
fprintf('\\hline\n');
fprintf('length & \\#OS  & \\texttt{bMS} & \\texttt{s\\&c} & $\\frac{\\texttt{s\\&c}}{\\texttt{bMS}}$ & \\texttt{bMS} & \\texttt{s\\&c} & $\\frac{\\texttt{s\\&c}}{\\texttt{bMS}}$ \\\\ \n');
fprintf('\\hline\n');




    for i=1:length(nlist)*length(OSlist)
        fprintf('$2^{%d}$ & %d  & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f  \\\\ \n', ...
            log2(data{1,1}(i,1)), data{1,1}(i,2), data{1,1}(i,7), data{1,1}(i,3), data{1,1}(i,3)/data{1,1}(i,7), ...
            data{1,2}(i,7), data{1,2}(i,3), data{1,2}(i,3)/data{1,2}(i,7));
    end

fprintf('\\hline \n');
fprintf('\\end{tabular}  \\\\ \n');
fprintf('\\begin{tabular}{||c|r||ccc|ccc||}\\hline\n');
fprintf('  \\multicolumn{2}{||r||}{Vector Distribution} & \\multicolumn{3}{|c|}{Half Normal} & \\multicolumn{3}{|c||}{Cauchy} \\\\ \n');
fprintf('\\hline\n');
fprintf('length & \\#OS  & \\texttt{bMS} & \\texttt{s\\&c} & $\\frac{\\texttt{s\\&c}}{\\texttt{bMS}}$ & \\texttt{bMS} & \\texttt{s\\&c} & $\\frac{\\texttt{s\\&c}}{\\texttt{bMS}}$  \\\\ \n');
fprintf('\\hline\n');


    for i=1:length(nlist)*length(OSlist)
        fprintf('$2^{%d}$ & %d  & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f  \\\\ \n', ...
            log2(data{1,3}(i,1)), data{1,3}(i,2), data{1,3}(i,7), data{1,3}(i,3), data{1,3}(i,3)/data{1,3}(i,7), ...
            data{1,4}(i,7), data{1,4}(i,3), data{1,4}(i,3)/data{1,4}(i,7));
    end
    
    
fprintf('\\hline \n');
fprintf('\\end{tabular} \\\\ \n');

fprintf(' \\\\ \n');
    
fprintf('\\begin{tabular}{||c|r||ccc|ccc||}\\hline\n');
fprintf('  \\multicolumn{2}{||r||}{Vector Type} & \\multicolumn{6}{|c||}{Double}  \\\\ \n');
fprintf('\\hline\n');
fprintf('  \\multicolumn{2}{||r||}{Vector Distribution} & \\multicolumn{3}{|c|}{Uniform} & \\multicolumn{3}{|c||}{Normal} \\\\ \n');
fprintf('\\hline\n');
fprintf('length & \\#OS  & \\texttt{bMS} & \\texttt{s\\&c} & $\\frac{\\texttt{s\\&c}}{\\texttt{bMS}}$ & \\texttt{bMS} & \\texttt{s\\&c} & $\\frac{\\texttt{s\\&c}}{\\texttt{bMS}}$  \\\\ \n');
fprintf('\\hline\n');


    for i=1:length(nlist)*length(OSlist)
        fprintf('$2^{%d}$ & %d  & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f & %0.2f  \\\\ \n', ...
            log2(data{2,1}(i,1)), data{2,1}(i,2), data{2,1}(i,7), data{2,1}(i,3), data{2,1}(i,3)/data{2,1}(i,7), ...
            data{2,2}(i,7), data{2,2}(i,3), data{2,2}(i,3)/data{2,2}(i,7));
    end
    
    
fprintf('\\hline \n');
fprintf('\\end{tabular} \\\\ \n');
    
fprintf(' \\\\ \n');

fprintf('\\begin{tabular}{||c|r||ccc||}\\hline\n');
fprintf('  \\multicolumn{2}{||r||}{Vector Type} & \\multicolumn{3}{|c||}{Unsigned Integers}  \\\\ \n');
fprintf('\\hline\n');
fprintf('  \\multicolumn{2}{||r||}{Vector Distribution} & \\multicolumn{3}{|c||}{Uniform}  \\\\ \n');
fprintf('\\hline\n');
fprintf('length & \\#OS  & \\texttt{bMS} & \\texttt{s\\&c} & $\\frac{\\texttt{s\\&c}}{\\texttt{bMS}}$  \\\\ \n');
fprintf('\\hline\n');




    for i=1:length(nlist)*length(OSlist)
        fprintf('$2^{%d}$ & %d  & %0.2f & %0.2f & %0.2f  \\\\ \n', ...
            log2(data{3,1}(i,1)), data{3,1}(i,2), data{3,1}(i,7), data{3,1}(i,3), data{3,1}(i,3)/data{3,1}(i,7));
    end
    
    
fprintf('\\hline \n');
fprintf('\\end{tabular}\n');
fprintf('\\end{tabular}}\n');
fprintf('\\end{table}\n');