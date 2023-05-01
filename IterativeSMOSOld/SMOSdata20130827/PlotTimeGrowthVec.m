% PlotTimeGrowthVecDistr
% this function makes the plots depicting the time growth withe the
% distribution of order statistics for floats and doubles.

fileprefix = 'Summary';
type=cell(2,1); type{1}='F'; type{2}='D';
typstr=cell(2,1); typstr{1}='Floats'; typstr{2}='Doubles';
vec=cell(4,1); vec{1}='U'; vec{2}='N'; vec{3}='H'; vec{4}='C';
vecstr=cell(4,1); vecstr{1}='Uniform'; vecstr{2}='Normal'; vecstr{3}='Half Normal'; vecstr{4}='Cauchy';
OS=cell(5,1); OS{1}='U'; OS{2}='R'; OS{3}='N'; OS{4}='C'; OS{5}='S';
bms='bucketMultiselect';
sc='sort&choose';
legtxt=cell(4,1);
% legtxt{5}=['Uniform - ' bms]; 
% legtxt{4}=['Uniform Random - ' bms]; 
% legtxt{3}=['Normal Random - ' bms]; 
% legtxt{2}=['Clustered - ' bms]; 
% legtxt{1}=['Sectioned - ' bms]; 
n=2^26;
OSlist=100:10:500;
%clist='rgbcm';


for v=1:2
    figure(v)
    hold off
    titlestr=sprintf('Uniformly Spaced Order Statistics, Vector distribution: %s, n=2^{26}', vecstr{v});
    pname=['TimeGrowthVecDist' vec{v} '.pdf'];
    for t=1:2
        filesuffix = [type{t} vec{v} OS{1}];
        fname = [fileprefix filesuffix '.csv'];
        data=csvread(fname);
        data=data((data(:,1)==n),:);
        data=data(ismember(data(:,2),OSlist),:);
        if strcmp(type{t},'F')
            line='--';
        else line='-';
        end
        plot(data(:,2), data(:,3), [line 'k.'], 'MarkerFaceColor', 'k', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{2*(t-1)+1}=[typstr{t} ' - ' sc];
        hold on
        plot(data(:,2), data(:,7), [line 'rs'], 'MarkerFaceColor', 'r', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{2*(t-1)+2}=[typstr{t} ' - ' bms];
    end
    legend(legtxt{1},legtxt{2},legtxt{3},legtxt{4});
    xlabel('number of order statistics','fontsize',14);
    ylabel('milliseconds','fontsize',14);

    axis([100 500 0 525]);
    %set(gca,'XTick',21:26)

    title(titlestr,'fontsize',14);
    print('-dpdf',pname);
end


for v=3:4
    figure(v)
    hold off
    titlestr=sprintf('Uniformly Spaced Order Statistics, Vector distribution: %s, n=2^{26}', vecstr{v});
    pname=['TimeGrowthVecDist' vec{v} '.pdf'];
        filesuffix = [type{1} vec{v} OS{1}];
        fname = [fileprefix filesuffix '.csv'];
        data=csvread(fname);
        data=data((data(:,1)==n),:);
        data=data(ismember(data(:,2),OSlist),:);
        line='--';
        plot(data(:,2), data(:,3), [line 'k.'], 'MarkerFaceColor', 'k', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{1}=[typstr{1} ' - ' sc];
        hold on
        plot(data(:,2), data(:,7), [line 'rs'], 'MarkerFaceColor', 'r', 'LineWidth', 2, 'MarkerSize', 2)
        legtxt{2}=[typstr{1} ' - ' bms];
    legend(legtxt{1},legtxt{2});
    xlabel('number of order statistics','fontsize',14);
    ylabel('milliseconds','fontsize',14);

    %axis([100 500 0 525]);
    %set(gca,'XTick',21:26)

    title(titlestr,'fontsize',14);
    print('-dpdf',pname);
end


