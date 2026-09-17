"""以真实汇总和逐模型记录绘制简洁矢量图，不新增评分或推断。"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from matplotlib.colors import LinearSegmentedColormap
import pymupdf
from matplotlib.text import Text


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'derived/figures'
QA = OUT / 'qa'
QA.mkdir(parents=True, exist_ok=True)
COL = {'B': '#2878A0', 'GAP': '#CC7755', 'SupCon-in': '#7C858D'}
INK = '#222222'
GRAY = '#70777D'
RULE = '#D7DCE0'
PALE = '#E7EFF4'
MARK = {'B': 'o', 'GAP': 's', 'SupCon-in': '^'}
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 8,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .6,
    'axes.spines.right': False, 'axes.spines.top': False, 'legend.frameon': False,
    'axes.unicode_minus': False, 'text.color': INK, 'axes.labelcolor': INK})

def rows(name):
    with (ROOT / 'tables' / name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def save(fig, name):
    """固定最终物理尺寸，保存可编辑及路径化矢量，位图仅供目视检查。"""
    # 仅变更展示标签，保留CSV方法键与历史权重命名。
    if name == 'Fig3_holstein_main':
        fig.subplots_adjust(left=.19)
        for ax in fig.axes:
            ax.set_yticks([2, 1, 0], labels=['SupCon-out', 'GAP', 'SupCon-in'])
    for label in fig.findobj(match=Text):
        label.set_text(re.sub(r'(?<![A-Za-z0-9])B(?:-selected)?(?![A-Za-z0-9])', 'SupCon-out', label.get_text()))
    fig.canvas.draw()
    # 公开版本不依赖本地技能插件；保持图形定义，保留矢量对象检查。
    fig.savefig(OUT / (name + '.pdf'))
    fig.savefig(OUT / (name + '.svg'))
    with mpl.rc_context({'svg.fonttype': 'path'}):
        fig.savefig(OUT / (name + '.word.svg'))
    fig.savefig(QA / (name + '.png'), dpi=300)
    with pymupdf.open(OUT / (name + '.pdf')) as pdf:
        assert all(not p.get_images() for p in pdf)
    assert '<image' not in (OUT / (name + '.svg')).read_text(encoding='utf-8')
    plt.close(fig)

def text(ax, x, y, s, size=8, color=INK, bold=False, ha='left'):
    return ax.text(x, y, s, fontsize=size, color=color, weight='bold' if bold else 'normal',
                   ha=ha, va='center', linespacing=1.5)

def arrow(ax, x, y, xx, yy, color=GRAY, dashed=False):
    ax.add_patch(FancyArrowPatch((x,y), (xx,yy), arrowstyle='-|>', mutation_scale=8,
                                lw=.7, color=color, linestyle='--' if dashed else '-'))

def canvas(height):
    fig = plt.figure(figsize=(6.25, height))
    ax = fig.add_axes([.025,.025,.95,.95])
    ax.set(xlim=(0,100), ylim=(0,100)); ax.axis('off')
    return fig, ax

def block(ax, x, y, width, height, label, color=COL['B']):
    ax.add_patch(Rectangle((x,y), width,height,facecolor='white',edgecolor=color,lw=.8))
    text(ax,x+width/2,y+height/2,label,color=color,ha='center')

def protocol():
    """使用实际五折三种子网格及外部支持预留规则组织协议图。"""
    fig,ax=canvas(3.5)
    text(ax,0,96,'a',10,bold=True); text(ax,4,96,'源域评测',9,bold=True)
    text(ax,0,84,'Holstein',11,bold=True)
    text(ax,0,74,'324 个评测身份',8.5)
    text(ax,0,65,'RGB · 224 × 224',8,color=GRAY)
    arrow(ax,28,76,37,76)
    text(ax,49,85,'5 折 × 3 种子',8.5,ha='center')
    for j in range(3):
        for i in range(5):
            ax.add_patch(Rectangle((39+i*4.2,69+j*3.8),3.2,2.6,fc=COL['B'],ec='none',alpha=.85))
    text(ax,49,63,'每方法 15 个模型',8,ha='center',color=GRAY)
    arrow(ax,62,76,70,76)
    text(ax,75,84,'完整候选库',9,bold=True)
    text(ax,75,74,'64 或 65 个身份',8.5)
    text(ax,75,65,'Rank-1 / Rank-5',8,color=GRAY)
    ax.plot([0,100],[54,54],color=RULE,lw=.6)
    text(ax,0,46,'b',10,bold=True);text(ax,4,46,'外部探索性分析',9,bold=True)
    text(ax,0,34,'SideViewCows2026',10,bold=True)
    text(ax,0,24,'607 图 / 63 身份',8.5)
    text(ax,0,14,'每身份至少 6 图',8,color=GRAY)
    arrow(ax,30,26,37,26)
    text(ax,40,35,'577 图 / 54 身份',9,bold=True)
    text(ax,40,25,'270 支持池 + 307 查询',8)
    text(ax,40,14,'K = 1、3、5；5 次抽样',8,color=GRAY)
    arrow(ax,71,26,77,26)
    text(ax,79,35,'冻结评分',9,bold=True)
    text(ax,79,25,'54 个候选身份',8)
    text(ax,79,14,'不更新模型参数',8,color=GRAY)
    ax.plot([60,60,92],[63,57,57],color=COL['B'],lw=.7,ls='--')
    arrow(ax,92,57,92,40,dashed=True,color=COL['B'])
    text(ax,75,47,'权重迁移',7.5,color=COL['B'])
    save(fig,'Fig1_study_protocol')

def architecture():
    """以非数值结构符号说明特征提取、三类聚合和共同评分。"""
    fig,ax=canvas(3.4)
    text(ax,0,97,'a',10,bold=True);text(ax,4,97,'共享编码结构，分别训练权重',9,bold=True)
    for off in [2,1,0]:
        ax.add_patch(Rectangle((1+off,77+off),10,11,fc='white',ec=GRAY,lw=.6))
    text(ax,6,82.5,'RGB',8,ha='center');text(ax,7,71,'224 × 224',7.5,ha='center')
    arrow(ax,15,83,22,83)
    for x,w,h in [(24,4,14),(30,4,11),(36,4,8)]:
        ax.add_patch(Rectangle((x,83-h/2),w,h,fc=PALE,ec=COL['B'],lw=.6))
        ax.add_patch(Polygon([[x+w,83-h/2],[x+w+1.5,84.5-h/2],[x+w+1.5,84.5+h/2],[x+w,83+h/2]],fc='#C2D5E2',ec=COL['B'],lw=.5))
    text(ax,33,71,'DenseNet-121',8,ha='center')
    arrow(ax,43,83,51,83)
    block(ax,53,78,23,10,'ReLU + 空间均值')
    arrow(ax,78,83,85,83)
    for j in range(6):
        ax.add_patch(Rectangle((87+j*1.6,80),1.1,6,fc=COL['B'],ec='none'))
    text(ax,92,71,'L2 归一化',8,ha='center')
    ax.plot([0,100],[63,63],color=RULE,lw=.6)
    text(ax,0,57,'b',10,bold=True);text(ax,4,57,'正例聚合次序',9,bold=True)
    for x,method,operation,sub in [(0,'B','负对数 → 正例均值','受限 SupCon-out'),
                                  (35,'GAP','类别均值 → 交叉熵','类别平均相似度'),
                                  (70,'SupCon-in','正例均值 → 负对数','SupCon-in')]:
        color=COL[method]
        ax.plot([x,x+28],[48,48],color=color,lw=1.8)
        text(ax,x,43,method,10,color=color,bold=True)
        text(ax,x,35,operation,8)
        text(ax,x,28,sub,7.5,color=GRAY)
    ax.plot([0,100],[21,21],color=RULE,lw=.6)
    text(ax,0,15,'c',10,bold=True);text(ax,4,15,'统一冻结评分',9,bold=True)
    text(ax,4,5,'查询 × 支持图',8.5)
    arrow(ax,24,5,32,5)
    text(ax,35,5,'逐图余弦相似度',8.5)
    arrow(ax,57,5,65,5)
    text(ax,67,5,'身份内均值',8.5)
    arrow(ax,84,5,90,5)
    text(ax,99,5,'排序',8.5,ha='right')
    save(fig,'Fig2_model_scoring')

def clean(ax):
    ax.tick_params(length=3,width=.6,pad=4,color=GRAY)
    for spine in ax.spines.values():spine.set_color(GRAY)

def quantitative():
    """保留全部原始点估计；逐模型点仅显示离散程度，不解释为置信区间。"""
    source=rows('holstein_per_model.csv')
    assert len(source)==15
    fig,axes=plt.subplots(1,2,figsize=(6.25,2.65),sharey=True)
    fig.subplots_adjust(left=.13,right=.975,bottom=.23,top=.81,wspace=.27)
    for ax,metric,letter in zip(axes,['R1','R5'],['a','b']):
        for idx,(method,key) in enumerate([('B','B'),('GAP','GAP'),('SupCon-in','SupCon_in')]):
            vals=np.array([float(r[key+'_'+metric])*100 for r in source]);y=2-idx
            ax.scatter(vals,y+np.linspace(-.13,.13,15),s=12,facecolors='none',edgecolors=COL[method],linewidths=.7)
            ax.scatter(vals.mean(),y+.27,s=27,color=COL[method],marker='D')
            ax.text(vals.mean(),y+.52,f'{vals.mean():.2f}',ha='center',fontsize=8,color=COL[method])
        ax.set(ylim=(-.35,2.75),yticks=[2,1,0],yticklabels=['B','GAP','SupCon-in'],xlim=(40,102),xticks=[40,60,80,100],xlabel='准确率 (%)')
        if metric=='R5':ax.set(xlim=(80,100.8),xticks=[80,85,90,95,100])
        ax.set_title(f'{letter}   '+('Rank-1' if metric=='R1' else 'Rank-5'),loc='left',fontsize=9,weight='bold',pad=9)
        clean(ax)
    fig.text(.54,.035,'空心点：15 个模型；菱形：模型等权均值',ha='center',fontsize=7.5,color=GRAY)
    save(fig,'Fig3_holstein_main')

    fig,axes=plt.subplots(1,2,figsize=(6.25,2.8),sharey=True)
    fig.subplots_adjust(left=.09,right=.98,bottom=.21,top=.78,wspace=.20)
    for ax,n,letter in zip(axes,[64,65],['a','b']):
        data=sorted([r for r in rows('holstein_candidate_k.csv') if r['stratum'].startswith(f'candidates={n},')],key=lambda r:int(r['stratum'].split('=')[-1]))
        for m,key in [('B','B_R1'),('GAP','GAP_R1'),('SupCon-in','SupCon_in_R1')]:
            values=[float(r[key])*100 for r in data]
            ax.plot([1,3,5],values,color=COL[m],marker=MARK[m],ms=4,lw=1.2,label=m)
        ax.set(xticks=[1,3,5],xlim=(.7,5.3),ylim=(40,100),yticks=[40,60,80,100],xlabel='每身份支持图数量 K')
        ax.set_title(f'{letter}   {n} 个候选身份',loc='left',fontsize=9,weight='bold',pad=9)
        clean(ax)
    axes[0].set_ylabel('Rank-1 (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,1.0),ncol=3,columnspacing=2.5,fontsize=8)
    save(fig,'Fig4_holstein_support')

    data=rows('sideview_all_available.csv')
    fig,ax=plt.subplots(figsize=(6.25,2.85))
    fig.subplots_adjust(left=.09,right=.80,bottom=.21,top=.94)
    for m in COL:
        selected=sorted([r for r in data if r['method']==m and r['preprocessing']=='neutral128'],key=lambda r:int(r['K']))
        vals=[float(r['micro_accuracy_pct']) for r in selected]
        assert len(vals)==3
        ax.plot([1,3,5],vals,color=COL[m],marker=MARK[m],ms=4.5,lw=1.4)
        for x,v in zip([1,3,5],vals):ax.annotate(f'{v:.2f}',(x,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8,color=COL[m])
        ax.text(1.03,vals[-1],m,transform=ax.get_yaxis_transform(),va='center',fontsize=8.5,color=COL[m])
    ax.set(xlim=(.7,5.3),ylim=(20,70),yticks=[20,30,40,50,60,70],xticks=[1,3,5],xlabel='每身份支持图数量 K',ylabel='微平均准确率 (%)')
    clean(ax);save(fig,'Fig5_sideview_transfer')

    fig,ax=plt.subplots(figsize=(6.25,2.5))
    fig.subplots_adjust(left=.27,right=.83,bottom=.15,top=.83)
    cmap=LinearSegmentedColormap.from_list('accuracy',['#F0F4F5','#B5CFD5','#76A3AE','#286374'])
    configs=[('baseline','原始 letterbox'),('neutral128','灰背景'),('geomalign_v3','geomalign-v3'),('blacktrim-v1','blacktrim-v1')]
    # 同一K列内比较四种预处理；最高点标记不表示显著性。
    maxima={k:max(float(r['micro_accuracy_pct']) for r in data if r['method']=='B' and int(r['K'])==k and r['preprocessing'] in {c[0] for c in configs}) for k in (1,3,5)}
    for i,(pre,label) in enumerate(configs):
        for j,k in enumerate([1,3,5]):
            match=[r for r in data if r['method']=='B' and r['preprocessing']==pre and int(r['K'])==k]
            assert len(match)==1
            val=float(match[0]['micro_accuracy_pct']);color=cmap((val-45)/20)
            ax.add_patch(Rectangle((j-.48,i-.46),.96,.92,fc=color,ec='none'))
            best=val==maxima[k]
            if best:
                ax.add_patch(Rectangle((j-.46,i-.44),.92,.88,fill=False,ec='#183F4B',lw=1.4))
            ax.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=9,color='white' if val>=60 else INK,weight='bold' if best else 'normal')
    ax.set(xlim=(-.5,2.5),ylim=(3.5,-.5),xticks=[0,1,2],xticklabels=['K = 1','K = 3','K = 5'],yticks=range(4),yticklabels=[x[1] for x in configs])
    ax.tick_params(length=0,pad=9);ax.xaxis.tick_top()
    for s in ax.spines.values():s.set_visible(False)
    # 手动画出矢量色标，避免栅格化色块进入投稿图件。
    for i in range(100):
        ax.add_patch(Rectangle((3.04,-.46+i*.0392),.14,.04,fc=cmap(1-i/99),ec='none',clip_on=False))
    text(ax,3.28,-.45,'65',7.5);text(ax,3.28,3.45,'45',7.5)
    fig.text(.54,.035,'SupCon-out：粗体与描框为列内最高点估计，非显著性标记',ha='center',fontsize=7,color=GRAY)
    save(fig,'Fig6_preprocessing')

if __name__=='__main__':
    protocol();architecture();quantitative()
    print('Six vector figure sets exported.')
