"""PrimeTab 方法图的纯矢量实现。

运行后导出可编辑文字的 SVG、矢量 PDF 和 PNG，均不嵌入位图。
序列、特征块和排名条用于示意方法关系。
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.path import Path as DrawingPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets'
PAPER, INK, BLUE = '#f4f3ef', '#11132b', '#2f36ee'
PALE, MID, SLATE = '#d6e3ff', '#a9beff', '#a8b3ca'
EDGE, PANEL = '#bcc6d7', '#e9eef8'
W, H = 1980, 925
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 3})
fig = plt.figure(figsize=(19.8,9.25),facecolor=PAPER)
ax = fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(H,0))
ax.set_axis_off()


def text(x,y,s,size=26,color=INK,ha='left',bold=False):
    family = 'DejaVu Sans' if s.isascii() else 'PingFang SC'
    weight = ('bold' if bold else 'normal') if s.isascii() else (600 if bold else 400)
    font = FontProperties(family=family,weight=weight)
    return ax.text(x,y,s,fontsize=size*.72,color=color,ha=ha,va='center',
        fontproperties=font,linespacing=1.3,zorder=8)


def rect(x,y,w,h,fill=PAPER,edge=EDGE,r=10,lw=1,z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        facecolor=fill,edgecolor=edge,linewidth=lw,zorder=z))


def line(points,color=INK,lw=1.5):
    ax.plot([p[0] for p in points],[p[1] for p in points],
        color=color,lw=lw,solid_capstyle='round',solid_joinstyle='round',zorder=5)


def arrow(points,color=INK,lw=1.5,head=14):
    path=DrawingPath(points,[DrawingPath.MOVETO]+[DrawingPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',
        mutation_scale=head,color=color,linewidth=lw,joinstyle='round',zorder=6))


def cell(x,y,w,h,color,edge=None):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=color,
        edgecolor=edge or color,linewidth=.55,zorder=4))


def values(n):
    palette=[PALE,MID,SLATE,PAPER]
    return [palette[(i*3+n+i//3)%4] for i in range(10)]


def vector(x,y,colors,w=28,h=24,pitch=33):
    for i,color in enumerate(colors):
        cell(x+i*pitch,y,w,h,color,EDGE)


def sheet(x,y,w=228,h=103,rows=3,target=False,stack=True):
    if stack:
        for dx,dy in [(12,-12),(6,-6)]:
            rect(x+dx,y+dy,w,h,fill=PAPER,edge=EDGE,r=5,lw=.8)
    rect(x,y,w,h,fill=PAPER,edge=EDGE,r=5,lw=.8)
    for row in range(rows):
        for col in range(6):
            color=(MID if target else [PALE,SLATE,PAPER][(col+row)%3])
            cell(x+18+col*27,y+20+row*23,22,17,color,EDGE)
        cell(x+191,y+20+row*23,19,17,BLUE if target else INK)


def rna(x,y,color):
    # A small schematic RNA stem-loop, drawn entirely as editable paths.
    points=[(x+4,y+37),(x+15,y+33),(x+25,y+16),
        (x+26,y+7),(x+17,y-5),(x+8,y-18),
        (x-1,y-30),(x+4,y-44),(x+17,y-43),
        (x+36,y-41),(x+26,y-19),(x+39,y-6),
        (x+43,y-2),(x+48,y-1),(x+53,y-1)]
    codes=[DrawingPath.MOVETO,DrawingPath.LINETO,DrawingPath.LINETO]+[DrawingPath.CURVE4]*12
    ax.add_patch(FancyArrowPatch(path=DrawingPath(points,codes),arrowstyle='-',
        color=color,linewidth=1.9,zorder=4))
    for i in range(4):
        a=x+7+i*5; b=y+36-i*5
        line([(a,b),(a-4,b-6)],color=color,lw=1.3)


def candidate(n,y,color,sequence):
    text(110,y-66,f'候选{n}',29,color,bold=True)
    rna(59,y-24,color)
    for i,base in enumerate(sequence):
        cell(112+i*29,y-35,25,31,PALE if n==1 else '#e0e3ec',EDGE)
        text(124.5+i*29,y-19.5,base,23,ha='center')
    arrow([(240,y+5),(240,y+45)],EDGE,head=11)
    vector(60,y+58,values(n))


def context(n,y,color):
    rect(475,y,875,156,fill='#fafaf8',edge=color,r=11,lw=1.1)
    text(502,y+27,f'参考集{n} + 候选{n}',29,color,bold=True)
    text(1281,y+27,'效率',25,ha='center')
    # Several faint offset row outlines indicate a stack of measured records.
    for dx,dy in [(6,-6),(3,-3)]:
        rect(671+dx,y+56+dy,471,27,PAPER,EDGE,r=2,lw=.6)
    text(502,y+69,'参考记录',25)
    for i,c in enumerate(values(3+n)):
        cell(672+i*48,y+56,40,27,c,EDGE)
    cell(1254,y+56,55,27,MID,EDGE)
    text(502,y+117,f'候选{n}',25,color)
    vector(672,y+104,values(n),w=40,h=27,pitch=48)
    rect(1254,y+104,55,27,PAPER,EDGE,r=2,lw=.7)
    text(1281.5,y+117.5,'?',27,color,ha='center',bold=True)


# Wordmark: restrained size and no subtitle or decorative header band.
prime=text(60,71,'Prime',56,bold=True)
fig.canvas.draw()
right=prime.get_window_extent(fig.canvas.get_renderer()).x1
right=ax.transData.inverted().transform((right,0))[0]
text(right,71,'Tab',56,BLUE,bold=True)

# Left: one edit gives two candidate sequences and identically encoded features.
text(60,205,'目标编辑',30,bold=True)
text(65,265,'… A C',29)
rect(177,240,104,49,PALE,BLUE,r=6,lw=1.1)
text(229,265,'G → C',28,BLUE,ha='center',bold=True)
text(290,265,'T G …',29)
arrow([(219,302),(219,333)],EDGE,head=11)
text(60,361,'候选生成',30,bold=True)
text(60,418,'73 项特征 → 107 维',25)
candidate(1,530,BLUE,'AUGCCUGA')
candidate(2,716,INK,'UACGGAUC')

# Shared measured reference library, with source and target arrows into it.
text(608,139,'来源细胞',28,ha='center',bold=True)
text(926,108,'实测参考库',35,BLUE,ha='center',bold=True)
text(1265,139,'目标细胞',28,ha='center',bold=True)
sheet(494,175,rows=3)
sheet(1148,175,rows=2,target=True)
sheet(812,165,rows=4,h=119)
# Target measurements are present in the shared source/target bank.
for col in range(6):
    cell(830+col*27,165+20+3*23,22,17,MID,EDGE)
cell(1003,254,19,17,BLUE)
arrow([(736,221),(799,221)])
arrow([(1143,221),(1056,221)])
arrow([(926,291),(926,377)],head=14)

# Retrieval is one module with two independent candidate lanes.
rect(430,382,970,478,fill=PANEL,edge='#d5dfef',r=18,lw=.9)
text(915,416,'逐候选参考检索',36,BLUE,ha='center',bold=True)
text(915,457,'每条候选 1000 条参考',25,ha='center')
context(1,490,BLUE)
context(2,676,INK)
# Candidate feature vectors align exactly with the unmeasured query rows.
arrow([(391,600),(471,600)],BLUE)
arrow([(391,786),(471,786)],INK)

# The same model weights are reused; each input is a separate context/query pair.
rect(1460,523,180,276,fill=INK,edge=INK,r=13,lw=1)
text(1550,627,'LDM',49,PAPER,ha='center',bold=True)
line([(1491,660),(1609,660)],MID,lw=1)
text(1550,711,'权重固定',30,MID,ha='center')
arrow([(1354,560),(1456,560)],BLUE)
arrow([(1354,746),(1456,746)],INK)

# Output is the ranking for this edit, with the candidate colours preserved.
rect(1696,507,228,289,fill='#eaeef3',edge='#eaeef3',r=14)
text(1810,543,'同一编辑内排名',27,ha='center',bold=True)
for rank,n,y,width,color in [(1,2,614,121,INK),(2,1,718,80,BLUE)]:
    text(1726,y,str(rank),29,INK,bold=True)
    text(1770,y,f'候选{n}',29,color,bold=True)
    cell(1770,y+30,width,11,color)
arrow([(1644,661),(1692,661)],head=14)

OUT.mkdir(parents=True,exist_ok=True)
for suffix in ['svg','pdf','png']:
    path=OUT/f'method-framework.{suffix}'
    fig.savefig(path,dpi=200,facecolor=PAPER,metadata={'Creator':'PrimeTab'})
    if suffix == 'svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    print(path)
plt.close(fig)
