#!/usr/bin/gnuplot -persist
#
#    
#    	G N U P L O T
#    	Version 5.4 patchlevel 2    last modified 2021-06-01 
#    
#    	Copyright (C) 1986-1993, 1998, 2004, 2007-2021
#    	Thomas Williams, Colin Kelley and many others
#    
#    	gnuplot home:     http://www.gnuplot.info
#    	faq, bugs, etc:   type "help FAQ"
#    	immediate help:   type "help"  (plot window: hit 'h')
# set terminal qt 0 size 900, 600 font "Helvetica,14" enhanced
# set output
unset clip points
set clip one
unset clip two
unset clip radial
set errorbars front 1.000000 
set border 31 front lt black linewidth 1.500 dashtype solid
set zdata 
set ydata 
set xdata 
set y2data 
set x2data 
set boxwidth
set boxdepth 0
set style fill  empty border
set style rectangle back fc  bgnd fillstyle   solid 1.00 border lt -1
set style circle radius graph 0.02 
set style ellipse size graph 0.05, 0.03 angle 0 units xy
set dummy x, y
set format x "%.2g" 
set format y "%.2f" 
set format x2 "% h" 
set format y2 "% h" 
set format z "% h" 
set format cb "% h" 
set format r "% h" 
set ttics format "% h"
set timefmt "%d/%m/%y,%H:%M"
set angles radians
set tics back
set grid nopolar
set grid xtics nomxtics ytics nomytics noztics nomztics nortics nomrtics \
 nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics
set grid layerdefault   lt 0 linecolor rgb "#bbbbbb"  linewidth 1.000,  lt 0 linecolor rgb "#bbbbbb"  linewidth 1.000
unset raxis
set theta counterclockwise right
set style parallel front  lt black linewidth 2.000 dashtype solid
set key notitle
set key fixed right top vertical Right noreverse enhanced autotitle box lt black linewidth 1.000 dashtype solid
set key noinvert samplen 2 spacing 1.2 width 0 height 0 
set key maxcolumns 0 maxrows 0
set key noopaque
unset label
unset arrow
set style increment userstyles
unset style line
set style line 1  linecolor rgb "#1f77b4"  linewidth 2.000 dashtype solid pointtype 7 pointsize 1.500
set style line 2  linecolor rgb "#ff7f0e"  linewidth 2.000 dashtype solid pointtype 5 pointsize 1.500
set style line 3  linecolor rgb "#2ca02c"  linewidth 2.000 dashtype solid pointtype 9 pointsize 1.500
set style line 4  linecolor rgb "#d62728"  linewidth 2.000 dashtype solid pointtype 13 pointsize 1.500
set style line 5  linecolor rgb "#9467bd"  linewidth 2.000 dashtype solid pointtype 11 pointsize 1.500
set style line 6  linecolor rgb "#8c564b"  linewidth 2.000 dashtype solid pointtype 3 pointsize 1.500
unset style arrow
set style histogram clustered gap 2 title textcolor lt -1
unset object
unset walls
set style textbox  transparent margins  1.0,  1.0 border  lt -1 linewidth  1.0
set offsets 0, 0, 0, 0
set pointsize 1
set pointintervalbox 1
set encoding iso_8859_1
unset polar
unset parametric
unset spiderplot
unset decimalsign
unset micro
unset minussign
set view 60, 30, 1, 1
set view azimuth 0
set rgbmax 255
set samples 100, 100
set isosamples 10, 10
set surface 
unset contour
set cntrlabel  format '%8.3g' font '' start 5 interval 20
set mapping cartesian
set datafile separator whitespace
set datafile commentschars '#!%'
set datafile nocolumnheaders
unset hidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels 5
set cntrparam levels auto
set cntrparam firstlinetype 0 unsorted
set cntrparam points 5
set size ratio 0 1,1
set origin 0,0
set style data lines
set style function lines
unset xzeroaxis
unset yzeroaxis
unset zzeroaxis
unset x2zeroaxis
unset y2zeroaxis
set xyplane relative 0.5
set tics scale  1, 0.5, 1, 1, 1
set mxtics default
set mytics default
set mztics default
set mx2tics default
set my2tics default
set mcbtics default
set mrtics default
set nomttics
set xtics border out scale 0.75,0.375 mirror norotate  autojustify
set xtics  norangelimit autofreq 
set ytics border out scale 0.75,0.375 mirror norotate  autojustify
set ytics  norangelimit autofreq 
set ztics border out scale 0.75,0.375 nomirror norotate  autojustify
set ztics  norangelimit autofreq 
unset x2tics
unset y2tics
set cbtics border out scale 0.75,0.375 mirror norotate  autojustify
set cbtics  norangelimit autofreq 
set rtics axis out scale 0.75,0.375 nomirror norotate  autojustify
set rtics  norangelimit autofreq 
unset ttics
set title "" 
set title  font ",18" textcolor lt -1 norotate
set timestamp bottom 
set timestamp "" 
set timestamp  font "" textcolor lt -1 norotate
set trange [ * : * ] noreverse nowriteback
set urange [ * : * ] noreverse nowriteback
set vrange [ * : * ] noreverse nowriteback
set xlabel "Nearest neighbour distance ({\305})" 
set xlabel  font ",16" textcolor lt -1 norotate
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ 1.46828 : 1.64284 ] noreverse writeback
set x2range [ 1.70458 : 1.85575 ] noreverse writeback
set ylabel "energy [eV/atom]" 
set ylabel  font ",16" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ -7.97363 : -7.58572 ] noreverse writeback
set y2range [ * : * ] noreverse writeback
set zlabel "" 
set zlabel  font "" textcolor lt -1 norotate
set zrange [ * : * ] noreverse writeback
set cblabel "" 
set cblabel  font "" textcolor lt -1 rotate
set cbrange [ * : * ] noreverse writeback
set rlabel "" 
set rlabel  font "" textcolor lt -1 norotate
set rrange [ * : * ] noreverse writeback
unset logscale
unset jitter
set zero 1e-08
set lmargin  10
set bmargin  5
set rmargin  10
set tmargin  5
set locale "en_US.UTF-8"
set pm3d explicit at s
set pm3d scansautomatic
set pm3d interpolate 1,1 flush begin noftriangles noborder corners2color mean
set pm3d clip z 
set pm3d nolighting
set palette positive nops_allcF maxcolors 0 gamma 1.5 color model RGB 
set palette rgbformulae 7, 5, 15
set colorbox default
set colorbox vertical origin screen 0.9, 0.2 size screen 0.05, 0.6 front  noinvert bdefault
set style boxplot candles range  1.50 outliers pt 7 separation 1 labels auto unsorted
set loadpath "/home/gabri/bin" 
set fontpath
set psdir
set fit brief errorvariables nocovariancevariables errorscaling prescale nowrap v5
round(x) = x - floor(x) < 0.5 ? floor(x) : ceil(x)
round2(x, n) = round(x*10**n)*10.0**(-n)
bin(x,s) = s*floor(x/s)
echo(x) = x
GNUTERM = "qt"
I = {0.0, 1.0}
VoxelDistance = 0.0
pdfplot = "set term push; set term pdfcairo enhanced color rounded size 8,6 lw 1.5 font 'Arial,28'; set border lw 1.5 front; set out 'temp.pdf' ; replot; set out; set term pop"
epsplot = "set term push; set term epscairo enhanced color rounded size 8,6 lw 1.5 font 'Arial,28'; set border lw 1.5 front; set out 'temp.eps' ; replot; set out; set term pop"
pngplot = "set term push; set term pngcairo enhanced color rounded size 1600,1200 lw 1.5 font 'Arial,28'; set border lw 1.5 front; set out 'temp.png' ; replot; set out; set term pop"
matlab = "set palette positive nops_allcF maxcolors 0 gamma 1.5 color model RGB;           set palette defined  (0  0.0 0.0 0.5,                                 1  0.0 0.0 1.0,                                 2  0.0 0.5 1.0,                                 3  0.0 1.0 1.0,                                 4  0.5 1.0 0.5,                                 5  1.0 1.0 0.0,                                 6  1.0 0.5 0.0,                                 7  1.0 0.0 0.0,                                 8  0.5 0.0 0.0 )"
rainbow = "set palette positive nops_allcF maxcolors 0 gamma 1.5 color model HSV;            set palette defined ( 0 0 1 1, 1 0.85 1 1 )"
mandelbrot = "set sample 500; set isosample 500; set pm3d map;               complex(x,y) = x*{1,0}+y*{0,1};               mandel(x,y,z,n) = (abs(z)>2.0 || n>=100) ? n : mandel(x,y,z*z+complex(x,y),n+1);               splot mandel(x,y,{0,0},0) w pm3d not"
## Last datafile plotted: "/home/gabri/Thesis/PROJECT/DATASETS/C/GAP2020/config_types/Diamond/eperatom_cubiclp.txt"
p'model_penergy.out' u ($1*1.7320508075688772/4.0):($2/2.0) w p t "model" ,'/home/gabri/Thesis/PROJECT/DATASETS/C/GAP2020/config_types/Diamond/eperatom_cubiclp.txt' u ($1*1.7320508075688772/4.0):2 w p t "reference"
#    EOF
