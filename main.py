#_________________________________________________________________________________________
# Package name		Google Trends Rescaler
# Author			József Vass <jozsef.vass@outlook.com, jzsfvss@gmail.com>
# Language			Python
# License			GNU General Public License v3.0
# Stage				proof of concept
# Version			Sep. 20, 2022
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
#_________________________________________________________________________________________
# Initialization
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
import math as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as spi

import time
import pytz
from datetime import timezone
from datetime import datetime

import pyinputplus as pyinput

# Setting the variables:
dataloc = './Data'
plotloc = './Plots';

year1 = 2017;
year2 = 2022;

tz = pytz.timezone('UTC')
#_________________________________________________________________________________________
# Importing the data
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
txt = '\nUse dataset (1) DS1, (2) DS2: '
opt = pyinput.inputInt(prompt=txt, min=1, max=2)

if opt == 1:
	opts = 'DS1'
else:
	opts = 'DS2'

print('\nImporting the data...', end="")
t1 = time.time()

if opt == 1:
	datamr = pd.read_csv('./Data/DS1/monthly_data.csv')
	cols = ['time_month', 'value_month']
	datam = datamr.loc[:, cols].to_numpy()

	datawr = pd.read_csv('./Data/DS1/weekly_data.csv')
	cols = ['time_week', 'value_week']
	dataw = datawr.loc[:, cols].to_numpy()

	datahr = pd.read_csv('./Data/DS1/hourly_data.csv')
	cols = ['time_hour', 'value_hour']
	datah = datahr.loc[:, cols].to_numpy()
else:
	datamr = pd.read_csv('./Data/DS2/monthly_data.csv')
	datam11 = list(datamr.iloc[1:,0].index)
	datam12 = [tz.localize(datetime.strptime(ts, '%Y-%m')).timestamp() for ts in datam11]
	datam1 = np.array(datam12, dtype='float64')
	datam2 = datamr.iloc[1:, 0].to_numpy(dtype='float64')
	datam = np.c_[datam1, datam2]

	dataw = np.empty((0, 2), dtype='float64')

	for yr in range(year1, year2 + 1):
		datawri = pd.read_csv('./Data/DS2/weekly_data_' + str(yr) + '.csv')
		datawi11 = list(datawri.iloc[1:,0].index)
		datawi12 = [tz.localize(datetime.strptime(ts, '%Y-%m-%d')).timestamp() for ts in datawi11]
		datawi1 = np.array(datawi12, dtype='float64')
		datawi2 = datawri.iloc[1:, 0].to_numpy(dtype='float64')
		datawi = np.c_[datawi1, datawi2]
		dataw = np.concatenate((dataw, datawi))

	datahr = pd.read_csv('./Data/DS1/hourly_data.csv')
	cols = ['time_hour', 'value_hour']
	datah = datahr.loc[:, cols].to_numpy(dtype='float64')

t2 = time.time()
print('done in {:.3f} secs.'.format(t2 - t1))
#_________________________________________________________________________________________
# Realigning the data
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
print('\nRealigning the data...', end="")
t1 = time.time()

# Length of all months between 01/2017 - 08/2022:
mlens = np.empty((0, 2), dtype='float64')
mlens0 = np.array([ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ], dtype='float64') # length of each month
mlens = mlens0

for i in range(1, 4+1):
    mlens = np.concatenate((mlens, mlens0))

mlens = np.concatenate((mlens, mlens0[0:8]))
mlens[3*12 + 2 - 1] = 29 # Accounting for the leap year 2020.

# Associate each value with the midpoint of the interval it represents:
datam2 = np.c_[datam[:, 0] + (24*60*60)*mlens/2, datam[:, 1]]
dataw2 = np.c_[dataw[:, 0] + (24*60*60)*3.5, dataw[:, 1]]
datah2 = np.c_[datah[:, 0] + (24*60*60)*(0.5/24), datah[:, 1]]

t2 = time.time()
print('done in {:.3f} secs.'.format(t2 - t1))
#_________________________________________________________________________________________
# Rescaling the data
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
print('\nRescaling the data...', end="")
t1 = time.time()

# Inter/extra-polate the monthly data at the weekly timestamps:
# datam2int = spi.interp1d(datam2[:, 0], datam2[:, 1], kind = 'cubic', fill_value='extrapolate')(dataw2[:, 0])
# datam2int = spi.CubicSpline(datam2[:, 0], datam2[:, 1], extrapolate=True)(dataw2[:, 0])
# datam2int = spi.Akima1DInterpolator(datam2[:, 0], datam2[:, 1])(dataw2[:, 0])
datam2int = spi.PchipInterpolator(datam2[:, 0], datam2[:, 1], extrapolate=True)(dataw2[:, 0])

# Finding the weekly data scaling coefficients over each year:
dataw3 = np.empty((0, 2), dtype='float64')
YR = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y")

for yr in range(year1, year2 + 1):
	booi = [ (str(yr) == YR(ts)) for ts in list(dataw[:, 0]) ] # weekly data in this yr
	nw = sum(booi)

	datam2i = datam2int[booi]
	dataw2i = dataw2[booi, 1]

	mtx = np.c_[ dataw2i, np.ones(nw) ];
	scswi = np.linalg.lstsq(mtx, datam2i, rcond=None)
	scswi = scswi[0]

	dataw3i = np.c_[dataw2[booi, 0], np.matmul(mtx, scswi) ]
	dataw3 = np.concatenate((dataw3, dataw3i))

# Inter/extra-polate the rescaled weekly data at the hourly timestamps:
dataw3int = spi.PchipInterpolator(dataw3[:, 0], dataw3[:, 1], extrapolate=True)(datah2[:, 0]);

# Finding the hourly data scaling coefficients over each week:
datah3 = np.empty((0, 2), dtype='float64')
wints = np.append(dataw[:, 0], dataw[-1, 0] + 7*24*60*60) # week start- and endpoints
nw = dataw.shape[0]

for wk in range(nw):
	booi = (datah[:, 0] >= wints[wk]) & (datah[:, 0] < wints[wk + 1])
	n = sum(booi)

	dataw3i = dataw3int[booi]; # weekly interpolated to hourly
	datah2i = datah2[booi, 1]; # hourly

	mtx = np.c_[ datah2i, np.ones(n) ];
	scshi = np.linalg.lstsq(mtx, dataw3i, rcond=None)
	scshi = scshi[0]

	datah3i = np.c_[datah2[booi, 0], np.matmul(mtx, scshi)]
	datah3 = np.concatenate((datah3, datah3i))

t2 = time.time()
print('done in {:.3f} secs.'.format(t2 - t1))
#_________________________________________________________________________________________
# Exporting the data
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
print('\nExporting the data...', end="")
t1 = time.time()

datahr3 = datahr
datahr3['value_hour'] = datah3[:, 1]

fpath = './Data/hourly_data_rescaled_' + opts + '.csv'
datahr3.to_csv(fpath, index=False)

t2 = time.time()
print('done in {:.3f} secs.'.format(t2 - t1))

print('Path: ' + fpath)
#_________________________________________________________________________________________
# Plotting the data
#‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
print('\nPlotting the data...', end="")
t1 = time.time()

lnw = 0.5 # default line width
ftsz = 6 # default font size

fig = plt.figure()
ax = plt.subplot(111)
plt.rcParams['figure.dpi'] = 800

plt.plot(datam2[:, 0], datam2[:, 1], 'b.', ms = 3*lnw)
plt.plot(dataw2[:, 0], datam2int, 'b', linewidth = str(lnw))

plt.plot(dataw2[:, 0], dataw2[:, 1], 'r', linewidth = str(lnw))
plt.plot(dataw3[:, 0], dataw3[:, 1], 'm', linewidth = str(1.5*lnw));
plt.plot(datah3[:, 0], datah3[:, 1], 'k', linewidth = str(0.5*lnw));

for yr in range(year1, year2 + 1):
	plt.axvline(x = tz.localize(datetime.strptime(str(yr), '%Y')).timestamp(), color = '#00FF00', linestyle = '-', linewidth = 0.5*lnw)

plt.xlabel('Time', fontsize = ftsz - 1)
plt.ylabel('Value', fontsize = ftsz - 1)
tit = 'Google Trends data for Bitcoin ' + str(year1) + ' - ' + str(year2) + ' (' + opts + ')'
plt.title(tit, fontsize = ftsz)
# fig.suptitle(tit, fontsize = ftsz + 1)

xticksv = list(datam[[12*x for x in range(6)], 0])
xtickss = [ str(yr) for yr in range(year1, year2 + 1) ]
plt.xticks(ticks = xticksv, labels = xtickss, fontsize = ftsz - 2)
yticksv = range(0, 110, 10)
ytickss = [ str(yt) for yt in yticksv ]
plt.yticks(ticks = yticksv, labels = ytickss, fontsize = ftsz - 2)
ax.xaxis.set_tick_params(width = lnw, length = 2)
ax.yaxis.set_tick_params(width = lnw, length = 2)
plt.grid(axis = 'y', linewidth = 0.5*lnw)

legs = [ 'monthly realigned', 'monthly realigned interpolated', 'weekly realigned', 'weekly realigned rescaled', 'hourly realigned rescaled' ]
hleg = plt.legend(legs, loc = 'upper right', prop = {'size': 3}, framealpha = 0.9, fancybox = False)
hleg.get_frame().set_linewidth(0.5*lnw)
hleg.get_frame().set_edgecolor('k')

ax.set_aspect(1)
ax.set_ylim(bottom = 0)
ax.spines.left.set_linewidth(lnw)
ax.spines.bottom.set_linewidth(lnw)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left)/(y_low - y_high))*0.5)

plt.show()

fpath = './Plots/fig_data_rescaled_' + opts + '.jpg'
fig.savefig(fpath, dpi = 800, bbox_inches = 'tight', pad_inches = 0.05)

t2 = time.time()
print('done in {:.3f} secs.'.format(t2 - t1))

print('Path: ' + fpath)