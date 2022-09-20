## Google Trends Rescaler

<table border="0"><tr>
<td width="125px">
<b>Author</b><br />
<b>Language</b><br />
<b>License</b><br />
<b>Stage</b><br />
<b>Version</b>
</td>
<td>
József Vass &lt;jozsef.vass@outlook.com, jzsfvss@gmail.com&gt;<br />
Python<br />
GNU General Public License v3.0<br />
proof of concept<br />
Sep. 20, 2022
</td>
</tr></table>

---

#### Package Contents

<table border="0"><tr>
<td width="260px">
./main.py<br />
./Data<br />
./Data/DS1/monthly_data.csv<br />
./Data/DS1/weekly_data.csv<br />
./Data/DS1/hourly_data.csv<br />
./Data/DS2/*.csv<br />
./Plots
</td>
<td>
Main program.<br />
Raw and output data folder (bitcoin search term, worldwide).<br />
Monthly data points – fully-consistent.<br />
Weekly data points – consistent per year (flawed for 2017).<br />
Hourly data points – consistent per week.<br />
Like DS1, but the weekly data has not been aggregated (2017 ok).<br />
Folder for output plots.
</td>
</tr></table>

---

#### Instructions

1. Download and unzip the package to a convenient location.

2. Open Spyder and set the working directory to the unzipped package folder.

3. Run `pip install P` for the following package names P: math, numpy, scipy, pandas, matplotlib, time, datetime, pytz, pyinputplus.

4. Execute the program with `runfile('main.py')`.

5. Find the output figure and data files in the Plots and Data folders respectively.

---

#### Problem

Google Trends rescales search volume data by an undisclosed (and likely unreliable) method to the interval [0, 100]. (Eg. outliers may be removed or set to 100, and values could be rounded, making the data crude.) On top of this, the data is not consistent in scaling if different frequencies / time intervals are chosen.

So the problem to be solved here is to make the hourly data fully-consistent in scaling across all years 2017–2022 in the data, or at least as much as possible.

---

#### Solution

My solution has two main steps, but the technique is the same: interpolate the higher-level data at the timestamps of the lower-level data (eg. monthly to weekly), then determine the rescaling affine transformation (an assumption) that gives the closest fit between the rescaled lower-level data and the interpolated higher-level data. Do this between both monthly and weekly, as well as weekly and hourly data – passing data consistency down from monthly to hourly.

<u>Steps</u>:
1. *Realignment*: move all data points to the midpoint of their respective time intervals, which is most representative of that value.
2. *Interpolation*: The monthly data is provided by Google as fully-consistent across all years. We interpolate it at the realigned timestamps of the weekly data. I used the `scipy.interpolate.PchipInterpolator` function (Piecewise Cubic Hermite Interpolating Polynomial – it avoids overshoots and can accurately interpolate flat parts), but included other interpolators as well in my comments on the code.
3. *Rescaling*: The weekly data is only self-consistent within each year. So taking the data for each year, we set up the least-squares matrix equation:<br />
&emsp;&emsp;[ weekly_data, ones ] &times; coefficients = interpolated_monthly_data<br />
and solve for the coefficients (a, b) using `numpy.linalg.lstsq`. Then compute:<br />
&emsp;&emsp;rescaled_weekly_data = a &times; weekly_data + b.
4. We take the same approach for the hourly data within each week. We interpolate the rescaled weekly data (consistent over all years) at the timestamps of the hourly data, then for each week we compute the scaling coefficients, rescale the hourly data, then concatenate it for all weeks.
5. Re-associate again the resulting hourly values with the start of each hour, then export the data.

The (a, b) coefficients could be determined for various metrics (error functions) by optimization, but here (for the sake of speed) I used the standard Euclidean distance for which the solution is explicitly given by least-squares regression.