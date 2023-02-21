<!DOCTYPE html>
<html>
<head>
	<meta http-equiv="content-type" content="text/html; charset=utf-8"/>
	
</head>
<body lang="en-US" text="#000000" link="#000080" vlink="#800000" dir="ltr">
<p class="western" align="center" style="margin-bottom: 0.14in">
<font size="4" style="font-size: 14pt"><u><b>Code Sample – Fitting
an Ellipse to  points on a grid</b></u></font></p>
<p class="western" align="center" style="margin-bottom: 0.14in"><u><b>Constantine
A. Raftopoulos, PhD</b></u></p>
<p class="western" align="center" style="margin-bottom: 0.14in"><font color="#000080"><u><a href="mailto:raftop@cs.ucla.edu"><b>raftop@cs.ucla.edu</b></a></u></font></p>
<p class="western" align="center" style="margin-bottom: 0.14in"><br/>
<br/>

</p>
<p class="western" align="justify" style="margin-bottom: 0.14in">This
program fits an Ellipse to chosen points on a
grid.  The system starts with a single window containing a 20x20 grid
of square points similar to this:</p>
<p class="western" align="center" style="margin-bottom: 0.14in"><img src="https://user-images.githubusercontent.com/40480140/220257865-414cece8-7e85-4f5b-8350-10a084f98c82.png" name="Image1" align="bottom" width="163" height="163" border="0"/>
</p>
<p class="western" align="justify" style="margin-bottom: 0.14in">All
points start out gray. The user is allowed to toggle the individual
points on the grid on and off. A button is added at the bottom of the
window. When the user clicks this button, an ellipse that best fits
the highlighted points is generated by means of  an iterative,
geometric least squares-based algorithm that does not rely on an
external library or code to find the best fit. Such a fit is shown
below for a circle.</p>
<p class="western" align="center" style="margin-bottom: 0.14in"><img src="https://user-images.githubusercontent.com/40480140/220258164-d9a2b4b0-4c12-4a21-8e15-7e1fcdca0880.png" name="Image2" align="bottom" width="181" height="179" border="0"/>
</p>
<p class="western" align="justify" style="margin-bottom: 0.14in"><br/>
<br/>

</p>
<p class="western" align="justify" style="margin-bottom: 0.14in"><font color="#000000"><font size="2" style="font-size: 11pt"><span lang="en-US"><span style="text-decoration: none">This
code fits an ellipse (least square sense) to the chosen by the user
points on the grid using a guided random walk. The source code is in
Python 3.8. Basic packages are used: numpy for vector matrix
operations, random generators, pdist and cdist from distances package
for fast Euclidean distance calculation. For Uix, Kivy is used for
better integration with Python but also capability for mobile
devices. PyInstaller was used to package for windows and linux
executables.</span></span></font></font></p>
<p class="western" align="justify" style="margin-bottom: 0.14in; text-decoration: none">
<font size="2" style="font-size: 11pt">An ellipse is uniquely defined
by 5 parameters: Center coords, 2 axis lengths and a slope. A random
walk is thus initiated in a 5 dimensional space each point in this
space representing a different ellipse. The initial parameter
estimation is performed by means of mean point coords for the center,
max of in-between points distances for the bigger axis and the mean
of the same distances for the smaller axis. For estimating the
ellipse slope a PCA treatment is used, where the dominant eigenvector
of the covariance of the points coords matrix indicates the direction
of maximum variance of point projections therefore a good estimate of
the initial ellipse slope. </font>
</p>
<p class="western" align="justify" style="margin-bottom: 0.14in"><font color="#000000"><font size="2" style="font-size: 11pt"><span lang="en-US"><span style="text-decoration: none">The
mean square error is calculated in each iteration by adding each
point's distances to the foci and taking the squared residual from
the large axis. This is because for a point on any ellipse the sum of
its distances to the foci is equal to the large axis. The process
iterates 100 times at max. Every time the button is pressed it
further improves by trying 100 more times (or at least trying). </span></span></font></font>
</p>
</body>
</html>
