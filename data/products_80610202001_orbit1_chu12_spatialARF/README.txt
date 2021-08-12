This directory contains everything necessary to run the gridded ARF analysis.

We are using data from the first orbit, restricting ourselves to the GTI for CHU12.

Event lists for FPMA and FPMB can be found in the event_lists/ subdirectory.

The ARF has been generated in a 13 x 13 grid of cells with 1 arcmin width and height.
Note: The way we will compute our expected signal will require reintegrating the surface brightness 
over the spatial size of a cell, then multiplying by the ARF. Per Brian:
"If your current fluence model has units of ph / cm2 / sec in bin i,j
 then you’ve accounted for the size of the bin i,j when converting your surface brightness template
 (which should have units of something like ph / cm2 / sec / arcsec2 or something)
 to fluence because you’ve integrated over the sky area in bin i,j."

The centers of these cells can be found in box_centers.txt.

The exposure for this GTI for each FPM can be found in exposures.txt.
